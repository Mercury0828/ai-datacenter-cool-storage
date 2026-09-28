"""Heat-rejection component models for the HOTLOOP chain (guide §4; parameters and provenance in model/params.py).

Units: heat in MW per MW of design heat on the path ("per-unit"), capacity rates in MW/K per MW, temperatures degC,
pressure Pa. All functions are vectorized over hours.

Chain (P1): rack coolant (TCS) <- CDU <- facility water (FWS) <- plate HX <- glycol loop <- dry cooler <- outdoor air.
For fixed flows every exchanger is linear, so the facility-water supply temperature at duty q is

    FWS = T_air + q * A,   A = R - 1/C_f,   R = 1/(eps_hx*Cmin_hx) + 1/(eps_dc*Cmin_dc) - 1/C_g        (1)

(derivation: DESIGN_DECISIONS DD-006). Series trim chiller: when the chain cannot hold FWS <= FWS_max, the chain
still processes return water at FWR = FWS_max + q/C_f and rejects Q_dc = (FWR - T_air)/R; the chiller removes
q - Q_dc. No fixed dry-bulb threshold is used anywhere (N4).
"""
import numpy as np

from . import psychro as ps

RHO_REF = 101325.0 / (287.042 * (25.0 + 273.15))  # air density at the dry-cooler rating point (25 degC, sea level)


# ---------------------------------------------------------------- heat-exchanger effectiveness
def eps_counterflow(ntu, cr):
    ntu, cr = np.broadcast_arrays(np.asarray(ntu, float), np.asarray(cr, float))
    out = np.empty(ntu.shape)
    one = np.abs(cr - 1.0) < 1e-9
    out[one] = ntu[one] / (1.0 + ntu[one])
    e = np.exp(-ntu[~one] * (1.0 - cr[~one]))
    out[~one] = (1.0 - e) / (1.0 - cr[~one] * e)
    return out


def eps_crossflow_unmixed(ntu, cr):
    """Both fluids unmixed (standard approximation, Incropera eq. 11.32)."""
    ntu = np.maximum(np.asarray(ntu, float), 1e-12)
    cr = np.maximum(np.asarray(cr, float), 1e-12)
    return 1.0 - np.exp((ntu**0.22 / cr) * (np.exp(-cr * ntu**0.78) - 1.0))


def ntu_for_eps_crossflow(eps, cr):
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if eps_crossflow_unmixed(mid, cr) > eps:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def air_density_dry(t, p):
    return p / (287.042 * (t + 273.15))


# ---------------------------------------------------------------- dry-cooler + HX chain
class DryChain:
    """Dry cooler of size k (multiples of the reference module) + plate HX, per MW of design heat on the path."""

    def __init__(self, P, k, fws_max):
        self.P, self.k, self.fws_max = P, float(k), float(fws_max)
        self.C_f = 1.0 / P["dT_fw"]
        self.C_g = 1.0 / P["dT_glycol"]
        # plate HX sized for its design approach at design duty (balanced counterflow if C_f == C_g)
        cmin, cmax = min(self.C_f, self.C_g), max(self.C_f, self.C_g)
        self.ua_hx = self._ua_hx(P, cmin, cmax)
        self.eps_hx = float(eps_counterflow(self.ua_hx / cmin, cmin / cmax))
        self.cmin_hx = cmin
        # reference dry-cooler module (k = 1) from the rating point (Thermokey standard conditions, PR-M1 M1-i-b)
        c_air_ref = 1.0 / P["dc_ref_air_dT"]
        c_gly_rating = 1.0 / (P["dc_ref_itd"] - P["dc_ref_approach"])  # rating glycol range = ITD - approach
        cmin_r, cmax_r = min(c_air_ref, c_gly_rating), max(c_air_ref, c_gly_rating)
        eps_rating = 1.0 / (cmin_r * P["dc_ref_itd"])
        self.ua_ref = ntu_for_eps_crossflow(eps_rating, cmin_r / cmax_r) * cmin_r
        self.c_air_ref = c_air_ref

    @staticmethod
    def _ua_hx(P, cmin, cmax):
        # approach at design duty q = 1: dT = (1/cmin)(1/eps - 1)  ->  eps = 1/(1 + cmin*dT)
        eps = 1.0 / (1.0 + cmin * P["dT_HX_design"])
        cr = cmin / cmax
        lo, hi = 1e-6, 200.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(eps_counterflow(mid, cr)) > eps:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi) * cmin

    def resistance(self, s, rho):
        """R [K/MW] at fan speed fraction s and air density rho (eq. 1)."""
        flow = np.asarray(s, float) * rho / RHO_REF
        c_air = self.k * self.c_air_ref * flow
        ua = self.k * self.ua_ref * np.power(np.maximum(flow, 1e-9), self.P["dc_ua_air_exp"])
        cmin = np.minimum(c_air, self.C_g)
        cmax = np.maximum(c_air, self.C_g)
        eps_dc = eps_crossflow_unmixed(ua / cmin, cmin / cmax)
        return 1.0 / (self.eps_hx * self.cmin_hx) + 1.0 / (eps_dc * cmin) - 1.0 / self.C_g

    def A(self, s, rho):
        return self.resistance(s, rho) - 1.0 / self.C_f

    def q_max(self, t_air, p, s=1.0):
        """Largest duty [per-unit] the chain can reject while holding FWS <= fws_max (0 if t_air >= fws_max)."""
        rho = air_density_dry(t_air, p)
        return np.maximum(self.fws_max - t_air, 0.0) / self.A(s, rho)

    def trim_need(self, q, t_air, p):
        """Series-trim duty needed at full fan to hold FWS = fws_max (>= 0)."""
        rho = air_density_dry(t_air, p)
        R = self.resistance(1.0, rho)
        q_dc = (self.fws_max + q / self.C_f - t_air) / R
        return np.maximum(q - q_dc, 0.0)

    def min_speed(self, q, t_air, p, iters=14):
        """Smallest fan speed in [s_min, 1] holding FWS <= fws_max at duty q (1.0 where even full speed fails).
        Bisection runs only on hours where the minimum speed is not already sufficient."""
        q, t_air, p = np.broadcast_arrays(np.asarray(q, float), np.asarray(t_air, float), np.asarray(p, float))
        rho = air_density_dry(t_air, p)
        smin = self.P["dc_min_speed"]
        out = np.full(t_air.shape, smin)
        idx = ~(t_air + q * self.A(smin, rho) <= self.fws_max)
        if idx.any():
            qq, tt, rr = q[idx], t_air[idx], rho[idx]
            lo = np.full(tt.shape, smin)
            hi = np.ones_like(lo)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                ok = tt + qq * self.A(mid, rr) <= self.fws_max
                hi = np.where(ok, mid, hi)
                lo = np.where(ok, lo, mid)
            out[idx] = hi
        return out

    def fan_power(self, s):
        return self.k * self.P["dc_fan_frac_ref"] * np.asarray(s, float) ** 3

    def duty_fixed_inlet(self, t_air, p, fwr, s=1.0):
        """Chain duty [per-unit] with facility inlet fixed at fwr and fixed flows (outlet may go below the limit)."""
        rho = air_density_dry(t_air, p)
        return np.maximum(fwr - t_air, 0.0) / self.resistance(s, rho)

    def min_speed_duty(self, qd, t_air, p, fwr, iters=14):
        """Smallest fan speed in [s_min, 1] with duty_fixed_inlet >= qd (1.0 where full speed is short)."""
        qd, t_air, p = np.broadcast_arrays(np.asarray(qd, float), np.asarray(t_air, float), np.asarray(p, float))
        rho = air_density_dry(t_air, p)
        smin = self.P["dc_min_speed"]
        out = np.full(t_air.shape, smin)
        idx = ~((fwr - t_air) / self.resistance(smin, rho) >= qd)
        if idx.any():
            qq, tt, rr = qd[idx], t_air[idx], rho[idx]
            lo = np.full(tt.shape, smin)
            hi = np.ones_like(lo)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                ok = (fwr - tt) / self.resistance(mid, rr) >= qq
                hi = np.where(ok, mid, hi)
                lo = np.where(ok, lo, mid)
            out[idx] = hi
        return out


    def dc_approach(self, q=1.0, t_air=35.0, p=101325.0, s=1.0):
        """Dry-cooler-only approach (glycol leaving - entering air) at duty q [K]; P5 floor applies to this."""
        rho = air_density_dry(t_air, p)
        flow = s * rho / RHO_REF
        c_air = self.k * self.c_air_ref * flow
        ua = self.k * self.ua_ref * flow ** self.P["dc_ua_air_exp"]
        cmin, cmax = min(c_air, self.C_g), max(c_air, self.C_g)
        eps_dc = float(eps_crossflow_unmixed(ua / cmin, cmin / cmax))
        return q * (1.0 / (eps_dc * cmin) - 1.0 / self.C_g)

    def design_approach(self, q=1.0, t_air=35.0, p=101325.0):
        """Dry-cooler + HX approach (FWS - T_air) at duty q, full fan [K]."""
        return q * float(self.A(1.0, air_density_dry(t_air, p)))

    def c_air(self, s, t_air, p):
        return self.k * self.c_air_ref * np.asarray(s, float) * air_density_dry(t_air, p) / RHO_REF


def adiabatic_air_temp(t_db, t_wb, eta):
    return t_db - eta * (t_db - t_wb)


def adiabatic_water_m3h(c_air, t_db, t_air, overspray):
    """Water supplied [m3/h per MW design] when pre-cooling air of capacity rate c_air [MW/K] from t_db to t_air."""
    evap_kg_s = c_air * 1000.0 * (t_db - t_air) / 2450.0  # c_air*1000 kW/K = m_da*cp; evaporation = m_da*cp*dT/h_fg
    return evap_kg_s * 3.6 * overspray


# ---------------------------------------------------------------- chiller
def chiller_cop(t_amb, t_lvg, P):
    te = t_lvg - P["ch_dT_evap"] + 273.15
    tc = t_amb + P["ch_dT_cond"] + 273.15
    cop = P["ch_etaII"] * te / np.maximum(tc - te, 1.0)
    return np.minimum(cop, P["ch_cop_max"])


# ---------------------------------------------------------------- cooling tower (open tower + plate HX)
CPW = 4.186  # kJ/(kg K)


class TowerChain:
    """Open counterflow cooling tower (Braun-Klein-Mitchell effectiveness-NTU with enthalpy capacity rates) + plate HX.

    Water "capacity" in enthalpy units C_wh = m_w*cpw/c_s (c_s = saturation-enthalpy slope between water inlet and
    outlet); air capacity m_a; Q = eps(NTU = UA_h/C_min, C_min/C_max) * C_min * (h_sat(T_w,in) - h_air,in). With this
    coupling the leaving water can only approach the entering-air saturation temperature (~ wet-bulb), never cross it
    (M1 R1 finding 1). UA_h [kg/s] = kct * ntu_ref * m_a,ref * s^ua_exp.
    """

    def __init__(self, P, kct, fws_max):
        self.P, self.kct, self.fws_max = P, float(kct), float(fws_max)
        self.C_f = 1.0 / P["dT_fw"]
        self.C_cw = 1.0 / P["ct_dT_cw"]
        cmin, cmax = min(self.C_f, self.C_cw), max(self.C_f, self.C_cw)
        self.ua_hx = DryChain._ua_hx(P, cmin, cmax)
        self.eps_hx = float(eps_counterflow(self.ua_hx / cmin, cmin / cmax))
        self.cmin_hx = cmin
        self.m_w = self.C_cw * 1000.0 / CPW  # kg/s per MW design
        self.m_a_ref = P["ct_ma_ref"]

    def _m_a(self, s):
        return self.kct * self.m_a_ref * np.asarray(s, float)

    def _ua(self, s):
        return self.kct * self.P["ct_ntu_ref"] * self.m_a_ref * np.asarray(s, float) ** self.P["ct_ua_exp"]

    def tower_duty(self, t_hot, h_in, p, s=1.0, inner=4, t_s=None):
        """Tower duty [MW per MW design] for water entering at t_hot and air entering with enthalpy h_in.
        t_s (entering-air saturation temperature, ~ wet-bulb) may be passed in to avoid recomputing it."""
        m_a, ua = self._m_a(s), self._ua(s)
        hs_hot = ps.h_sat_fast(t_hot, p)
        if t_s is None:
            t_s = ps.t_from_h_sat_fast(h_in, p, iters=24)
        t_out = t_s + 0.5 * np.maximum(t_hot - t_s, 0.0)
        q = np.zeros(np.shape(t_hot))
        for _ in range(inner):
            dt = np.maximum(t_hot - t_out, 0.05)
            c_s = np.maximum((hs_hot - ps.h_sat_fast(t_out, p)) / dt, 1e-3)
            c_wh = self.m_w * CPW / c_s
            cmin, cmax = np.minimum(m_a, c_wh), np.maximum(m_a, c_wh)
            eps = eps_counterflow(ua / cmin, cmin / cmax)
            q = eps * cmin * np.maximum(hs_hot - h_in, 0.0)  # kW
            t_out = t_hot - q / (self.m_w * CPW)
        return q / 1000.0

    def chain_duty(self, fwr, t_wb, p, s=1.0, iters=20):
        """Largest duty with facility inlet fwr (fixed flows): solves tower_duty(T_cw,hot(Q)) = Q by bisection."""
        h_in = ps.h_sat_fast(t_wb, p)
        t_s = np.asarray(t_wb, float)  # h_in is the saturated enthalpy at t_wb, so its saturation temperature is t_wb
        a = 1.0 / (self.eps_hx * self.cmin_hx) - 1.0 / self.C_cw  # T_cw,hot = fwr - Q*a
        lo = np.zeros(np.shape(t_wb))
        hi = np.maximum(fwr - t_wb, 0.0) / max(a, 1e-9) + 1e-9
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            ok = self.tower_duty(fwr - mid * a, h_in, p, s, t_s=t_s) >= mid
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        return lo

    def trim_need(self, q, t_wb, p):
        fwr = self.fws_max + q / self.C_f
        return np.maximum(q - self.chain_duty(fwr, t_wb, p), 0.0)

    def fan_speed(self, q, qcap_full):
        """Declared approximation (DD-009): speed ~ required duty share of full-fan capability, clipped."""
        return np.clip(q / np.maximum(qcap_full, 1e-9), self.P["ct_min_speed"], 1.0)

    def water_m3h(self, q, t_wb, p, s):
        """Evaporation + blowdown [m3/h per MW design]; leaving air saturated at h_out = h_in + Q/m_a."""
        h_in = ps.h_sat_fast(t_wb, p)
        w_in = ps.w_sat(t_wb, p)
        m_a = np.maximum(self._m_a(s), 1e-6)
        t_out = ps.t_from_h_sat_fast(h_in + q * 1000.0 / m_a, p, iters=24)
        evap_kg_s = m_a * np.maximum(ps.w_sat(t_out, p) - w_in, 0.0)
        coc = self.P["ct_cycles"]
        return evap_kg_s * 3.6 * coc / (coc - 1.0)

    def fan_power(self, s):
        return self.kct * self.P["ct_fan_frac_ref"] * np.asarray(s, float) ** 3

    def approach_to_wetbulb(self, t_wb=25.0, p=101325.0):
        """Tower leaving water minus wet-bulb at design duty (facility holds the limit) [K]."""
        q = 1.0
        fwr = self.fws_max + q / self.C_f
        qc = float(self.chain_duty(fwr, np.array([t_wb]), p)[0])
        a = 1.0 / (self.eps_hx * self.cmin_hx) - 1.0 / self.C_cw
        return (fwr - min(qc, q) * a) - min(qc, q) / self.C_cw - t_wb, qc
