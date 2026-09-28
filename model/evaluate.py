"""Hour-by-hour evaluation of heat-rejection designs over site-years (Phase-0 M1/R1 harness).

A *path* is one heat-rejection loop with its own supply limit: the liquid loop (limit FWS_max = T_TCS,max - dT_CDU)
or, in the hybrid variant, the air path (limit T_sa_max - dT_coil). Everything is per MW of design heat on the path.

Per-hour operating policy for a dry-family design (DD-007):
  1. dry mode at the lowest fan speed that holds the limit, if full speed can;
  2. adiabatic pre-cooling (if installed) when dry mode cannot hold the limit, or when it is cheaper;
  3. otherwise the residual need (series trim duty at full fan, adiabatic on if installed) goes to the trim
     chiller (up to C_ch) and then to the tank; anything left is unmet -> the site-year violates.
  Tank: charged greedily from spare dry-mode capacity (up to tank_charge_max), discharged only for unmet need;
  full at 1 January of every site-year. Greedy fill-when-possible is feasibility-optimal for a single lossless store.
A site-year *violates* iff any hour has unmet need > 1e-9 (rack-coolant limit exceeded after all recourse).
"""
import numpy as np
from numba import njit

from . import components as C
from . import psychro as ps

HOURS = 8760
TOL = 1e-9


# ------------------------------------------------------------------ weather
def load_weather(path):
    """Per-site ERA5 table -> dict of (years, 8760) arrays (Feb 29 dropped; DD-007)."""
    z = np.load(path)
    t0 = np.datetime64(str(z["t0"]).replace("Z", ""), "h")
    n = int(z["n_hours"])
    time = t0 + np.arange(n).astype("timedelta64[h]")
    t = z["t2m_c100"].astype(float) / 100.0
    td = z["td_c100"].astype(float) / 100.0
    p = z["psfc_pa10"].astype(float) * 10.0
    md = (time.astype("datetime64[D]") - time.astype("datetime64[M]")).astype(int) + 1
    mo = time.astype("datetime64[M]").astype(int) % 12 + 1
    keep = ~((mo == 2) & (md == 29))
    t, td, p, time = t[keep], td[keep], p[keep], time[keep]
    years = time.astype("datetime64[Y]").astype(int) + 1970
    uy = np.unique(years)
    ny = len(uy)
    assert len(t) == ny * HOURS, (len(t), ny)
    W = {"years": uy, "t_db": t.reshape(ny, HOURS), "t_dp": td.reshape(ny, HOURS), "p": p.reshape(ny, HOURS)}
    W["t_wb"] = ps.wet_bulb(W["t_db"], W["t_dp"], W["p"])
    return W


def shift_weather(W, dT, hot_amp=0.0):
    """Warming stress: T_db + dT (optionally amplified on hot hours: + hot_amp*dT*(rank above p90)),
    dew point shifted to keep relative humidity constant, wet-bulb recomputed (DD-008)."""
    t = W["t_db"]
    if hot_amp:
        p90 = np.percentile(t, 90, axis=1, keepdims=True)
        tmax = t.max(axis=1, keepdims=True)
        frac = np.clip((t - p90) / np.maximum(tmax - p90, 1e-6), 0.0, 1.0)
        d = dT * (1.0 + hot_amp * frac)
    else:
        d = dT
    t_new = t + d
    rh = np.clip(ps.rel_humidity(t, W["t_dp"]), 1e-4, 1.0)
    pw = rh * ps.p_ws(t_new)
    td_new = ps.dewpoint_from_w(ps.w_from_pw(pw, W["p"]), W["p"])
    td_new = np.minimum(td_new, t_new)
    out = dict(W)
    out.update(t_db=t_new, t_dp=td_new, t_wb=ps.wet_bulb(t_new, td_new, W["p"]))
    return out


# ------------------------------------------------------------------ storage kernels
@njit(cache=True)
def _tank_year(need, spare, e_max, charge_max, c_ch):
    """One site-year: chiller covers min(C, need) first, tank covers the rest; charge from spare capacity.
    Returns (max unmet, total discharge, total charge, charge series)."""
    e = e_max
    worst = 0.0
    dis = 0.0
    chg = 0.0
    charge = np.zeros(need.shape[0])
    for h in range(need.shape[0]):
        r = need[h] - min(c_ch, need[h])
        if r > 0.0:
            d = min(r, e)
            e -= d
            dis += d
            if r - d > worst:
                worst = r - d
        elif spare[h] > 0.0:
            c = min(spare[h], charge_max, e_max - e)
            e += c
            chg += c
            charge[h] = c
    return worst, dis, chg, charge


@njit(cache=True)
def _min_chiller_with_tank(need, spare, e_max, charge_max):
    """Smallest chiller capacity (per-unit) making the year violation-free with the given tank (bisection)."""
    hi = 0.0
    for h in range(need.shape[0]):
        if need[h] > hi:
            hi = need[h]
    w, _, _, _ = _tank_year(need, spare, e_max, charge_max, 0.0)
    if w <= 1e-9:
        return 0.0
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        w, _, _, _ = _tank_year(need, spare, e_max, charge_max, mid)
        if w <= 1e-9:
            hi = mid
        else:
            lo = mid
    return hi


@njit(cache=True)
def sequent_peak(need, spare):
    """Minimum storage (per-unit hours) for one site-year with the declared greedy dispatch (store full at the start;
    discharge only for need; charge from spare only in hours without need): K_t = max(0, K_{t-1} + need_t - spare_t*1[need_t=0]).
    Returns (max K, hours from the last K = 0 before the maximum to the maximum, hours with need > 0)."""
    k = 0.0
    kmax = 0.0
    start = 0
    best_len = 0
    n_need = 0
    for h in range(need.shape[0]):
        if need[h] > 0.0:
            k += need[h]
            n_need += 1
        else:
            k -= spare[h]
            if k < 0.0:
                k = 0.0
        if k <= 0.0:
            start = h + 1
        if k > kmax:
            kmax = k
            best_len = h - start + 1
    return kmax, best_len, n_need


# ------------------------------------------------------------------ dry-family design
def eval_dry(P, W, k, ad, tank_h, lim):
    """Evaluate a dry-cooler design (size k, adiabatic yes/no, tank hours) on all site-years for a path with supply
    limit `lim`. Returns per-year arrays; chiller capacity is *not* fixed here: `cch_req` is the smallest capacity
    that makes each year violation-free (0 if the year is violation-free without a chiller)."""
    chain = C.DryChain(P, k, lim)
    tdb, twb, p = W["t_db"], W["t_wb"], W["p"]
    q = 1.0
    qmax_dry = chain.q_max(tdb, p)
    need_dry = chain.trim_need(q, tdb, p)
    if ad:
        tad = C.adiabatic_air_temp(tdb, twb, P["eta_ad"])
        qmax_ad = chain.q_max(tad, p)
        need_ad = chain.trim_need(q, tad, p)
        need = need_ad
    else:
        need = need_dry
    ok_dry = qmax_dry >= q
    # spare capacity for charging (dry mode only): with fixed flows the warmest water available is the design return
    # fwr = lim + q/C_f, so the extra duty is (fwr - T_air)/R(full fan) - q, the store being charged below the limit
    # (M1 R1 finding 2; DD-009)
    fwr = lim + q / chain.C_f
    spare = np.where(ok_dry, np.clip(chain.duty_fixed_inlet(tdb, p, fwr) - q, 0.0, P["tank_charge_max"]), 0.0)
    e_max = float(tank_h)
    ny = tdb.shape[0]
    cch_req = np.zeros(ny)
    charge = np.zeros_like(tdb)
    for y in range(ny):
        if e_max > 0:
            cch_req[y] = _min_chiller_with_tank(need[y], spare[y], e_max, P["tank_charge_max"])
        else:
            cch_req[y] = need[y].max()
    return {"chain": chain, "need": need, "need_dry": need_dry, "ok_dry": ok_dry, "spare": spare,
            "qmax_dry": qmax_dry, "qmax_ad": qmax_ad if ad else None, "tad": tad if ad else None,
            "cch_req": cch_req, "e_max": e_max, "ad": ad, "k": k}


def opex_dry(P, W, ev, c_ch, lim):
    """Annual operating quantities of a dry-family design with chiller capacity c_ch (per-unit), per site-year."""
    chain = ev["chain"]
    tdb, twb, p = W["t_db"], W["t_wb"], W["p"]
    q = 1.0
    ny = tdb.shape[0]
    need = ev["need"]
    ok_dry = ev["ok_dry"]
    # tank dispatch with the chosen chiller capacity
    unmet = np.zeros(ny)
    charge = np.zeros_like(tdb)
    if ev["e_max"] > 0:
        for y in range(ny):
            w, _, _, ch = _tank_year(need[y], ev["spare"][y], ev["e_max"], P["tank_charge_max"], c_ch)
            unmet[y] = w
            charge[y] = ch
    else:
        unmet = np.maximum(need - c_ch, 0.0).max(axis=1)
    viol = unmet > TOL
    # dry-mode hours: fan speed for duty (q + charge) at the fixed design return temperature (same balance as spare)
    fwr = lim + q / chain.C_f
    s_dry = np.where(ok_dry, chain.min_speed_duty(q + charge, tdb, p, fwr), 1.0)
    fan_dry = chain.fan_power(s_dry)
    water = np.zeros_like(tdb)
    fan = fan_dry.copy()
    ad_on = np.zeros(tdb.shape, bool)
    if ev["ad"]:
        tad = ev["tad"]
        ok_ad = ev["qmax_ad"] >= q
        s_ad = np.where(ok_ad, chain.min_speed(q, tad, p), 1.0)
        fan_ad = chain.fan_power(s_ad)
        w_ad = C.adiabatic_water_m3h(chain.c_air(s_ad, tad, p), tdb, tad, P["ad_overspray"])
        cost_dry = fan_dry * P["elec_price"]
        cost_ad = fan_ad * P["elec_price"] + w_ad * P["water_price"]
        use_ad = (~ok_dry) | ((cost_ad < cost_dry) & (charge <= 0))
        ad_on = use_ad
        fan = np.where(use_ad, fan_ad, fan_dry)
        water = np.where(use_ad, w_ad, 0.0)
    trim = np.minimum(need, c_ch)
    cop = C.chiller_cop(tdb, lim, P)
    ch_el = np.where(trim > 0, trim / cop, 0.0)
    fan = np.where(need > 0, chain.fan_power(1.0), fan)
    elec = fan + ch_el
    return {"viol": viol, "unmet_max": unmet, "fan_MWh": fan.sum(axis=1), "ch_MWh": ch_el.sum(axis=1),
            "water_m3": water.sum(axis=1), "ad_hours": ad_on.sum(axis=1), "trim_hours": (trim > 0).sum(axis=1),
            "peak_MW": elec.max(axis=1), "elec_MWh": elec.sum(axis=1)}


# ------------------------------------------------------------------ tower design
def eval_tower(P, W, kct, lim):
    """Open-tower design of size kct (+ plate HX), Braun-type coupled model (DD-009). Trim need = series-trim duty
    when the tower cannot hold the limit at full fan. Fan speed ~ duty share of full-fan capability (energy only)."""
    tw = C.TowerChain(P, kct, lim)
    twb, p = W["t_wb"], W["p"]
    q = 1.0
    fwr = lim + q / tw.C_f
    qcap = tw.chain_duty(fwr, twb, p)
    need = np.maximum(q - qcap, 0.0)
    s = tw.fan_speed(q, qcap)
    water = tw.water_m3h(np.minimum(q, qcap), twb, p, s)
    fan = np.where(need > 0, tw.fan_power(1.0), tw.fan_power(s))
    return {"tower": tw, "need": need, "water_m3h": water, "fan": fan, "cch_req": need.max(axis=1), "kct": kct}


def opex_tower(P, W, ev, c_ch, lim):
    need = ev["need"]
    unmet = np.maximum(need - c_ch, 0.0).max(axis=1)
    trim = np.minimum(need, c_ch)
    cop = C.chiller_cop(W["t_db"], lim, P)
    ch_el = np.where(trim > 0, trim / cop, 0.0)
    elec = ev["fan"] + ch_el
    return {"viol": unmet > TOL, "unmet_max": unmet, "fan_MWh": ev["fan"].sum(axis=1), "ch_MWh": ch_el.sum(axis=1),
            "water_m3": ev["water_m3h"].sum(axis=1), "ad_hours": np.zeros(need.shape[0]),
            "trim_hours": (trim > 0).sum(axis=1), "peak_MW": elec.max(axis=1), "elec_MWh": elec.sum(axis=1)}
