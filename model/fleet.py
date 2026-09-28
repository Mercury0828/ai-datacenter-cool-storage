"""Cached evaluation of the full design menu at one site-climate (Phase 1, package 2).

`eval_dry_c` / `opex_dry_c` return exactly what `evaluate.eval_dry` / `evaluate.opex_dry` return (bit-identical; see
tests/test_fleet_cache.py) but share the hour-by-hour arrays that depend only on the dry-cooler size k (chain duty limits,
series-trim need, charging spare, adiabatic air temperature, adiabatic fan speed, fan power and pad water) across the tank,
adiabatic and chiller options of the menu. Only the arrays that depend on the tank dispatch (fan speed in charging hours) are
recomputed per design, elementwise, so the arithmetic per hour is unchanged.
"""
import numpy as np

from . import components as C
from . import evaluate as E


class SiteCache:
    def __init__(self, P, W, lim):
        self.P, self.W, self.lim = P, W, float(lim)
        self.q = 1.0
        self._k = {}
        self._tad = None
        self._cop = None

    @property
    def tad(self):
        if self._tad is None:
            self._tad = C.adiabatic_air_temp(self.W["t_db"], self.W["t_wb"], self.P["eta_ad"])
        return self._tad

    @property
    def cop(self):
        if self._cop is None:
            self._cop = C.chiller_cop(self.W["t_db"], self.lim, self.P)
        return self._cop

    def k_arrays(self, k, ad=False):
        key = float(k)
        a = self._k.get(key)
        P, W, q, lim = self.P, self.W, self.q, self.lim
        tdb, p = W["t_db"], W["p"]
        if a is None:
            chain = C.DryChain(P, k, lim)
            qmax_dry = chain.q_max(tdb, p)
            need_dry = chain.trim_need(q, tdb, p)
            ok_dry = qmax_dry >= q
            fwr = lim + q / chain.C_f
            spare = np.where(ok_dry, np.clip(chain.duty_fixed_inlet(tdb, p, fwr) - q, 0.0, P["tank_charge_max"]), 0.0)
            s_base = chain.min_speed_duty(np.full(tdb.shape, q), tdb, p, fwr)  # dry-mode speed with no charging
            a = {"chain": chain, "qmax_dry": qmax_dry, "need_dry": need_dry, "ok_dry": ok_dry, "fwr": fwr, "spare": spare,
                 "s_base": s_base}
            self._k[key] = a
        if ad and "need_ad" not in a:
            chain, tad = a["chain"], self.tad
            a["qmax_ad"] = chain.q_max(tad, p)
            a["need_ad"] = chain.trim_need(q, tad, p)
            ok_ad = a["qmax_ad"] >= q
            s_ad = np.where(ok_ad, chain.min_speed(q, tad, p), 1.0)
            a["fan_ad"] = chain.fan_power(s_ad)
            a["w_ad"] = C.adiabatic_water_m3h(chain.c_air(s_ad, tad, p), tdb, tad, P["ad_overspray"])
        return a


def eval_dry_c(cache, k, ad, tank_h):
    """Same return values as evaluate.eval_dry(P, W, k, ad, tank_h, lim)."""
    P = cache.P
    a = cache.k_arrays(k, ad)
    need = a["need_ad"] if ad else a["need_dry"]
    e_max = float(tank_h)
    ny = need.shape[0]
    cch_req = np.zeros(ny)
    for y in range(ny):
        if e_max > 0:
            cch_req[y] = E._min_chiller_with_tank(need[y], a["spare"][y], e_max, P["tank_charge_max"])
        else:
            cch_req[y] = need[y].max()
    return {"chain": a["chain"], "need": need, "need_dry": a["need_dry"], "ok_dry": a["ok_dry"], "spare": a["spare"],
            "qmax_dry": a["qmax_dry"], "qmax_ad": a["qmax_ad"] if ad else None, "tad": cache.tad if ad else None,
            "cch_req": cch_req, "e_max": e_max, "ad": ad, "k": k}


def opex_dry_c(cache, ev, c_ch, prices=None, detail=False):
    """Same return values as evaluate.opex_dry(P, W, ev, c_ch, lim). `prices` = (elec_price, water_price) overrides the
    prices used by the adiabatic dispatch rule (defaults: P). With detail=True also returns hourly unmet, electricity."""
    P, W, q = cache.P, cache.W, cache.q
    a = cache.k_arrays(ev["k"], ev["ad"])
    chain = a["chain"]
    tdb, p = W["t_db"], W["p"]
    ny = tdb.shape[0]
    need, ok_dry = ev["need"], a["ok_dry"]
    unmet = np.zeros(ny)
    charge = np.zeros_like(tdb)
    unmet_h = np.zeros_like(tdb) if detail else None
    if ev["e_max"] > 0:
        for y in range(ny):
            w, _, _, ch = E._tank_year(need[y], a["spare"][y], ev["e_max"], P["tank_charge_max"], c_ch)
            unmet[y] = w
            charge[y] = ch
            if detail and w > E.TOL:
                unmet_h[y] = _unmet_hours(need[y], a["spare"][y], ev["e_max"], P["tank_charge_max"], c_ch)
    else:
        unmet = np.maximum(need - c_ch, 0.0).max(axis=1)
        if detail:
            unmet_h = np.maximum(need - c_ch, 0.0)
    viol = unmet > E.TOL
    s = a["s_base"].copy()
    idx = charge > 0
    if idx.any():
        s[idx] = chain.min_speed_duty(q + charge[idx], tdb[idx], p[idx], a["fwr"])
    s_dry = np.where(ok_dry, s, 1.0)
    fan_dry = chain.fan_power(s_dry)
    water = np.zeros_like(tdb)
    fan = fan_dry.copy()
    ad_on = np.zeros(tdb.shape, bool)
    if ev["ad"]:
        pe, pw = prices if prices is not None else (P["elec_price"], P["water_price"])
        cost_dry = fan_dry * pe
        cost_ad = a["fan_ad"] * pe + a["w_ad"] * pw
        use_ad = (~ok_dry) | ((cost_ad < cost_dry) & (charge <= 0))
        ad_on = use_ad
        fan = np.where(use_ad, a["fan_ad"], fan_dry)
        water = np.where(use_ad, a["w_ad"], 0.0)
    trim = np.minimum(need, c_ch)
    ch_el = np.where(trim > 0, trim / cache.cop, 0.0)
    fan = np.where(need > 0, chain.fan_power(1.0), fan)
    elec = fan + ch_el
    out = {"viol": viol, "unmet_max": unmet, "fan_MWh": fan.sum(axis=1), "ch_MWh": ch_el.sum(axis=1),
           "water_m3": water.sum(axis=1), "ad_hours": ad_on.sum(axis=1), "trim_hours": (trim > 0).sum(axis=1),
           "peak_MW": elec.max(axis=1), "elec_MWh": elec.sum(axis=1), "water_peak_m3h": water.max(axis=1)}
    if detail:
        out.update(unmet_h=unmet_h, elec_h=elec, water_h=water, ch_el_h=ch_el)
    return out


def _unmet_hours(need, spare, e_max, charge_max, c_ch):
    """Hourly unmet duty of one site-year under the same greedy dispatch as evaluate._tank_year."""
    e = e_max
    out = np.zeros(need.shape[0])
    for h in range(need.shape[0]):
        r = need[h] - min(c_ch, need[h])
        if r > 0.0:
            d = min(r, e)
            e -= d
            out[h] = r - d
        elif spare[h] > 0.0:
            e += min(spare[h], charge_max, e_max - e)
    return out
