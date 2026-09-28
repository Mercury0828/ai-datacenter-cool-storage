"""Design menu, compliance under the risk standard, least-cost zone and feasibility (guide §4; DD-007).

Phase-0 R1 uses empirical (SAA-type) compliance: with N site-years and risk level eta, a design complies iff at most
m = floor(eta * N + 1e-9) site-years violate (e.g. N = 36, eta = 1/20 -> m = 1). A chiller design is sized to the
(m+1)-th largest annual chiller requirement, so it complies by construction. The DR version is R2's job (M3).
"""
import math

import numpy as np

from . import components as C
from . import evaluate as E


def allowed_violations(eta, n_years):
    return int(math.floor(eta * n_years + 1e-9))


def k_max(P, lim):
    """Largest dry-cooler size allowed by the P5 approach floor (dry-cooler approach at design duty, 35 degC, 1 atm)."""
    lo, hi = 0.3, 6.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if C.DryChain(P, mid, lim).dc_approach() > P["dc_approach_floor"]:
            lo = mid
        else:
            hi = mid
    return lo


def k_grid(P, lim, n=8, kmin=0.6):
    return list(np.round(np.linspace(kmin, k_max(P, lim), n), 4))


def capex_dry(P, k, ad, tank_h, c_ch):
    return (k * P["capex_dc_ref"] * (1.0 + (P["capex_ad_frac"] if ad else 0.0))
            + tank_h * P["capex_tank_per_mwh"] + c_ch * P["capex_chiller"])


def capex_tower(P, kct, c_ch):
    return kct * P["capex_tower"] + c_ch * P["capex_chiller"]


def annual_cost(P, capex, ox):
    return P["crf"] * capex + float(np.mean(ox["elec_MWh"])) * P["elec_price"] + float(np.mean(ox["water_m3"])) * P["water_price"]


def chiller_for_risk(cch_req, m):
    s = np.sort(np.asarray(cch_req))[::-1]
    return float(s[m]) if len(s) > m else 0.0


def enumerate_path(P, W, lim, water_allowed=True, keep_opex=True, prune=True):
    """All candidate designs for one path, each with compliance, annualized cost and attributes (per MW design).

    Pass 1 evaluates compliance (and the risk-sized chiller) for every design. Pass 2 computes operating cost only for
    compliant designs in ascending order of annualized capex, stopping once CRF*capex alone exceeds the best total
    found (operating cost >= 0, so the bound is exact: the least-cost design is unchanged). Pruned designs keep
    cost = None; their compliance flags remain available for F_dry / F_nc.
    """
    ny = W["t_db"].shape[0]
    m = allowed_violations(P["eta_risk"], ny)
    cands = []
    for k in k_grid(P, lim):
        for ad in ((False, True) if water_allowed else (False,)):
            for th in P["tank_hours_grid"]:
                ev = E.eval_dry(P, W, k, ad, th, lim)
                cc = chiller_for_risk(ev["cch_req"], m)
                for c_ch in ((0.0, cc) if cc > 1e-9 else (0.0,)):
                    n_viol = int(np.sum(ev["cch_req"] > c_ch + 1e-9))
                    cands.append({"family": "dry", "k": float(k), "ad": bool(ad), "tank_h": float(th), "c_ch": c_ch,
                                  "kct": 0.0, "n_viol": n_viol, "compliant": n_viol <= m,
                                  "capex": capex_dry(P, k, ad, th, c_ch), "cost": None, "evap": bool(ad),
                                  "dc_approach": C.DryChain(P, k, lim).dc_approach(), "ox": None, "_ev": ev})
    if water_allowed:
        for kct in P["kct_grid"]:
            ev = E.eval_tower(P, W, kct, lim)
            cc = chiller_for_risk(ev["cch_req"], m)
            for c_ch in ((0.0, cc) if cc > 1e-9 else (0.0,)):
                n_viol = int(np.sum(ev["cch_req"] > c_ch + 1e-9))
                cands.append({"family": "tower", "k": 0.0, "ad": False, "tank_h": 0.0, "c_ch": c_ch, "kct": float(kct),
                              "n_viol": n_viol, "compliant": n_viol <= m, "capex": capex_tower(P, kct, c_ch),
                              "cost": None, "evap": True, "dc_approach": None, "ox": None, "_ev": ev})
    if keep_opex:
        best = float("inf")
        for d in sorted((d for d in cands if d["compliant"]), key=lambda d: d["capex"]):
            if prune and P["crf"] * d["capex"] >= best:
                break
            if d["family"] == "dry":
                ox = E.opex_dry(P, W, d["_ev"], d["c_ch"], lim)
            else:
                ox = E.opex_tower(P, W, d["_ev"], d["c_ch"], lim)
            d["cost"] = annual_cost(P, d["capex"], ox)
            d["ox"] = {kk: float(np.mean(v)) for kk, v in ox.items() if kk not in ("viol",)}
            best = min(best, d["cost"])
    for d in cands:
        d.pop("_ev", None)
    return cands, m


def classify(designs):
    """F_dry, F_nc and least-cost zone for one path (or a combined site)."""
    comp = [d for d in designs if d["compliant"]]
    f_dry = any((d["c_ch"] <= 1e-9) and (not d["evap"]) for d in comp)
    f_nc = any(d["c_ch"] <= 1e-9 for d in comp)
    if not comp:
        return {"F_dry": False, "F_nc": False, "zone": "Zempty", "best": None}
    priced = [d for d in comp if d["cost"] is not None]
    best = min(priced, key=lambda d: d["cost"])
    zone = "Z3" if best["c_ch"] > 1e-9 else ("Z2" if best["evap"] else "Z1")
    return {"F_dry": f_dry, "F_nc": f_nc, "zone": zone, "best": best}


def design_point_dry_feasible(dbt, flt, min_approach):
    """Mokkapati & Das design-point rule: dry feasible iff DBT_design + minimum approach <= target fluid leaving T."""
    return np.asarray(dbt) + min_approach <= flt + 1e-9
