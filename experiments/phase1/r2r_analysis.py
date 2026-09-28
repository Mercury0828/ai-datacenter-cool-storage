"""PR-R2R analysis: DR (drift-aware radius) vs comparable-margin baselines (BL2-shift, BL3 robust box, BL1-shift), each over its
knob, selected on training years 1990–2014 (R2 precompute), tested on untouched CMIP6 site-years (ssp245_median central,
ssp585_p90 secondary) and on the observed 2015–2025 (F0). Frontier (lower-left hull, linear interpolation) and the declared rule at
pooled test violation 5 % (also 2.5 % and 10 %). Writes experiments/phase1/out/r2r_summary.json.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import theilslopes

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase0"))
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402
from r2_aggregate import CAPCH, CRF, ETA, load, margins, wcp  # noqa: E402
from r2_dro import bl1_need  # noqa: E402

OUT = ROOT / "experiments/phase1/out"
KAPPA = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
HORIZON = 2040.5 - 2002.0
LEVELS = [0.025, 0.05, 0.10]


def delta_s(site):
    W = E.load_weather(ROOT / f"data/weather/{site}.npz")
    yrs = W["years"]
    tr = (yrs >= 1990) & (yrs <= 2014)
    jja = W["t_db"][:, 151 * 24:243 * 24].mean(axis=1)  # Jun 1 .. Aug 31 of a 365-day year
    slope = theilslopes(jja[tr], yrs[tr])[0]
    return max(0.0, float(slope)) * HORIZON, float(slope)


def shifted(Rj, sigma, deltas):
    """Yearly requirement after a uniform shift sigma (linear interpolation on the delta grid), all years."""
    g = np.searchsorted(deltas, sigma)
    if g == 0:
        return Rj[0]
    if g >= len(deltas):
        return Rj[-1]
    a = (sigma - deltas[g - 1]) / (deltas[g] - deltas[g - 1])
    return (1 - a) * Rj[g - 1] + a * Rj[g]


def cost(s, j, C):
    return CRF * (s["cap0"][j] + C * CAPCH) + s["op_tr"][j, 0 if C <= 1e-9 else 1]


def select(s, need_fn):
    best = None
    for j in range(len(s["designs"])):
        C = need_fn(j)
        if C is None:
            continue
        c = cost(s, j, C)
        if best is None or c < best[2]:
            best = (j, C, c)
    return best


def dr_need(s, j, eps, tr):
    Rj = s["R"][j]
    if wcp(margins(Rj, 0.0, tr, s["deltas"]), eps) <= ETA + 1e-12:
        return 0.0
    lo, hi = 0.0, float(Rj[-1, tr].max()) + 1e-9
    if wcp(margins(Rj, hi, tr, s["deltas"]), eps) > ETA + 1e-12:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if wcp(margins(Rj, mid, tr, s["deltas"]), eps) <= ETA + 1e-12:
            hi = mid
        else:
            lo = mid
    return hi


def hull_cost(points, v):
    """Cost on the lower-left hull at violation v (linear interpolation); None if no point reaches v."""
    pts = sorted(points)
    pareto = []
    for pv, pc in pts:
        if not pareto or pc < pareto[-1][1] - 1e-12:
            pareto.append((pv, pc))
    if pareto[0][0] > v + 1e-12:
        return None
    for (v0, c0), (v1, c1) in zip(pareto, pareto[1:]):
        if v0 <= v <= v1:
            return c0 + (c1 - c0) * (v - v0) / max(v1 - v0, 1e-15)
    return pareto[-1][1]


def main():
    data = load()
    tests = {}
    for site in data:
        tests[site] = {sc: np.load(OUT / "r2r" / f"{site}__{sc}.npz")["R_test"] for sc in ("ssp245_median", "ssp585_p90")}
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    res = {"sites": {}, "methods": {}}
    picks = {}
    for site, s in data.items():
        yrs = s["years"]
        tr = np.where((yrs >= 1990) & (yrs <= 2014))[0]
        te = np.where((yrs >= 2015) & (yrs <= 2025))[0]
        D_s, slope = delta_s(site)
        res["sites"][site] = {"delta_s_K": D_s, "jja_trend_K_per_yr": slope, "w": s["w"]}
        cond = json.loads(str(np.load(ROOT / f"experiments/phase0/out/r2/sites/{site}.npz")["cond"]))
        p_mean = float(np.mean(E.load_weather(ROOT / f"data/weather/{site}.npz")["p"]))
        for kap in KAPPA:
            sig = kap * D_s
            picks[(site, "DR", kap)] = select(s, lambda j: dr_need(s, j, sig, tr))
            picks[(site, "BL2-shift", kap)] = select(s, lambda j: float(np.sort(shifted(s["R"][j], sig, s["deltas"])[tr])[::-1][1]))
            picks[(site, "BL3", kap)] = select(s, lambda j: float(shifted(s["R"][j], sig, s["deltas"])[tr].max()))
        for kap in (0.0, 1.0):
            sig = kap * D_s
            for v in s["bl1v"]:
                c = {"dry": (cond[v]["dry"][0] + sig, cond[v]["dry"][1] + sig), "wet": (cond[v]["wet"][0] + sig, cond[v]["wet"][1] + sig)}
                nb = [bl1_need(P, tuple(d), c, p_mean, lim) for d in s["designs"]]
                picks[(site, "BL1-shift", f"{kap:g}|{v}")] = select(s, lambda j: max(nb[j], 0.0))
        # test violations for every pick
        for key, val in list(picks.items()):
            if key[0] != site or val is None or len(val) != 3:
                continue
            j, C, c = val
            viol = {sc: int(np.sum(tests[site][sc][j] > C + 1e-9)) for sc in tests[site]}
            viol["F0"] = int(np.sum(s["R"][j][0, te] > C + 1e-9))
            picks[key] = (j, C, c, viol)
    methods = {}
    for (site, m, kap), (j, C, c, viol) in picks.items():
        methods.setdefault((m, kap), []).append((site, j, C, c, viol))
    W = sum(s["w"] for s in data.values())
    for (m, kap), rows in methods.items():
        rec = {"cost_per_MW_yr": sum(data[st]["w"] * c for st, _, _, c, _ in rows) / W}
        for sc, n in (("ssp245_median", 36), ("ssp585_p90", 36), ("F0", 11)):
            rec[f"viol_{sc}"] = sum(data[st]["w"] * v[sc] for st, _, _, _, v in rows) / (W * n)
        rec["chiller_GW_th"] = sum(data[st]["w"] * C for st, _, C, _, _ in rows) / 1e3
        res["methods"][f"{m}|{kap}"] = rec
    rule = {}
    for sc in ("ssp245_median", "ssp585_p90", "F0"):
        rule[sc] = {}
        for lev in LEVELS:
            fr = {}
            for m in ("DR", "BL2-shift", "BL3", "BL1-shift"):
                pts = [(r[f"viol_{sc}"], r["cost_per_MW_yr"]) for k, r in res["methods"].items() if k.split("|")[0] == m]
                fr[m] = hull_cost(pts, lev)
            base = [v for k, v in fr.items() if k != "DR" and v is not None]
            best_base = min(base) if base else None
            dr = fr["DR"]
            verdict = ("DR cannot match" if dr is None else "PASS (baselines cannot match)" if best_base is None else
                       "PASS" if dr <= 0.97 * best_base else "FAIL" if dr >= 1.03 * best_base else "INCONCLUSIVE")
            rule[sc][f"{lev:g}"] = {"frontier_cost": fr, "best_baseline": best_base,
                                    "dr_vs_best_pct": None if (dr is None or best_base is None) else 100 * (dr / best_base - 1),
                                    "verdict": verdict}
    res["rule"] = rule
    res["verdict_central_5pct"] = rule["ssp245_median"]["0.05"]["verdict"]
    (OUT / "r2r_summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for k, r in sorted(res["methods"].items()):
        print(f"{k:22s} cost {r['cost_per_MW_yr']:9.1f}  viol ssp245 {r['viol_ssp245_median']:.3f}  ssp585p90 {r['viol_ssp585_p90']:.3f}  F0 {r['viol_F0']:.3f}  chiller {r['chiller_GW_th']:.2f} GW_th")
    print(json.dumps(rule, indent=1, default=float))


if __name__ == "__main__":
    main()
