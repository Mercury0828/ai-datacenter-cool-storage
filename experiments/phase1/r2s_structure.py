"""PR-R2S (ledger 2026-09-26, declared after the gate-(a) ruling, before any run): structure of the R2 result.
(a) Saturation audit: DR minimal capacity at kappa = 1 (eps_s = Delta_s > eta*D = 0.25 K) vs the 2nd-largest training requirement at
    a uniform +5 K (BL2-shift at 5 K), per site and design; site selection costs; eps = 0.24 K reported for contrast.
(b) Oracle headroom on the exposed benchmark: sitewise and pooled (multiple-choice MILP) oracles over every design x test-requirement
    breakpoint, vs the cheapest executable registered-method point, at 2.5 / 5 / 10 % pooled test violation; central SSP2-4.5 median,
    secondary SSP5-8.5 p90 and F0.
(c) Fleet consequences on the 15 R2 sites: adjusted (BL2-shift kappa = 1) vs stationary (kappa = 0) vs over-robust (DR kappa = 1).
Writes experiments/phase1/out/r2s_summary.json.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase0"))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from r2_aggregate import ETA, load  # noqa: E402
from r2r_analysis import OUT, cost, delta_s, dr_need, select  # noqa: E402

LEVELS = [0.025, 0.05, 0.10]
TESTS = ("ssp245_median", "ssp585_p90", "F0")
D_SUPPORT = 5.0


def bl2_5k(s, j, tr):
    r = np.sort(s["R"][j][-1, tr])[::-1]
    assert abs(s["deltas"][-1] - D_SUPPORT) < 1e-12
    c = float(r[1])
    return c if c > 1e-9 else 0.0


def audit(data):
    res = {"eta_D_K": ETA * D_SUPPORT, "sites": {}}
    for site, s in data.items():
        yrs = s["years"]
        tr = np.where((yrs >= 1990) & (yrs <= 2014))[0]
        D_s, _ = delta_s(site)
        diffs, diffs_low = [], []
        for j in range(len(s["designs"])):
            c_dr = dr_need(s, j, D_s, tr)
            c_5k = bl2_5k(s, j, tr)
            diffs.append(None if c_dr is None else abs(c_dr - c_5k))
            c_low = dr_need(s, j, 0.24, tr)
            diffs_low.append(None if c_low is None else abs(c_low - c_5k))
        dr_sel = select(s, lambda j: dr_need(s, j, D_s, tr))
        bl_sel = select(s, lambda j: bl2_5k(s, j, tr))
        valid = [d for d in diffs if d is not None]
        res["sites"][site] = {"delta_s_K": D_s, "designs": len(diffs), "dr_infeasible": sum(d is None for d in diffs),
                              "max_abs_capacity_diff": max(valid) if valid else None,
                              "designs_equal_1e-6": int(sum(d is not None and d <= 1e-6 for d in diffs)),
                              "dr_cost": dr_sel[2], "bl2_5K_cost": bl_sel[2], "cost_equal": abs(dr_sel[2] - bl_sel[2]) <= 1e-6 * bl_sel[2],
                              "eps_0.24_designs_differing": int(sum(d is not None and d > 1e-6 for d in diffs_low))}
    res["all_sites_equal"] = all(v["cost_equal"] and v["designs_equal_1e-6"] == v["designs"] - v["dr_infeasible"] for v in res["sites"].values())
    return res


def site_curves(data, tests):
    """Per site and test set: the cheapest cost for each violating-year count v (0..n) over every design and capacity breakpoint."""
    curves = {}
    for site, s in data.items():
        yrs = s["years"]
        te = np.where((yrs >= 2015) & (yrs <= 2025))[0]
        curves[site] = {}
        for sc in TESTS:
            R = tests[site][sc] if sc != "F0" else np.stack([s["R"][j][0, te] for j in range(len(s["designs"]))])
            n = R.shape[1]
            best = np.full(n + 1, np.inf)
            for j in range(R.shape[0]):
                caps = np.unique(np.concatenate([[0.0], R[j][R[j] > 1e-9]]))
                for C in caps:
                    v = int(np.sum(R[j] > C + 1e-9))
                    c = cost(s, j, C)
                    if c < best[v]:
                        best[v] = c
            # a looser violation budget can use any tighter option: make the curve non-increasing in v
            best = np.minimum.accumulate(best)
            curves[site][sc] = {"n": n, "cost_by_v": best.tolist()}
    return curves


def pooled_oracle(data, curves, sc, level):
    sites = list(data)
    W = sum(data[st]["w"] for st in sites)
    c, a_viol, rows_assign = [], [], []
    idx = []
    for k, st in enumerate(sites):
        cv = curves[st][sc]
        n = cv["n"]
        for v, cost_v in enumerate(cv["cost_by_v"]):
            if not np.isfinite(cost_v):
                continue
            idx.append((k, v))
            c.append(data[st]["w"] * cost_v / W)
            a_viol.append(data[st]["w"] * v / (n * W))
    c = np.array(c); a_viol = np.array(a_viol)
    m = len(sites)
    A_assign = np.zeros((m, len(idx)))
    for col, (k, v) in enumerate(idx):
        A_assign[k, col] = 1.0
    cons = [LinearConstraint(A_assign, np.ones(m), np.ones(m)), LinearConstraint(a_viol[None, :], -np.inf, level)]
    r = milp(c, constraints=cons, integrality=np.ones(len(idx)), bounds=Bounds(0, 1), options={"mip_rel_gap": 1e-6})
    assert r.success, r.message
    x = np.round(r.x).astype(int)
    chosen = {sites[k]: v for col, (k, v) in enumerate(idx) if x[col] == 1}
    return float(r.fun), float(a_viol @ x), chosen


def executable(summary, sc, level):
    best = {}
    for key, rec in summary.items():
        m = key.split("|")[0]
        if rec[f"viol_{sc}"] <= level + 1e-12 and (m not in best or rec["cost_per_MW_yr"] < best[m][0]):
            best[m] = (rec["cost_per_MW_yr"], rec[f"viol_{sc}"], key)
    return best


def main():
    data = load()
    tests = {site: {sc: np.load(OUT / "r2r" / f"{site}__{sc}.npz")["R_test"] for sc in ("ssp245_median", "ssp585_p90")} for site in data}
    res = {"audit": audit(data)}
    print("audit all equal:", res["audit"]["all_sites_equal"], flush=True)
    curves = site_curves(data, tests)
    summ = json.loads((OUT / "r2r_c_summary.json").read_text(encoding="utf-8"))["corrected"]["refined_grid"]["methods"]
    W = sum(s["w"] for s in data.values())
    res["oracle"] = {}
    for sc in TESTS:
        res["oracle"][sc] = {}
        for lev in LEVELS:
            k_pool, v_pool, chosen = pooled_oracle(data, curves, sc, lev)
            n = curves[next(iter(data))][sc]["n"]
            vmax = int(np.floor(lev * n + 1e-9))
            k_site = sum(data[st]["w"] * curves[st][sc]["cost_by_v"][vmax] for st in data) / W
            ex = executable(summ, sc, lev)
            base = {m: v for m, v in ex.items() if m != "DR"}
            best_base = min(base.values()) if base else None
            rec = {"pooled_oracle_cost": k_pool, "pooled_oracle_violation": v_pool, "pooled_oracle_v_by_site": chosen,
                   "sitewise_oracle_cost": k_site, "sitewise_violation_cap": vmax,
                   "executable": {m: {"cost": v[0], "viol": v[1], "knob": v[2]} for m, v in ex.items()},
                   "best_executable_baseline": None if best_base is None else {"cost": best_base[0], "knob": best_base[2]}}
            if best_base is not None:
                rec["headroom_pooled_oracle_pct"] = 100 * (1 - k_pool / best_base[0])
                rec["headroom_sitewise_oracle_pct"] = 100 * (1 - k_site / best_base[0])
            res["oracle"][sc][f"{lev:g}"] = rec
            print(sc, lev, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items() if k.startswith("headroom") or k.endswith("cost")},
                  flush=True)
    # (c) fleet consequences
    rows = {site: json.loads((OUT / "r2r" / "b" / f"{site}.json").read_text(encoding="utf-8"))["rows"] for site in data}
    def fleet(key):
        return (sum(data[st]["w"] * rows[st][key]["C"] for st in data), sum(data[st]["w"] * rows[st][key]["cost"] for st in data))
    c_adj, k_adj = fleet("BL2-shift|k1")
    c_sta, k_sta = fleet("BL2-shift|k0")
    c_dr, k_dr = fleet("DR|k1")
    res["fleet"] = {"capacity_MW": W, "delta_C_adjusted_minus_stationary_MW_th": c_adj - c_sta,
                    "excess_C_overrobust_minus_adjusted_MW_th": c_dr - c_adj,
                    "delta_cost_adjusted_minus_stationary_MUSD_yr": (k_adj - k_sta) / 1e6,
                    "excess_cost_overrobust_minus_adjusted_MUSD_yr": (k_dr - k_adj) / 1e6,
                    "C_stationary_MW_th": c_sta, "C_adjusted_MW_th": c_adj, "C_overrobust_MW_th": c_dr}
    (OUT / "r2s_summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res["fleet"], indent=1))


if __name__ == "__main__":
    main()
