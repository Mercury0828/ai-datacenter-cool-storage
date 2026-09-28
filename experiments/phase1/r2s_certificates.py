"""PR-R2S follow-up (internal review r2-structure R1 non-blocking notes; no change to r2s_summary.json): solver certificates for the nine
pooled-oracle MILPs (status, mip_gap, mip_dual_bound, integrality residual, pooled violation of the rounded solution), the
within-site mixture relaxation (LP relaxation of the same multiple-choice problem; exploratory, not registered), and the headroom
against the registered interpolated frontier (r2r_c_summary rule, pareto_polyline and convex_hull best_baseline).
Writes experiments/phase1/out/r2s_certificates.json.
"""
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from r2s_structure import LEVELS, OUT, TESTS, load, site_curves


def problem(data, curves, sc):
    sites = list(data)
    W = sum(data[st]["w"] for st in sites)
    c, a_viol, idx = [], [], []
    for k, st in enumerate(sites):
        cv = curves[st][sc]
        for v, cost_v in enumerate(cv["cost_by_v"]):
            if np.isfinite(cost_v):
                idx.append((k, v))
                c.append(data[st]["w"] * cost_v / W)
                a_viol.append(data[st]["w"] * v / (cv["n"] * W))
    A = np.zeros((len(sites), len(idx)))
    for col, (k, _) in enumerate(idx):
        A[k, col] = 1.0
    return np.array(c), np.array(a_viol), A


def solve(c, a_viol, A, level, integer):
    cons = [LinearConstraint(A, np.ones(A.shape[0]), np.ones(A.shape[0])), LinearConstraint(a_viol[None, :], -np.inf, level)]
    return milp(c, constraints=cons, integrality=np.ones(len(c)) if integer else np.zeros(len(c)), bounds=Bounds(0, 1),
                options={"mip_rel_gap": 1e-6})


def main():
    data = load()
    tests = {site: {sc: np.load(OUT / "r2r" / f"{site}__{sc}.npz")["R_test"] for sc in ("ssp245_median", "ssp585_p90")} for site in data}
    curves = site_curves(data, tests)
    saved = json.loads((OUT / "r2s_summary.json").read_text(encoding="utf-8"))["oracle"]
    rule = json.loads((OUT / "r2r_c_summary.json").read_text(encoding="utf-8"))["corrected"]["refined_grid"]["rule"]
    res = {}
    for sc in TESTS:
        c, a_viol, A = problem(data, curves, sc)
        res[sc] = {}
        for lev in LEVELS:
            key = f"{lev:g}"
            r = solve(c, a_viol, A, lev, True)
            lp = solve(c, a_viol, A, lev, False)
            assert r.success and lp.success, (r.message, lp.message)
            x = np.round(r.x)
            rec_saved = saved[sc][key]
            base_exec = rec_saved["best_executable_baseline"]["cost"]
            rec = {"status": int(r.status), "message": r.message, "objective": float(r.fun),
                   "mip_gap": float(getattr(r, "mip_gap", np.nan)), "mip_dual_bound": float(getattr(r, "mip_dual_bound", np.nan)),
                   "integrality_residual": float(np.abs(r.x - x).max()),
                   "assignment_residual": float(np.abs(A @ x - 1.0).max()),
                   "pooled_violation_rounded": float(a_viol @ x), "objective_rounded": float(c @ x),
                   "matches_saved_objective": bool(abs(float(c @ x) - rec_saved["pooled_oracle_cost"]) <= 1e-6 * rec_saved["pooled_oracle_cost"]),
                   "mixture_relaxation_cost": float(lp.fun),
                   "headroom_mixture_vs_executable_pct": 100 * (1 - float(lp.fun) / base_exec)}
            for interp in ("pareto_polyline", "convex_hull"):
                b = rule[sc][key][interp]["best_baseline"]
                rec[f"best_baseline_{interp}"] = b
                rec[f"headroom_pooled_vs_{interp}_pct"] = 100 * (1 - rec_saved["pooled_oracle_cost"] / b)
            res[sc][key] = rec
            print(sc, key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("message",)}, flush=True)
    (OUT / "r2s_certificates.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
