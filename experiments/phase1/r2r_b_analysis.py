"""PR-R2R-b (ledger 2026-09-26, declared before this run): frontier-resolution correction of PR-R2R. Every method's frontier is
re-traced on refined knob grids (union with the registered grids), applied identically to all methods; selection, test sets, frontier
construction and the 3 % rule at 2.5 / 5 / 10 % are the registered ones (imported unchanged from r2r_analysis.py).
  DR        — kappa in {0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25} x eps_s and absolute eps in {0.005 … 0.3} K
  BL2-shift — registered kappa grid and absolute sigma in {0.1, 0.2, …, 3.0} K (same for BL3)
  BL1-shift — registered kappa in {0, 1} x Delta_s and absolute sigma in {0, 0.25, …, 2.0} K, x 8 variants
Writes experiments/phase1/out/r2r_b_summary.json (the registered r2r_summary.json is left untouched).
Usage: python experiments/phase1/r2r_b_analysis.py [--workers 2]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase0"))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402
from r2_aggregate import load  # noqa: E402
from r2_dro import bl1_need  # noqa: E402
from r2r_analysis import KAPPA, LEVELS, OUT, delta_s, dr_need, hull_cost, select, shifted  # noqa: E402

DR_KAPPA_FINE = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]
DR_ABS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
SHIFT_ABS = [round(0.1 * i, 1) for i in range(1, 31)]
BL1_ABS = [0.25 * i for i in range(9)]
TESTS = (("ssp245_median", 36), ("ssp585_p90", 36), ("F0", 11))
_DATA = None


def knobs(D_s):
    """(method, knob label, absolute K) for one site; registered grids first, refined grids after."""
    out = []
    for kap in sorted(set(KAPPA) | set(DR_KAPPA_FINE)):
        out.append(("DR", f"k{kap:g}", kap * D_s))
    out += [("DR", f"a{e:g}", e) for e in DR_ABS]
    for m in ("BL2-shift", "BL3"):
        out += [(m, f"k{kap:g}", kap * D_s) for kap in KAPPA]
        out += [(m, f"a{s:g}", s) for s in SHIFT_ABS]
    out += [("BL1-shift", f"k{kap:g}", kap * D_s) for kap in (0.0, 1.0)]
    out += [("BL1-shift", f"a{s:g}", s) for s in BL1_ABS]
    return out


def run_site(site):
    """Per-site picks and test violations, cached in out/r2r/b/<site>.json (resumable on the shared workstation)."""
    out = OUT / "r2r" / "b" / f"{site}.json"
    if out.exists():
        r = json.loads(out.read_text(encoding="utf-8"))
        return site, r["info"], r["rows"]
    s = _DATA[site]
    tests = {sc: np.load(OUT / "r2r" / f"{site}__{sc}.npz")["R_test"] for sc in ("ssp245_median", "ssp585_p90")}
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    yrs = s["years"]
    tr = np.where((yrs >= 1990) & (yrs <= 2014))[0]
    te = np.where((yrs >= 2015) & (yrs <= 2025))[0]
    D_s, slope = delta_s(site)
    cond = json.loads(str(np.load(ROOT / f"experiments/phase0/out/r2/sites/{site}.npz")["cond"]))
    p_mean = float(np.mean(E.load_weather(ROOT / f"data/weather/{site}.npz")["p"]))
    picks = {}
    for m, lab, sig in knobs(D_s):
        if m == "DR":
            picks[(m, lab)] = select(s, lambda j: dr_need(s, j, sig, tr))
        elif m == "BL2-shift":
            picks[(m, lab)] = select(s, lambda j: float(np.sort(shifted(s["R"][j], sig, s["deltas"])[tr])[::-1][1]))
        elif m == "BL3":
            picks[(m, lab)] = select(s, lambda j: float(shifted(s["R"][j], sig, s["deltas"])[tr].max()))
        else:
            for v in s["bl1v"]:
                c = {"dry": (cond[v]["dry"][0] + sig, cond[v]["dry"][1] + sig), "wet": (cond[v]["wet"][0] + sig, cond[v]["wet"][1] + sig)}
                nb = [bl1_need(P, tuple(d), c, p_mean, lim) for d in s["designs"]]
                picks[(m, f"{lab}|{v}")] = select(s, lambda j: max(nb[j], 0.0))
    rows = {}
    for key, val in picks.items():
        if val is None:
            continue
        j, C, c = val
        viol = {sc: int(np.sum(tests[sc][j] > C + 1e-9)) for sc in tests}
        viol["F0"] = int(np.sum(s["R"][j][0, te] > C + 1e-9))
        rows["|".join(key)] = {"j": int(j), "C": float(C), "cost": float(c), "viol": viol}
    info = {"delta_s_K": D_s, "jja_trend_K_per_yr": slope}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"info": info, "rows": rows}), encoding="utf-8")
    os.replace(tmp, out)
    return site, info, rows


def _init():
    global _DATA
    _DATA = load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    (OUT / "r2r" / "b").mkdir(parents=True, exist_ok=True)
    (OUT / "r2r" / "r2r_b.pid").write_text(str(os.getpid()), encoding="utf-8")
    data = load()
    per_site, site_info = {}, {}
    if a.workers <= 1:
        _init()
        results = map(run_site, sorted(data))
    else:
        pool = mp.Pool(a.workers, initializer=_init)
        results = pool.imap_unordered(run_site, sorted(data))
    for n, (site, info, rows) in enumerate(results, 1):
        per_site[site], site_info[site] = rows, info
        print(n, site, len(rows), flush=True)
    W = sum(s["w"] for s in data.values())
    keys = set.intersection(*(set(r) for r in per_site.values()))
    res = {"sites": site_info, "methods": {}}
    for k in sorted(keys):
        rec = {"cost_per_MW_yr": sum(data[st]["w"] * per_site[st][k]["cost"] for st in data) / W,
               "chiller_GW_th": sum(data[st]["w"] * per_site[st][k]["C"] for st in data) / 1e3}
        for sc, n in TESTS:
            rec[f"viol_{sc}"] = sum(data[st]["w"] * per_site[st][k]["viol"][sc] for st in data) / (W * n)
        res["methods"][k] = rec
    rule = {}
    for sc, _ in TESTS:
        rule[sc] = {}
        for lev in LEVELS:
            fr, npts = {}, {}
            for m in ("DR", "BL2-shift", "BL3", "BL1-shift"):
                pts = [(r[f"viol_{sc}"], r["cost_per_MW_yr"]) for k, r in res["methods"].items() if k.split("|")[0] == m]
                fr[m] = hull_cost(pts, lev)
                npts[m] = len(pts)
            base = [v for k, v in fr.items() if k != "DR" and v is not None]
            best_base = min(base) if base else None
            dr = fr["DR"]
            verdict = ("DR cannot match" if dr is None else "PASS (baselines cannot match)" if best_base is None else
                       "PASS" if dr <= 0.97 * best_base else "FAIL" if dr >= 1.03 * best_base else "INCONCLUSIVE")
            rule[sc][f"{lev:g}"] = {"frontier_cost": fr, "n_points": npts, "best_baseline": best_base,
                                    "dr_vs_best_pct": None if (dr is None or best_base is None) else 100 * (dr / best_base - 1),
                                    "verdict": verdict}
    # frontier-resolution diagnostic: largest violation gap between consecutive Pareto points of each method (central test)
    gaps = {}
    for m in ("DR", "BL2-shift", "BL3", "BL1-shift"):
        pts = sorted((r["viol_ssp245_median"], r["cost_per_MW_yr"]) for k, r in res["methods"].items() if k.split("|")[0] == m)
        par = []
        for v, c in pts:
            if not par or c < par[-1][1] - 1e-12:
                par.append((v, c))
        gaps[m] = {"pareto_points": len(par), "max_violation_gap": max((b[0] - a[0] for a, b in zip(par, par[1:])), default=None),
                   "pareto": par}
    res["rule"] = rule
    res["frontier_resolution_central"] = gaps
    res["verdict_central_5pct"] = rule["ssp245_median"]["0.05"]["verdict"]
    res["registered_verdict_central_5pct"] = json.loads((OUT / "r2r_summary.json").read_text(encoding="utf-8"))["verdict_central_5pct"]
    (OUT / "r2r_b_summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for sc, _ in TESTS:
        for lev in LEVELS:
            r = rule[sc][f"{lev:g}"]
            print(sc, lev, r["verdict"], {k: None if v is None else round(v, 1) for k, v in r["frontier_cost"].items()},
                  None if r["dr_vs_best_pct"] is None else round(r["dr_vs_best_pct"], 2))
    for m, g in gaps.items():
        print(m, g["pareto_points"], g["max_violation_gap"])


if __name__ == "__main__":
    main()
