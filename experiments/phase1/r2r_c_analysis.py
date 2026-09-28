"""PR-R2R-c (ledger 2026-09-26, declared after internal review r2-repair R1, before its run): baselines beyond the 5 K grid.
1. BL2-shift / BL3 at registered shifts above the 5 K precompute grid (kappa = 2 at Stargate UAE, Shackelford, New Mexico) use the
   exact yearly requirements of all 68 R2 designs recomputed on the training weather shifted by the actual kappa * Delta_s
   (r2_dro.req_and_opex, RH-preserving shift as DD-010; monotone in the shift as in r2_aggregate.load); DR keeps its 5 K support.
2. BL1-shift's mean site pressure from the training years only.
3. Registered frontier convention (Pareto polyline, r2r_analysis.hull_cost) plus the lower convex hull as a sensitivity.
DR rows are copied from the refined per-site checkpoints (out/r2r/b/<site>.json; DR is unaffected). Built-in check: with items 1-2
switched off, the recomputed baseline rows equal the stored ones and both original summaries are reproduced exactly.
Writes experiments/phase1/out/r2r_c_summary.json.
Usage: python experiments/phase1/r2r_c_analysis.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
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
from r2_dro import ETA2, bl1_need, design_list, req_and_opex  # noqa: E402
from r2r_analysis import KAPPA, LEVELS, OUT, hull_cost, select, shifted  # noqa: E402
from r2r_b_analysis import TESTS, knobs  # noqa: E402

EXT = OUT / "r2r" / "ext"
METHODS = ("DR", "BL2-shift", "BL3", "BL1-shift")


def ext_requirements(site, sigma, designs):
    """Exact yearly requirements of every R2 design on the site's weather shifted by sigma (> 5 K); cached on disk."""
    EXT.mkdir(parents=True, exist_ok=True)
    f = EXT / f"{site}__{sigma:.9f}.npz"
    if f.exists():
        return np.load(f)["R"]
    P = PR.params(ch_etaII=ETA2)
    lim = PR.fws_max(P)
    ds = design_list(P, lim)
    assert [json.loads(json.dumps(d)) for d in ds] == [list(d) for d in designs], "design order differs from the R2 precompute"
    W = E.shift_weather(E.load_weather(ROOT / f"data/weather/{site}.npz"), sigma)
    R = np.array([req_and_opex(P, W, d, lim, want_opex=False)[0] for d in ds])
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, R=R, sigma=sigma)
    os.replace(tmp, f)
    return R


def shifted_req(s, site, j, sig, exact):
    """Yearly requirements after a uniform shift sig: grid interpolation up to 5 K; beyond it the exact recomputation (exact=True)
    or the registered implementation's clipped 5 K row (exact=False)."""
    if sig <= s["deltas"][-1] + 1e-12 or not exact:
        return shifted(s["R"][j], sig, s["deltas"])
    Rx = ext_requirements(site, sig, s["designs"])
    return np.maximum(Rx[j], s["R"][j][-1])  # monotone in the shift, as r2_aggregate.load enforces on the grid


def baseline_rows(site, s, tests, exact, train_p):
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    yrs = s["years"]
    tr = np.where((yrs >= 1990) & (yrs <= 2014))[0]
    te = np.where((yrs >= 2015) & (yrs <= 2025))[0]
    D_s = json.loads((OUT / "r2r" / "b" / f"{site}.json").read_text(encoding="utf-8"))["info"]["delta_s_K"]
    cond = json.loads(str(np.load(ROOT / f"experiments/phase0/out/r2/sites/{site}.npz")["cond"]))
    Wp = E.load_weather(ROOT / f"data/weather/{site}.npz")["p"]
    p_mean = float(np.mean(Wp[tr])) if train_p else float(np.mean(Wp))
    picks = {}
    for m, lab, sig in knobs(D_s):
        if m == "BL2-shift":
            picks[(m, lab)] = select(s, lambda j: float(np.sort(shifted_req(s, site, j, sig, exact)[tr])[::-1][1]))
        elif m == "BL3":
            picks[(m, lab)] = select(s, lambda j: float(shifted_req(s, site, j, sig, exact)[tr].max()))
        elif m == "BL1-shift":
            for v in s["bl1v"]:
                c = {"dry": (cond[v]["dry"][0] + sig, cond[v]["dry"][1] + sig), "wet": (cond[v]["wet"][0] + sig, cond[v]["wet"][1] + sig)}
                nb = [bl1_need(P, tuple(d), c, p_mean, lim) for d in s["designs"]]
                picks[(m, f"{lab}|{v}")] = select(s, lambda j: max(nb[j], 0.0))
    rows = {}
    for key, (j, C, c) in picks.items():
        viol = {sc: int(np.sum(tests[sc][j] > C + 1e-9)) for sc in tests}
        viol["F0"] = int(np.sum(s["R"][j][0, te] > C + 1e-9))
        rows["|".join(key)] = {"j": int(j), "C": float(C), "cost": float(c), "viol": viol}
    return rows


def convex_cost(points, v):
    """Lower convex hull of (violation, cost) points, restricted to its non-increasing part; linear interpolation at v."""
    pts = sorted(set(points))
    hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0:
            hull.pop()
        hull.append(p)
    par = []
    for q in hull:
        if not par or q[1] < par[-1][1] - 1e-12:
            par.append(q)
    if par[0][0] > v + 1e-12:
        return None
    for (v0, c0), (v1, c1) in zip(par, par[1:]):
        if v0 <= v <= v1:
            return c0 + (c1 - c0) * (v - v0) / max(v1 - v0, 1e-15)
    return par[-1][1]


def summarize(data, per_site, label_ok):
    W = sum(s["w"] for s in data.values())
    keys = [k for k in set.intersection(*(set(r) for r in per_site.values())) if label_ok(k)]
    methods = {}
    for k in sorted(keys):
        rec = {"cost_per_MW_yr": sum(data[st]["w"] * per_site[st][k]["cost"] for st in data) / W,
               "chiller_GW_th": sum(data[st]["w"] * per_site[st][k]["C"] for st in data) / 1e3}
        for sc, n in TESTS:
            rec[f"viol_{sc}"] = sum(data[st]["w"] * per_site[st][k]["viol"][sc] for st in data) / (W * n)
        methods[k] = rec
    rule = {}
    for sc, _ in TESTS:
        rule[sc] = {}
        for lev in LEVELS:
            out = {}
            for conv, fn in (("pareto_polyline", hull_cost), ("convex_hull", convex_cost)):
                fr = {m: fn([(r[f"viol_{sc}"], r["cost_per_MW_yr"]) for k, r in methods.items() if k.split("|")[0] == m], lev)
                      for m in METHODS}
                base = [v for k, v in fr.items() if k != "DR" and v is not None]
                best = min(base) if base else None
                dr = fr["DR"]
                verdict = ("DR cannot match" if dr is None else "PASS (baselines cannot match)" if best is None else
                           "PASS" if dr <= 0.97 * best else "FAIL" if dr >= 1.03 * best else "INCONCLUSIVE")
                out[conv] = {"frontier_cost": fr, "best_baseline": best,
                             "dr_vs_best_pct": None if (dr is None or best is None) else 100 * (dr / best - 1), "verdict": verdict}
            rule[sc][f"{lev:g}"] = out
    return methods, rule


def registered_label(k):
    m, rest = k.split("|", 1)
    if m == "BL1-shift":
        return rest.split("|")[0] in ("k0", "k1")
    return rest.startswith("k") and float(rest[1:]) in KAPPA


def main():
    data = load()
    stored = {site: json.loads((OUT / "r2r" / "b" / f"{site}.json").read_text(encoding="utf-8"))["rows"] for site in data}
    tests = {site: {sc: np.load(OUT / "r2r" / f"{site}__{sc}.npz")["R_test"] for sc in ("ssp245_median", "ssp585_p90")} for site in data}
    res = {}
    for variant, exact, train_p in (("original_implementation", False, False), ("corrected", True, True)):
        per_site = {}
        for site, s in data.items():
            rows = {k: v for k, v in stored[site].items() if k.startswith("DR|")}
            rows.update(baseline_rows(site, s, tests[site], exact, train_p))
            if variant == "original_implementation":
                bad = [k for k in stored[site] if stored[site][k] != rows.get(k)]
                assert not bad, (site, bad[:5])
            per_site[site] = rows
            print(variant, site, flush=True)
        reg_m, reg_rule = summarize(data, per_site, registered_label)
        ref_m, ref_rule = summarize(data, per_site, lambda k: True)
        res[variant] = {"registered_grid": {"methods": reg_m, "rule": reg_rule}, "refined_grid": {"methods": ref_m, "rule": ref_rule}}
    # built-in check: the original implementation reproduces both original summaries (registered convention)
    o_reg = json.loads((OUT / "r2r_summary.json").read_text(encoding="utf-8"))["rule"]
    o_ref = json.loads((OUT / "r2r_b_summary.json").read_text(encoding="utf-8"))["rule"]
    for sc, _ in TESTS:
        for lev in LEVELS:
            a = res["original_implementation"]["registered_grid"]["rule"][sc][f"{lev:g}"]["pareto_polyline"]
            b = res["original_implementation"]["refined_grid"]["rule"][sc][f"{lev:g}"]["pareto_polyline"]
            for mine, orig in ((a, o_reg[sc][f"{lev:g}"]), (b, o_ref[sc][f"{lev:g}"])):
                assert mine["verdict"] == orig["verdict"], (sc, lev, mine["verdict"], orig["verdict"])
                for m in METHODS:
                    x, y = mine["frontier_cost"][m], orig["frontier_cost"][m]
                    assert (x is None and y is None) or abs(x - y) < 1e-6, (sc, lev, m, x, y)
    res["built_in_check"] = "original implementation reproduces r2r_summary.json and r2r_b_summary.json rule tables exactly"
    (OUT / "r2r_c_summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for grid in ("registered_grid", "refined_grid"):
        for sc, _ in TESTS:
            for lev in LEVELS:
                r = res["corrected"][grid]["rule"][sc][f"{lev:g}"]
                print(grid, sc, lev, {c: (v["verdict"], None if v["dr_vs_best_pct"] is None else round(v["dr_vs_best_pct"], 2)) for c, v in r.items()})


if __name__ == "__main__":
    main()
