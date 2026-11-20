"""W3 (MANIFEST.md, Applied Energy analyses): what the 35-of-36 screen accepts and what removing it takes.
Per hub cell and climate (historical, +1.2 K, SSP2-4.5 circa 2040 central) on the CDS tables of record, at k_max:
  Q-A  annual store requirements E_y (sequent peak), sorted; E_(m+1) for m = 0..3
  Q-C  the worst year (year of E_(1); ties -> earliest year) simulated with a store of capacity E_(2): unmet heat, hours with
       unmet duty, longest run of consecutive unmet hours, peak hourly unmet duty, minimum constant trim duty c that leaves no
       unmet hour with the E_(2) store (chiller removes up to c of each hour's need before the store discharges; bisection 1e-6)
  Q-D  hw_screen.screen_path with m = 0 (F_dry, F_nc at 36/36)
Consistency gate: E_(2) must equal h_star of out/m2/cds_rebuild/<cell>.json to 1e-9 h (checked in the summary step).
Usage: python experiments/phase1/w3_risk.py [--workers 2] [--part cells|summary|all]
Outputs: experiments/phase1/out/w3/<cell>.json, w3_summary.json, w3_sites.csv
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from bv1_min_storage import need_spare  # noqa: E402
from cmip_scenarios import SCEN, scenario_weather  # noqa: E402
from hw_screen import screen_path  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

CDS = pathlib.Path("D:/_caches/era5/cds/tables")
OUT = ROOT / "experiments/phase1/out/w3"
REB = ROOT / "experiments/phase1/out/m2/cds_rebuild"
CLIMS = ("historical", "stress_plus1.2K", "ssp245_median")
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expected_hashes():
    out = {}
    for line in (ROOT / "release/inputs.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            out[p.strip()] = h
    return out


def bounded(need, spare, cap, c=0.0):
    """Greedy store of capacity cap (store full at the start; D = heat drawn from the full store). The chiller removes up to c of
    each hour's need first; the store then discharges up to its remaining heat; charging from spare only in hours without need."""
    d = 0.0
    unmet = np.zeros(need.shape[0])
    for h in range(need.shape[0]):
        n = need[h]
        if n > 0.0:
            n = max(n - c, 0.0)
            take = min(n, cap - d)
            d += take
            unmet[h] = n - take
        else:
            d = max(0.0, d - spare[h])
    return unmet


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


TOL = 1e-9  # an hour has unmet duty when unmet > 1e-9 (model/evaluate.py convention)


def feasible(need, spare, cap, c):
    return bounded(need, spare, cap, c).max() <= TOL


def min_trim(need, spare, cap):
    if feasible(need, spare, cap, 0.0):
        return 0.0
    lo, hi = 0.0, float(need.max())
    assert feasible(need, spare, cap, hi)
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        if feasible(need, spare, cap, mid):
            hi = mid
        else:
            lo = mid
    return hi


def run_cell(args):
    cell, table = args
    out = OUT / f"{cell}.json"
    if out.exists():
        return cell, "skip"
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = E.load_weather(table)
    dall = json.loads((ROOT / "experiments/phase1/out/cmip/deltas_all.json").read_text(encoding="utf-8"))
    exp, q, detrend = SCEN["ssp245_median"]
    Ws = {"historical": W, "stress_plus1.2K": E.shift_weather(W, 1.2),
          "ssp245_median": scenario_weather(W, np.array(dall[exp][q])[:, dall["cells"].index(cell)], detrend)}
    rec = {"cell": cell, "commit": COMMIT, "table": str(table)}
    kmax = max(D.k_grid(P, lim))  # 1.4948, the size behind every headline h*
    for cl, Wx in Ws.items():
        _, need, spare, _ = need_spare(P, Wx, kmax, lim)
        ey = np.array([E.sequent_peak(need[y], spare[y])[0] for y in range(need.shape[0])])
        order = np.argsort(-ey, kind="stable")  # ties -> earliest year first
        e_sorted = ey[order]
        yw = int(order[0])
        cap = float(e_sorted[1])
        r = {"E_sorted_top4": [float(x) for x in e_sorted[:4]], "worst_year_index": yw, "worst_year": int(np.asarray(W["years"])[yw]),
             "n_years": int(ey.shape[0])}
        if e_sorted[0] > e_sorted[1] + 1e-12:
            um = bounded(need[yw], spare[yw], cap)
            mask = um > TOL
            r.update({"unmet_h": float(um.sum()), "hours_unmet": int(mask.sum()), "longest_run_h": longest_run(mask),
                      "peak_unmet": float(um.max()), "min_trim": float(min_trim(need[yw], spare[yw], cap))})
            chk = bounded(need[yw], spare[yw], cap, r["min_trim"])
            r["min_trim_check_unmet"] = float(chk.sum())
        else:
            r.update({"unmet_h": 0.0, "hours_unmet": 0, "longest_run_h": 0, "peak_unmet": 0.0, "min_trim": 0.0, "min_trim_check_unmet": 0.0})
        sp0 = screen_path(P, Wx, lim, 0)
        r["m0_screen"] = {"F_dry": bool(sp0["F_dry"]), "F_nc": bool(sp0["F_nc"]), "viol_dry": sp0["viol_dry"],
                          "viol_ad": sp0["viol_ad"], "viol_ct": sp0["viol_ct"], "h_max": sp0["h_star"]}
        rec[cl] = r
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float), encoding="utf-8")
    os.replace(tmp, out)
    return cell, "done"


def wmed(x, w):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def summary():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    recs = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in OUT.glob("cell_*.json")}
    assert len(recs) == sites.cell_id.nunique(), (len(recs), sites.cell_id.nunique())
    w = sites.B_full_MW.astype(float).to_numpy()
    res = {"commit": COMMIT, "frame_GW": float(w.sum() / 1e3), "cells": len(recs)}
    rows = []
    gate = {}
    for cl in CLIMS:
        reb = sites.cell_id.map(lambda c: json.loads((REB / f"{c}.json").read_text(encoding="utf-8"))[cl]["h_star"]).to_numpy()
        es = np.array([recs[c][cl]["E_sorted_top4"] for c in sites.cell_id])
        gate[cl] = {"max_abs_diff_E2_vs_hstar": float(np.abs(es[:, 1] - reb).max()),
                    "F_dry_8h_m1_GW": float(w[es[:, 1] <= 8 + 1e-9].sum() / 1e3)}
        blk = {}
        for m in range(4):
            e = es[:, m]
            feas = e <= 8 + 1e-9
            blk[f"m{m}"] = {"F_dry_curve_GW": {str(h): float(w[e <= h + 1e-9].sum() / 1e3) for h in (0, 2, 4, 8, 12, 24)},
                            "median_store_h_feasible": wmed(e[feas], w[feas]) if feas.any() else None,
                            "mean_store_h_feasible": float(np.average(e[feas], weights=w[feas])) if feas.any() else None,
                            "stored_GWh_feasible": float((e[feas] * w[feas]).sum() / 1e3)}
        f1 = es[:, 1] <= 8 + 1e-9
        add = es[:, 0] - es[:, 1]
        g = lambda k: np.array([recs[c][cl][k] for c in sites.cell_id], float)  # noqa: E731
        unmet, hrs, run, peak, trim = g("unmet_h"), g("hours_unmet"), g("longest_run_h"), g("peak_unmet"), g("min_trim")
        pos = f1 & (unmet > TOL)
        blk["QB_remove_accepted_year"] = {"added_store_h_median_feasible_m1": wmed(add[f1], w[f1]), "added_store_h_max": float(add[f1].max()),
                                          "added_GWh": float((add[f1] * w[f1]).sum() / 1e3),
                                          "GW_within_8h_at_m0_of_m1_feasible": float(w[f1 & (es[:, 0] <= 8 + 1e-9)].sum() / 1e3)}
        blk["QC_excluded_year_with_m1_store"] = {
            "sites_with_shortfall": int(pos.sum()), "GW_with_shortfall": float(w[pos].sum() / 1e3),
            "fleet_unmet_GWh": float((unmet[f1] * w[f1]).sum() / 1e3),
            "unmet_h_median_over_shortfall_sites": wmed(unmet[pos], w[pos]) if pos.any() else 0.0, "unmet_h_max": float(unmet[f1].max()),
            "hours_unmet_max": int(hrs[f1].max()), "longest_run_h_max": int(run[f1].max()),
            "longest_run_h_median_over_shortfall_sites": wmed(run[pos], w[pos]) if pos.any() else 0.0,
            "peak_unmet_max": float(peak[f1].max()), "min_trim_GWth_fleet": float((trim[f1] * w[f1]).sum() / 1e3),
            "min_trim_per_unit_median_feasible_m1": wmed(trim[f1], w[f1]),  # registered cohort
            "hours_unmet_median_over_shortfall_sites": wmed(hrs[pos], w[pos]) if pos.any() else 0.0,  # registered, was missing
            "min_trim_per_unit_max": float(trim[f1].max()),
            "min_trim_check_max_unmet": float(max(recs[c][cl]["min_trim_check_unmet"] for c in sites.cell_id))}
        m0 = [recs[c][cl]["m0_screen"] for c in sites.cell_id]
        blk["QD_m0_screen"] = {"F_dry_GW": float(w[np.array([x["F_dry"] for x in m0])].sum() / 1e3),
                               "F_nc_GW": float(w[np.array([x["F_nc"] for x in m0])].sum() / 1e3)}
        res[cl] = blk
        for i, s in enumerate(sites.itertuples()):
            r = recs[s.cell_id][cl]
            rows.append({"climate": cl, "site_id": s.site_id, "cell_id": s.cell_id, "B_full_MW": s.B_full_MW,
                         "E1": es[i, 0], "E2": es[i, 1], "E3": es[i, 2], "E4": es[i, 3], "worst_year": r["worst_year"],
                         "unmet_h": r["unmet_h"], "hours_unmet": r["hours_unmet"], "longest_run_h": r["longest_run_h"],
                         "peak_unmet": r["peak_unmet"], "min_trim": r["min_trim"], "m0_F_dry": r["m0_screen"]["F_dry"],
                         "m0_F_nc": r["m0_screen"]["F_nc"]})
    res["consistency_gate"] = gate
    res["consistency_gate_pass"] = bool(all(v["max_abs_diff_E2_vs_hstar"] <= 1e-9 for v in gate.values()))
    expected = {"historical": 34.289631374, "stress_plus1.2K": 28.194631374, "ssp245_median": 29.051631374}  # consistency gate
    for cl, v in expected.items():
        if not (res["consistency_gate_pass"] and abs(gate[cl]["F_dry_8h_m1_GW"] - v) <= 1e-9):
            raise SystemExit(f"CONSISTENCY GATE FAILED ({cl}) — stop")
    pd.DataFrame(rows).to_csv(OUT / "w3_sites.csv", index=False)
    (OUT / "w3_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part in ("cells", "all"):
        pts = pd.read_csv(ROOT / "experiments/phase1/out/m2/cds_points.csv")
        cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
        exp = expected_hashes()
        todo = []
        for c in cells:
            row = pts[pts.names.str.contains(f"hub_cells:{c}(?:;|$)", regex=True)]
            assert len(row) == 1, c
            t = CDS / f"{row.point.iloc[0]}.npz"
            key = f"D:/_caches/era5/cds/tables/{row.point.iloc[0]}.npz"
            got = sha256(t)
            if exp.get(key) != got:
                raise SystemExit(f"CHECKSUM MISMATCH {t}: {got} vs {exp.get(key)} — stop")
            if not (OUT / f"{c}.json").exists():
                todo.append((c, t))
        print(f"{len(cells)} cells, checksums OK, {len(todo)} pending", flush=True)
        with mp.Pool(a.workers) as pool:
            for n, r in enumerate(pool.imap_unordered(run_cell, todo), 1):
                print(n, r, flush=True)
    if a.part in ("summary", "all"):
        summary()


if __name__ == "__main__":
    main()
