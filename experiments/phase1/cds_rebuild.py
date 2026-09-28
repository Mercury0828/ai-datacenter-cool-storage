"""DD-017 reconciliation and headline rebuild on the CDS tables (native surface pressure).
Part 1 (reconcile): every CDS point vs every mirror table that uses the same ERA5 point — max |dT|, max |dTd| (agreement ≤ 0.05 K)
and the derived-minus-native pressure (mean / max) per table.
Part 2 (rebuild): for the 80 hub cells, the headline screen (`hw_screen.screen_path`: h*(k_max) BV-1 kernel, F_dry at 8 h with the
tank, F_nc) on CDS weather — historical, +1.2 K (uniform RH-preserving), SSP2-4.5 circa 2040 (PR-CMIP deltas) — beside the
mirror-based items; fleet F_dry(h) curve, F_dry(8 h), F_nc (B-full) and the capacity changing class.
Usage: python experiments/phase1/cds_rebuild.py [--workers 2] [--part reconcile|rebuild|all]
Outputs: experiments/phase1/out/m2/cds_reconcile.json; experiments/phase1/out/m2/cds_rebuild/<cell>.json; cds_rebuild_summary.json
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
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from cmip_scenarios import SCEN, scenario_weather  # noqa: E402
from hw_screen import screen_path  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

CDS = pathlib.Path("D:/_caches/era5/cds/tables")
OUT = ROOT / "experiments/phase1/out/m2"
MIRROR_DIRS = {"hub_cells": pathlib.Path("D:/_caches/era5/weather_hub"), "bv3_stations": pathlib.Path("D:/_caches/era5/weather_stations"),
               "as2_uscrn_stations": pathlib.Path("D:/_caches/era5/weather_uscrn"), "md_hubs": ROOT / "data/weather_mdhubs",
               "nonepoch_hot_sites": ROOT / "data/weather_nonepoch", "phase0_sites": ROOT / "data/weather"}


def reconcile():
    pts = pd.read_csv(OUT / "cds_points.csv")
    rows = []
    for r in pts.itertuples():
        f = CDS / f"{r.point}.npz"
        if not f.exists():
            rows.append({"point": r.point, "status": "missing CDS table"})
            continue
        c = np.load(f)
        for name in r.names.split(";"):
            tab, site = name.split(":", 1)
            mf = MIRROR_DIRS[tab] / f"{site}.npz"
            if not mf.exists():
                rows.append({"point": r.point, "table": name, "status": "no mirror table"})
                continue
            m = np.load(mf)
            assert int(m["n_hours"]) == int(c["n_hours"]) and str(m["t0"]).replace("Z", "") == str(c["t0"]).replace("Z", "")
            dt = (m["t2m_c100"].astype(np.int32) - c["t2m_c100"].astype(np.int32)) / 100.0
            dd = (m["td_c100"].astype(np.int32) - c["td_c100"].astype(np.int32)) / 100.0
            dp = (m["psfc_pa10"].astype(np.int32) - c["psfc_pa10"].astype(np.int32)) * 10.0
            rows.append({"point": r.point, "table": name, "status": "ok", "max_abs_dT": float(np.abs(dt).max()), "mean_dT": float(dt.mean()),
                         "max_abs_dTd": float(np.abs(dd).max()), "mean_dTd": float(dd.mean()), "mean_dp_Pa": float(dp.mean()),
                         "max_abs_dp_Pa": float(np.abs(dp).max()), "mean_dp_pct": float(100 * dp.mean() / (c["psfc_pa10"].mean() * 10.0))})
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    res = {"tables_compared": int(len(ok)), "points": int(pts.shape[0]), "missing": df[df.status != "ok"].to_dict(orient="records"),
           "max_abs_dT": float(ok.max_abs_dT.max()), "max_abs_dTd": float(ok.max_abs_dTd.max()),
           "T_Td_agreement": bool(ok.max_abs_dT.max() <= 0.05 + 1e-9 and ok.max_abs_dTd.max() <= 0.05 + 1e-9),
           "pressure_mean_dp_pct": {"min": float(ok.mean_dp_pct.min()), "median": float(ok.mean_dp_pct.median()), "max": float(ok.mean_dp_pct.max())},
           "worst_T": ok.sort_values("max_abs_dT", ascending=False).head(5)[["table", "max_abs_dT"]].to_dict(orient="records"),
           "worst_p": ok.sort_values("mean_dp_pct", ascending=False).head(5)[["table", "mean_dp_pct", "max_abs_dp_Pa"]].to_dict(orient="records")}
    df.to_csv(OUT / "cds_reconcile_tables.csv", index=False)
    (OUT / "cds_reconcile.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("missing",)}, indent=1))
    return res


def cds_file_for_cell(cell, pts):
    row = pts[pts.names.str.contains(f"hub_cells:{cell}(?:;|$)", regex=True)]
    assert len(row) == 1, cell
    return CDS / f"{row.point.iloc[0]}.npz"


def rebuild_cell(args):
    cell, cds_file = args
    out = OUT / "cds_rebuild" / f"{cell}.json"
    if out.exists():
        return cell, "skip"
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = E.load_weather(cds_file)
    dall = json.loads((ROOT / "experiments/phase1/out/cmip/deltas_all.json").read_text(encoding="utf-8"))
    exp, q, detrend = SCEN["ssp245_median"]
    Ws = {"historical": W, "stress_plus1.2K": E.shift_weather(W, 1.2),
          "ssp245_median": scenario_weather(W, np.array(dall[exp][q])[:, dall["cells"].index(cell)], detrend)}
    rec = {"cell": cell}
    for cl, Wx in Ws.items():
        m = D.allowed_violations(P["eta_risk"], Wx["t_db"].shape[0])
        rec[cl] = screen_path(P, Wx, lim, m)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float), encoding="utf-8")
    os.replace(tmp, out)
    return cell, "done"


def mirror_values(cell):
    bv1 = {cl: json.loads((ROOT / f"experiments/phase1/out/bv1/uncorrected/{cell}__{cl}.json").read_text(encoding="utf-8"))["k"]["1.4948"]["h_star"]
           for cl in ("historical", "stress_plus1.2K")}
    scen = json.loads((ROOT / f"experiments/phase1/out/cmip/scen/ssp245_median__{cell}.json").read_text(encoding="utf-8"))
    return {"historical": bv1["historical"], "stress_plus1.2K": bv1["stress_plus1.2K"], "ssp245_median": scen["h_star"]}


def summarize_rebuild():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    recs = {json.loads(f.read_text(encoding="utf-8"))["cell"]: json.loads(f.read_text(encoding="utf-8")) for f in (OUT / "cds_rebuild").glob("cell_*.json")}
    w = sites.B_full_MW.astype(float)
    res = {"cells": len(recs)}
    for cl in ("historical", "stress_plus1.2K", "ssp245_median"):
        h_c = sites.cell_id.map(lambda c: recs[c][cl]["h_star"])
        h_m = sites.cell_id.map(lambda c: mirror_values(c)[cl])
        fnc = sites.cell_id.map(lambda c: recs[c][cl]["F_nc"]).astype(bool)
        lost = sites[(h_m <= 8 + 1e-9) & (h_c > 8 + 1e-9)]
        gained = sites[(h_m > 8 + 1e-9) & (h_c <= 8 + 1e-9)]
        res[cl] = {"F_dry_8h_GW_cds": float(w[h_c <= 8 + 1e-9].sum() / 1e3), "F_dry_8h_GW_mirror": float(w[h_m <= 8 + 1e-9].sum() / 1e3),
                   "F_dry_curve_GW_cds": {str(x): float(w[h_c <= x + 1e-9].sum() / 1e3) for x in (0, 2, 4, 8, 12, 24, 48)},
                   "F_nc_GW_cds": float(w[fnc].sum() / 1e3), "max_abs_dh_star": float(np.abs(h_c - h_m).max()),
                   "median_dh_star": float(np.median(h_c - h_m)),
                   "class_lost": lost[["site_id", "B_full_MW"]].to_dict(orient="records"),
                   "class_gained": gained[["site_id", "B_full_MW"]].to_dict(orient="records")}
    (OUT / "cds_rebuild_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    (OUT / "cds_rebuild").mkdir(parents=True, exist_ok=True)
    if a.part in ("reconcile", "all"):
        reconcile()
    if a.part in ("rebuild", "all"):
        pts = pd.read_csv(OUT / "cds_points.csv")
        cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
        todo = [(c, cds_file_for_cell(c, pts)) for c in cells if not (OUT / "cds_rebuild" / f"{c}.json").exists()]
        print(f"{len(todo)} cells pending", flush=True)
        with mp.Pool(a.workers) as pool:
            for n, r in enumerate(pool.imap_unordered(rebuild_cell, todo), 1):
                print(n, r, flush=True)
        summarize_rebuild()


if __name__ == "__main__":
    main()
