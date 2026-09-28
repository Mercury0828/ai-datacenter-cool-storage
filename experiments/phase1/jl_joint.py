"""PR-JL (ledger 2026-09-28, declared before any run): the 1 K lever under central warming combined with adverse nights.

For every hub cell, on the CDS data of record, the store h* (dry chain at the base k_max, 1-of-36 screen) under the central
SSP2-4.5 scenario around 2040, at the base 6 K total approach and with the plate-exchanger approach halved (5 K), and under
the same scenario combined with the two registered night perturbations of PR-AS / PR-AS2: airport-style (+1 K on the
below-mean hours of hot days) and rural-style (nights -1.70 K, days +0.38 K on hot days).
Writes experiments/phase1/out/jl/<cell>.json and experiments/phase1/out/jl_summary.json.
Usage: python experiments/phase1/jl_joint.py [--workers 3]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from as2_rural_bound import rural_diurnal  # noqa: E402
from as_stress import hstar, need_spare, warm_nights  # noqa: E402
from cmip_scenarios import SCEN, scenario_weather  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / "experiments/phase1/out/jl"
WX = pathlib.Path("D:/_caches/era5/cds/weather_hub_cds")


def run_item(args):
    cell, lon = args
    out = OUT / f"{cell}.json"
    if out.exists():
        return cell, "skip"
    t0 = time.time()
    P6 = PR.params(ch_etaII=0.5076)
    P5 = PR.params(ch_etaII=0.5076, dT_HX_design=1.0)
    lim = PR.fws_max(P6)
    k0 = max(D.k_grid(P6, lim))
    W = E.load_weather(WX / f"{cell}.npz")
    dall = json.loads((ROOT / "experiments/phase1/out/cmip/deltas_all.json").read_text(encoding="utf-8"))
    exp, q, detrend = SCEN["ssp245_median"]
    Ws = scenario_weather(W, np.array(dall[exp][q])[:, dall["cells"].index(cell)], detrend)
    m = D.allowed_violations(P6["eta_risk"], W["t_db"].shape[0])
    weathers = {"central": Ws, "central_airport_nights": warm_nights(Ws, lon, 1.0), "central_rural": rural_diurnal(Ws, lon)}
    r = {}
    for wn, Wx in weathers.items():
        for tag, P in (("6K", P6), ("5K", P5)):
            n, s = need_spare(P, Wx, k0, lim, 1.0)
            r[f"{wn}|{tag}"] = hstar(n, s, m)
    rec = {"cell": cell, "h_star": r, "elapsed_s": round(time.time() - t0, 1)}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, "done"


def summarize():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    mw = sites.groupby("cell_id").B_full_MW.sum()
    recs = {f.stem: json.loads(f.read_text(encoding="utf-8"))["h_star"] for f in OUT.glob("cell_*.json")}
    keys = sorted(next(iter(recs.values())).keys())
    res = {"cells": len(recs)}
    for k in keys:
        ok = [c for c, r in recs.items() if r[k] <= 8 + 1e-9]
        res[f"F_dry_8h_GW|{k}"] = float(mw.reindex(ok).sum() / 1e3)
    for wn in ("central", "central_airport_nights", "central_rural"):
        kept = [c for c, r in recs.items() if r[f"{wn}|5K"] <= 8 + 1e-9 < r[f"{wn}|6K"]]
        res[f"lever_kept_GW|{wn}"] = float(mw.reindex(kept).sum() / 1e3)
        res[f"lever_kept_cells|{wn}"] = sorted(kept)
    (ROOT / "experiments/phase1/out/jl_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv")
    items = [(c.site_id, float(c.era5_lon)) for c in cells.itertuples() if not (OUT / f"{c.site_id}.json").exists()]
    print(f"{len(items)} items pending", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)} {r}", flush=True)
    res = summarize()
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("lever_kept_cells")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
