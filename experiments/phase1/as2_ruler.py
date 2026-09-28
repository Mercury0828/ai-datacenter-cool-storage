"""PR-AS2 AS2-2: engineering ruler for the approach axes — h*(k_max) for every cell × climate (uncorrected weather) on the grid
dry-cooler approach floor {3, 3.5, 4, 4.5, 5} K × plate-HX approach {1, 1.5, 2, 2.5, 3} K (k_max re-derived from each floor).
Usage: python experiments/phase1/as2_ruler.py [--workers 3]
Output: experiments/phase1/out/as2/ruler/<cell>__<climate>.json
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

import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from as_stress import COMMIT, hstar, need_spare  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

CLIMATES = {"historical": 0.0, "stress_plus1.2K": 1.2}
OUT = ROOT / "experiments/phase1/out/as2/ruler"
WX = pathlib.Path("D:/_caches/era5/weather_hub")
FLOORS = [3.0, 3.5, 4.0, 4.5, 5.0]
HXS = [1.0, 1.5, 2.0, 2.5, 3.0]


def run_item(args):
    cell, climate, dT = args
    out = OUT / f"{cell}__{climate}.json"
    if out.exists():
        return cell, climate, "skip"
    t0 = time.time()
    W = E.load_weather(WX / f"{cell}.npz")
    if dT:
        W = E.shift_weather(W, dT)
    grid = {}
    for fl in FLOORS:
        for hx in HXS:
            P = PR.params(ch_etaII=0.5076, dc_approach_floor=fl, dT_HX_design=hx)
            lim = PR.fws_max(P)
            m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
            k = D.k_max(P, lim)
            need, spare = need_spare(P, W, k, lim, P["tank_charge_max"])
            grid[f"floor{fl:g}_hx{hx:g}"] = {"k_max": float(k), "h_star": hstar(need, spare, m)}
    rec = {"cell": cell, "climate": climate, "commit": COMMIT, "grid": grid, "elapsed_s": round(time.time() - t0, 1)}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, climate, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    items = [(c, cl, dT) for c in cells for cl, dT in CLIMATES.items() if not (OUT / f"{c}__{cl}.json").exists()]
    print(f"{len(items)} items pending; commit {COMMIT}", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)} {r}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
