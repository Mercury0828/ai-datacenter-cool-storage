"""PR-SB SB-2 / SB-3 (ledger PR-SB, declared before any run): minimum storage h* on the CDS tables at the 80 hub cells for
  - m:<model>   the base design (total approach 6 K) under each of the 33 per-model SSP2-4.5 2031-2050 monthly fields (PR-CMIP
                operator: trend-preserving monthly delta on detrended analogue years, as the headline scenario);
  - median_5K / median_7K   the SSP2-4.5 median field at total approach 5 / 7 K (plate-HX approach 1 / 3 K);
  - hist_6K     the historical record at the base design (check: must reproduce the CDS rebuild h*).
h* uses the headline screen's path (bv1_min_storage.need_spare at max(k_grid), sequent peak, the m-th largest year; exactly as
hw_screen.screen_path), so that the historical h* reproduces the CDS rebuild bit for bit (PR-SB clarification 1). One item per cell (weather loaded once); every variant is written as it completes (resumable).
Usage: python experiments/phase1/sb_screen.py [--workers 3]
Output: experiments/phase1/out/sb/screen/<cell>__<variant>.json
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
from bv1_min_storage import need_spare  # noqa: E402
from cds_rebuild import cds_file_for_cell  # noqa: E402
from cmip_scenarios import scenario_weather  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402
from r2f_cmip_2060 import MODELS  # the 33 valid SSP2-4.5 models (PR-CMIP set)  # noqa: E402

OUT = ROOT / "experiments/phase1/out/sb/screen"
CMIP = ROOT / "experiments/phase1/out/cmip"
HX = {"6K": 2.0, "5K": 1.0, "7K": 3.0, "6.5K": 2.5}
EXTRA = ["hist_5K", "hist_6.5K", "hist_7K", "s12_5K", "s12_7K", "p10_245_6K", "p90_585_6K"]  # PR-SB clarification 3


def variants(extra=False):
    return EXTRA if extra else ["hist_6K", "median_5K", "median_7K"] + [f"m:{m}" for m in MODELS]


def h_of(W, hx):
    P = PR.params(ch_etaII=0.5076, dT_HX_design=hx)
    lim = PR.fws_max(PR.params(ch_etaII=0.5076))  # facility-water limit; the plate-HX approach enters through P (as PR-CMIP)
    _, need, spare, _ = need_spare(P, W, max(D.k_grid(P, lim)), lim)
    ey = np.array([E.sequent_peak(need[y], spare[y])[0] for y in range(need.shape[0])])
    return float(np.sort(ey)[::-1][D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])])


def run_cell(args):
    cell, cds_file, extra = args
    todo = [v for v in variants(extra) if not (OUT / f"{cell}__{v.replace(':', '_')}.json").exists()]
    if not todo:
        return cell, "skip"
    W = E.load_weather(cds_file)
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    ci = cells.index(cell)
    med = np.array(json.loads((CMIP / "deltas_all.json").read_text(encoding="utf-8"))["ssp245"]["median"])[:, ci]
    dall = json.loads((CMIP / "deltas_all.json").read_text(encoding="utf-8"))
    for v in todo:
        if v == "hist_6K":
            h = h_of(W, HX["6K"])
        elif v.startswith("hist_"):
            h = h_of(W, HX[v.split("_")[1]])
        elif v.startswith("s12_"):
            h = h_of(E.shift_weather(W, 1.2), HX[v.split("_")[1]])
        elif v == "p10_245_6K":
            h = h_of(scenario_weather(W, np.array(dall["ssp245"]["p10"])[:, ci], True), HX["6K"])
        elif v == "p90_585_6K":
            h = h_of(scenario_weather(W, np.array(dall["ssp585"]["p90"])[:, ci], True), HX["6K"])
        elif v.startswith("median_"):
            h = h_of(scenario_weather(W, med, True), HX[v.split("_")[1]])
        else:
            d = np.array(json.loads((CMIP / f"deltas_{v[2:]}.json").read_text(encoding="utf-8"))["ssp245"])[:, ci]
            assert np.isfinite(d).all(), (v, cell)
            h = h_of(scenario_weather(W, d, True), HX["6K"])
        out = OUT / f"{cell}__{v.replace(':', '_')}.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cell": cell, "variant": v, "h_star": float(h)}), encoding="utf-8")
        os.replace(tmp, out)
    return cell, f"done {len(todo)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--extra", action="store_true", help="PR-SB clarification 3 variants")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pts = pd.read_csv(ROOT / "experiments/phase1/out/m2/cds_points.csv")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    items = [(c, cds_file_for_cell(c, pts), a.extra) for c in cells]
    print(f"{len(items)} cells x {len(variants(a.extra))} variants", flush=True)
    with mp.Pool(a.workers, maxtasksperchild=10) as pool:
        for n, r in enumerate(pool.imap_unordered(run_cell, items), 1):
            print(n, r, flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
