"""PR-R2F amendment 2 (F3): per-year delta = 0 operating cost of every DD-010 design on the R2 training years 1990-2014 (raw weather),
without and with an unlimited chiller, so that inner fits use fitting-year means and inner scoring uses the omitted block's years.
The 25-year mean must reproduce the registered op_tr (checked). Output: experiments/phase1/out/r2f/opyear/<site>.npz (op[j, 2, 25])
Usage: python experiments/phase1/r2f_opyear.py [--workers 2]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
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
from r2_dro import ETA2, design_list, req_and_opex  # noqa: E402
from r2f_precompute import r2_sites, train_weather  # noqa: E402

OUT = ROOT / "experiments/phase1/out/r2f/opyear"


def run_site(site):
    out = OUT / f"{site}.npz"
    if out.exists():
        return site, "skip"
    P = PR.params(ch_etaII=ETA2)
    lim = PR.fws_max(P)
    W = train_weather(site)
    ds = design_list(P, lim)
    op = np.zeros((len(ds), 2, len(W["years"])))
    for j, d in enumerate(ds):
        _, ox = req_and_opex(P, W, d, lim, want_opex=True)
        for c, o in enumerate(ox):
            op[j, c] = o["elec_MWh"] * P["elec_price"] + o["water_m3"] * P["water_price"]
    reg = np.load(ROOT / f"experiments/phase0/out/r2/sites/{site}.npz")["op_tr"]
    dev = float(np.max(np.abs(op.mean(axis=2) - reg) / np.maximum(np.abs(reg), 1.0)))
    tmp = OUT / f"{site}.tmp.npz"
    np.savez_compressed(tmp, op=op, years=W["years"], max_rel_dev_vs_op_tr=dev)
    os.replace(tmp, out)
    return site, f"done (max rel. deviation of the 25-year mean from op_tr {dev:.2e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_site, r2_sites()):
            print(r, flush=True)


if __name__ == "__main__":
    main()
