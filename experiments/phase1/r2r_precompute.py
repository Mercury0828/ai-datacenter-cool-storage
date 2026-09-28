"""PR-R2R precompute: yearly chiller requirement of every R2 design (DD-010 menu) at the 15 R2 sites on the untouched CMIP6 test
weather (PR-CMIP trend-preserving monthly delta of the site's hub cell): ssp245_median (central test) and ssp585_p90 (secondary).
Usage: python experiments/phase1/r2r_precompute.py [--workers 3]
Output: experiments/phase1/out/r2r/<site>__<scenario>.npz (R_test[j, year], designs)
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
sys.path.insert(0, str(ROOT / "experiments/phase0"))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from cmip_scenarios import scenario_weather  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402
from r2_dro import ETA2, design_list, req_and_opex  # noqa: E402

OUT = ROOT / "experiments/phase1/out/r2r"
SCEN = {"ssp245_median": ("ssp245", "median"), "ssp585_p90": ("ssp585", "p90")}


def run_item(args):
    site, cell, scen = args
    out = OUT / f"{site}__{scen}.npz"
    if out.exists():
        return site, scen, "skip"
    dall = json.loads((ROOT / "experiments/phase1/out/cmip/deltas_all.json").read_text(encoding="utf-8"))
    exp, q = SCEN[scen]
    delta12 = np.array(dall[exp][q])[:, dall["cells"].index(cell)]
    P = PR.params(ch_etaII=ETA2)
    lim = PR.fws_max(P)
    W = scenario_weather(E.load_weather(ROOT / f"data/weather/{site}.npz"), delta12, True)
    ds = design_list(P, lim)
    R = np.zeros((len(ds), W["t_db"].shape[0]))
    for j, d in enumerate(ds):
        R[j] = req_and_opex(P, W, d, lim, want_opex=False)[0]
    tmp = OUT / f"{site}__{scen}.tmp.npz"
    np.savez_compressed(tmp, R_test=R, designs=np.array([json.dumps(d) for d in ds]), years=W["years"], delta12=delta12)
    os.replace(tmp, out)
    return site, scen, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    r2sites = [p.stem for p in (ROOT / "experiments/phase0/out/r2/sites").glob("*.npz") if not p.stem.endswith(("partial", "tmp"))]
    hub = pd.read_csv(ROOT / "data/sites/hub_sites.csv").set_index("site_id").cell_id
    items = [(s, hub[s], sc) for s in sorted(r2sites) for sc in SCEN if not (OUT / f"{s}__{sc}.npz").exists()]
    print(f"{len(items)} items pending ({len(r2sites)} sites)", flush=True)
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(n, r, flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
