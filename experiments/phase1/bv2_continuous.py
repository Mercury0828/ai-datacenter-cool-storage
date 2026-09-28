"""BV-2 continuous-state check (DD-013): at k_max, the sequent peak run across 1 January (no annual reset) for every
cell × climate of one weather tag; h*_c = 2nd-largest within-year maximum of the continuous state. Built-in check: for years
entered with zero carried state, the within-year maximum equals the reset sequent peak (BV-1 kernel).
Usage: python experiments/phase1/bv2_continuous.py [--workers 2] [--weather-dir DIR] [--tag NAME]
Output: experiments/phase1/out/bv2_continuous/<tag>/<cell>__<climate>.json
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
from numba import njit  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from bv1_min_storage import CLIMATES, COMMIT, need_spare  # noqa: E402
from model import components as C  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402


@njit(cache=True)
def sequent_peak_continuous(need, spare):
    ny, nh = need.shape
    k = 0.0
    emax = np.zeros(ny)
    carry = np.zeros(ny)
    for y in range(ny):
        carry[y] = k
        for h in range(nh):
            if need[y, h] > 0.0:
                k += need[y, h]
            else:
                k -= spare[y, h]
                if k < 0.0:
                    k = 0.0
            if k > emax[y]:
                emax[y] = k
    return emax, carry


def run_item(args):
    cell, climate, dT, wx_dir, out_dir = args
    out = pathlib.Path(out_dir) / f"{cell}__{climate}.json"
    if out.exists():
        return cell, climate, "skip"
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = E.load_weather(pathlib.Path(wx_dir) / f"{cell}.npz")
    if dT:
        W = E.shift_weather(W, dT)
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    k = max(D.k_grid(P, lim))
    _, need, spare, _ = need_spare(P, W, k, lim)
    need, spare = np.ascontiguousarray(need), np.ascontiguousarray(spare)
    emax, carry = sequent_peak_continuous(need, spare)
    reset = np.array([E.sequent_peak(need[y], spare[y])[0] for y in range(need.shape[0])])
    z = carry == 0.0
    assert np.allclose(emax[z], reset[z], atol=1e-9), "continuous kernel disagrees with the reset kernel on zero-carry years"
    assert np.all(emax >= reset - 1e-9)
    # adiabatic-assisted route of the F_nc screen (dry k_max + adiabatic + tank; charging in dry mode only, as in eval_dry)
    ch = C.DryChain(P, k, lim)
    need_ad = np.ascontiguousarray(ch.trim_need(1.0, C.adiabatic_air_temp(W["t_db"], W["t_wb"], P["eta_ad"]), W["p"]))
    emax_ad, carry_ad = sequent_peak_continuous(need_ad, spare)
    reset_ad = np.array([E.sequent_peak(need_ad[y], spare[y])[0] for y in range(need_ad.shape[0])])
    rec = {"cell": cell, "climate": climate, "dT": dT, "m": m, "k": float(k), "commit": COMMIT,
           "h_star_reset": float(np.sort(reset)[::-1][m]), "h_star_continuous": float(np.sort(emax)[::-1][m]),
           "E_y_continuous": emax.tolist(), "carry_into_jan1": carry.tolist(), "max_carry": float(carry.max()),
           "years_with_carry": int(np.sum(carry > 1e-9)),
           "ad_h_star_reset": float(np.sort(reset_ad)[::-1][m]), "ad_h_star_continuous": float(np.sort(emax_ad)[::-1][m]),
           "ad_max_carry": float(carry_ad.max()), "ad_years_with_carry": int(np.sum(carry_ad > 1e-9))}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, climate, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--weather-dir", default="D:/_caches/era5/weather_hub")
    ap.add_argument("--tag", default="uncorrected")
    a = ap.parse_args()
    out_dir = ROOT / "experiments/phase1/out/bv2_continuous" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    items = [(c, cl, dT, a.weather_dir, str(out_dir)) for c in cells for cl, dT in CLIMATES.items()
             if not (out_dir / f"{c}__{cl}.json").exists()]
    print(f"{len(items)} items pending; tag {a.tag}", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(run_item, items), 1):
            if n % 20 == 0:
                print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
