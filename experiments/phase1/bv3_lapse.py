"""BV-3 addendum, sensitivity (iv) (DD-014): lapse-rate-adjusted weather for the ten sites whose DEM elevation differs from the
ERA5 cell orography by more than 100 m. Uniform RH-preserving shift ΔT = −0.0065·Δz on top of the uncorrected / corrected
tables, historical and +1.2 K. Per item: h*(k_max) (reset and continuous sequent peak, m = 1) and the Phase-0 hub-screen
classification (dry k_max + 8 h tank; adiabatic k_max + 8 h tank; tower kct_max; exact short-circuit).
Usage: python experiments/phase1/bv3_lapse.py [--workers 2]
Output: experiments/phase1/out/bv3/lapse/<site>__<weather>__<climate>.json
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
from bv1_min_storage import CLIMATES, COMMIT, need_spare  # noqa: E402
from bv2_continuous import sequent_peak_continuous  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

WEATHER = {"uncorrected": "D:/_caches/era5/weather_hub", "corrected": "D:/_caches/era5/weather_hub_corrected"}
OUT = ROOT / "experiments/phase1/out/bv3/lapse"
LAPSE = 0.0065  # K per m


def run_item(args):
    site, cell, dz, wx, climate, dT = args
    out = OUT / f"{site}__{wx}__{climate}.json"
    if out.exists():
        return site, wx, climate, "skip"
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    kmax = D.k_max(P, lim)
    shift = dT - LAPSE * dz
    W = E.shift_weather(E.load_weather(pathlib.Path(WEATHER[wx]) / f"{cell}.npz"), shift)
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    _, need, spare, _ = need_spare(P, W, max(D.k_grid(P, lim)), lim)
    reset = np.array([E.sequent_peak(need[y], spare[y])[0] for y in range(need.shape[0])])
    cont, _ = sequent_peak_continuous(np.ascontiguousarray(need), np.ascontiguousarray(spare))
    viol_dry = int(np.sum(E.eval_dry(P, W, kmax, False, 8.0, lim)["cch_req"] > 1e-9))
    viol_ad = viol_ct = None
    if viol_dry > m:
        viol_ad = int(np.sum(E.eval_dry(P, W, kmax, True, 8.0, lim)["cch_req"] > 1e-9))
        if viol_ad > m:
            viol_ct = int(np.sum(E.eval_tower(P, W, max(P["kct_grid"]), lim)["cch_req"] > 1e-9))
    f_nc = viol_dry <= m or (viol_ad is not None and viol_ad <= m) or (viol_ct is not None and viol_ct <= m)
    rec = {"site": site, "cell": cell, "weather": wx, "climate": climate, "dz_m": dz, "lapse_shift_K": -LAPSE * dz,
           "total_shift_K": shift, "m": m, "h_star_reset": float(np.sort(reset)[::-1][m]),
           "h_star_continuous": float(np.sort(cont)[::-1][m]), "viol_dry_kmax_tank8": viol_dry, "viol_ad_kmax_tank8": viol_ad,
           "viol_tower_kctmax": viol_ct, "F_dry": viol_dry <= m, "F_nc": bool(f_nc), "commit": COMMIT}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return site, wx, climate, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    s = pd.read_csv(ROOT / "data/sites/hub_sites_elevation.csv")
    s = s[(s.era5_orog_m > -900) & (s.dz_site_minus_cell_m.abs() > 100)]
    items = [(r.site_id, r.cell_id, float(r.dz_site_minus_cell_m), wx, cl, dT) for r in s.itertuples()
             for wx in WEATHER for cl, dT in CLIMATES.items() if not (OUT / f"{r.site_id}__{wx}__{cl}.json").exists()]
    print(f"{len(items)} items pending ({len(s)} sites)", flush=True)
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(n, r, flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
