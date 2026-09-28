"""Package 3 (hardware sensitivity; ledger PR-HW): factorial feasibility screen on the full frame (80 cells × historical / +1.2 K)
for the liquid path at T_TCS ∈ {40, 45, 50} °C (facility-water limit T_TCS − 4 K) and the air path of hybrid racks at supply-air
limits T_sa ∈ {27, 32, 35} °C (air-path water limit T_sa − 3 K). Per path: h*(k_max) (sequent peak, m = 1) and the Phase-0 exact
short-circuit screen (dry k_max + 8 h store → F_dry; else adiabatic k_max + 8 h store; else tower kct_max → F_nc). Site
classifications for air fraction f_air > 0 combine both paths (both must hold); f_air = 0 uses the liquid path only.
Usage: python experiments/phase1/hw_screen.py [--workers 2]
Output: experiments/phase1/out/hw/<cell>__<climate>.json (atomic; resumable).
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
from bv1_min_storage import CLIMATES, COMMIT, need_spare  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / "experiments/phase1/out/hw"
LIQUID_T = [40.0, 45.0, 50.0]
AIR_TSA = [27.0, 32.0, 35.0]


def screen_path(P, W, lim, m):
    kmax = D.k_max(P, lim)
    _, need, spare, _ = need_spare(P, W, max(D.k_grid(P, lim)), lim)
    ey = np.array([E.sequent_peak(need[y], spare[y])[0] for y in range(need.shape[0])])
    viol_dry = int(np.sum(E.eval_dry(P, W, kmax, False, 8.0, lim)["cch_req"] > 1e-9))
    viol_ad = viol_ct = None
    if viol_dry > m:
        viol_ad = int(np.sum(E.eval_dry(P, W, kmax, True, 8.0, lim)["cch_req"] > 1e-9))
        if viol_ad > m:
            viol_ct = int(np.sum(E.eval_tower(P, W, max(P["kct_grid"]), lim)["cch_req"] > 1e-9))
    f_nc = viol_dry <= m or (viol_ad is not None and viol_ad <= m) or (viol_ct is not None and viol_ct <= m)
    return {"lim": lim, "h_star": float(np.sort(ey)[::-1][m]), "viol_dry": viol_dry, "viol_ad": viol_ad, "viol_ct": viol_ct,
            "F_dry": viol_dry <= m, "F_nc": bool(f_nc)}


def run_item(args):
    cell, climate, dT, wx_dir = args
    out = OUT / f"{cell}__{climate}.json"
    if out.exists():
        return cell, climate, "skip"
    t0 = time.time()
    W = E.load_weather(pathlib.Path(wx_dir) / f"{cell}.npz")
    if dT:
        W = E.shift_weather(W, dT)
    rec = {"cell": cell, "climate": climate, "dT": dT, "commit": COMMIT, "liquid": {}, "air": {}}
    for T in LIQUID_T:
        P = PR.params(ch_etaII=0.5076, T_TCS_max=T)
        m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
        rec["liquid"][f"{T:g}"] = screen_path(P, W, PR.fws_max(P), m)
    for Tsa in AIR_TSA:
        P = PR.params(ch_etaII=0.5076, T_sa_max=Tsa)
        m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
        rec["air"][f"{Tsa:g}"] = screen_path(P, W, P["T_sa_max"] - P["dT_coil"], m)
    rec["m"] = m
    rec["elapsed_s"] = round(time.time() - t0, 1)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, climate, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--weather-dir", default="D:/_caches/era5/weather_hub")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    items = [(c, cl, dT, a.weather_dir) for c in cells for cl, dT in CLIMATES.items() if not (OUT / f"{c}__{cl}.json").exists()]
    print(f"{len(items)} items pending; commit {COMMIT}", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)} {r}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
