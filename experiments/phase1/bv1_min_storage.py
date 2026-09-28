"""BV-1 (pre-registered in ledger PR-BV, commit 8bb6703): minimum storage h* for dry-only, compressor-free designs on the
full Epoch frame (80 ERA5 land cells), historical weather and the +1.2 K stress, for every dry-cooler size on the k-grid.

Per cell × climate: for each k, need_t (series-trim duty at full fan) and spare_t (charging capacity at the design return
temperature, capped) -> sequent peak per site-year -> E*_y; h*(k) = 2nd-largest E*_y (≤ 1 violating year of 36). BV-2 quantities
(charge temperature swing in the critical year) are recorded for k_max. One JSON per item, atomic; resumable; `--fresh` restarts.
Usage: python experiments/phase1/bv1_min_storage.py [--workers 4] [--fresh] [--weather-dir DIR] [--tag NAME] [--dT-extra X]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import components as C  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
CLIMATES = {"historical": 0.0, "stress_plus1.2K": 1.2}
RHO_CP_MWH_PER_M3K = 4.186e3 * 1000 / 3.6e9  # MWh per m3 per K (water)


def need_spare(P, W, k, lim):
    ch = C.DryChain(P, k, lim)
    tdb, p = W["t_db"], W["p"]
    q = 1.0
    qmax = ch.q_max(tdb, p)
    need = ch.trim_need(q, tdb, p)
    fwr = lim + q / ch.C_f
    spare = np.where(qmax >= q, np.clip(ch.duty_fixed_inlet(tdb, p, fwr) - q, 0.0, P["tank_charge_max"]), 0.0)
    return ch, need, spare, fwr


def run_item(args):
    cell, climate, dT, wx_dir, out_dir, dT_extra = args
    out = pathlib.Path(out_dir) / f"{cell}__{climate}.json"
    if out.exists():
        return cell, climate, "skip"
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = E.load_weather(pathlib.Path(wx_dir) / f"{cell}.npz")
    shift = dT + dT_extra
    if shift:
        W = E.shift_weather(W, shift)
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    rec = {"cell": cell, "climate": climate, "dT": shift, "m": m, "commit": COMMIT, "k": {}}
    for k in D.k_grid(P, lim):
        ch, need, spare, fwr = need_spare(P, W, k, lim)
        ey, dur, nh = [], [], []
        for y in range(need.shape[0]):
            a, b, c = E.sequent_peak(need[y], spare[y])
            ey.append(float(a)); dur.append(int(b)); nh.append(int(c))
        order = np.argsort(ey)[::-1]
        h_star = float(ey[order[m]]) if len(ey) > m else 0.0
        rec["k"][f"{k:.4f}"] = {"h_star": h_star, "E_y_top3": [float(ey[i]) for i in order[:3]],
                                "critical_year_idx": int(order[m]), "critical_episode_h": int(dur[order[m]]),
                                "need_hours_critical_year": int(nh[order[m]]), "years_with_need": int(np.sum(np.array(nh) > 0))}
        if abs(k - max(D.k_grid(P, lim))) < 1e-9 and h_star > 0:
            # BV-2: charge temperatures in the critical year with a tank of exactly h*
            y = order[m]
            _, _, _, charge = E._tank_year(need[y], spare[y], h_star, P["tank_charge_max"], 0.0)
            t_out = fwr - (1.0 + charge) / ch.C_f            # chain outlet while charging (fixed facility flow)
            swing = np.where(charge > 0, fwr - t_out, 0.0)  # usable swing of the stored water vs the return temperature
            w = charge.sum()
            mean_swing = float((charge * swing).sum() / w) if w > 0 else float("nan")
            rec["bv2"] = {"mean_charge_swing_K": mean_swing,
                          "volume_m3_per_MW_at_10K": h_star / (RHO_CP_MWH_PER_M3K * 10.0),
                          "volume_m3_per_MW_actual_swing": h_star / (RHO_CP_MWH_PER_M3K * mean_swing) if w > 0 else None}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, climate, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--weather-dir", default="D:/_caches/era5/weather_hub")
    ap.add_argument("--tag", default="uncorrected")
    ap.add_argument("--dT-extra", type=float, default=0.0)
    a = ap.parse_args()
    out_dir = ROOT / "experiments/phase1/out/bv1" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    if a.fresh:
        for f in out_dir.glob("*.json"):
            f.unlink()
    items = [(c, cl, dT, a.weather_dir, str(out_dir), a.dT_extra) for c in cells for cl, dT in CLIMATES.items()
             if not (out_dir / f"{c}__{cl}.json").exists()]
    print(f"{len(items)} items pending of {2 * len(cells)}; commit {COMMIT}; tag {a.tag}", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, (c, cl, st) in enumerate(pool.imap_unordered(run_item, items), 1):
            if n % 10 == 0:
                print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
