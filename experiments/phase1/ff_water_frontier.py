"""Package 2 fix after internal review full-frame-consequences R1 (blocking finding): the finite-water compressor frontier (PR-FF E2) used a
pads-first adiabatic schedule, which does not minimize water when a store is installed. This adds a second causal witness policy,
storage-first: in an hour where dry mode cannot carry the load, discharge the store if it covers the whole dry-mode shortfall (pads
off); otherwise switch the existing pads on and discharge what remains; recharge only from dry-mode spare capacity (as before).
Per item and dry-cooler size k, for every store on the grid (2, 4, 8 h) and the exact pads-first minimal store: the risk-sized chiller
under this policy (yearly minimum by bisection, then the 2nd-largest year, m = 1) and the water it then uses (mean and maximum annual
m³ per MW, maximum hourly draw). The aggregator takes, for each budget, the smallest chiller over BOTH policies (and towers, dry-only):
a witness-based value, i.e. an upper bound on the physical minimum; only the zero-water (ban) value is exact within the domain.
Only items whose finite-water C_phys was positive at some store ≥ 2 h are recomputed (all others are already 0 and cannot fall).
Usage: python experiments/phase1/ff_water_frontier.py [--workers 3] --tag TAG --weather-dir DIR
Output: experiments/phase1/out/ff_water/<tag>/<cell>__<climate>.json
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
from numba import njit  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import fleet as F  # noqa: E402
from model import params as PR  # noqa: E402

COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
CLIMATES = {"historical": 0.0, "stress_plus1.2K": 1.2}


@njit(cache=False)
def sf_year(need_dry, need_ad, spare, w_ad, e_max, charge_max, c_ch):
    """One site-year under the storage-first policy: (worst unmet, water m3 per MW, max hourly draw, pad hours)."""
    e = e_max
    worst = 0.0
    water = 0.0
    wpeak = 0.0
    pads = 0
    for h in range(need_dry.shape[0]):
        nd = need_dry[h]
        if nd > 0.0:
            rd = nd - min(c_ch, nd)
            if rd <= e:
                e -= rd
            else:
                na = need_ad[h]
                ra = na - min(c_ch, na)
                d = min(ra, e)
                e -= d
                if ra - d > worst:
                    worst = ra - d
                water += w_ad[h]
                pads += 1
                if w_ad[h] > wpeak:
                    wpeak = w_ad[h]
        elif spare[h] > 0.0:
            c = min(spare[h], charge_max, e_max - e)
            e += c
    return worst, water, wpeak, pads


@njit(cache=False)
def sf_min_chiller(need_dry, need_ad, spare, w_ad, e_max, charge_max):
    hi = 0.0
    for h in range(need_dry.shape[0]):
        if need_dry[h] > hi:
            hi = need_dry[h]
    w, _, _, _ = sf_year(need_dry, need_ad, spare, w_ad, e_max, charge_max, 0.0)
    if w <= 1e-9:
        return 0.0
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        w, _, _, _ = sf_year(need_dry, need_ad, spare, w_ad, e_max, charge_max, mid)
        if w <= 1e-9:
            hi = mid
        else:
            lo = mid
    return hi


def run_item(args):
    cell, climate, dT, wx_dir, out_dir = args
    out = pathlib.Path(out_dir) / f"{cell}__{climate}.json"
    if out.exists():
        return cell, climate, "skip"
    t0 = time.time()
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = E.load_weather(pathlib.Path(wx_dir) / f"{cell}.npz")
    if dT:
        W = E.shift_weather(W, dT)
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    cache = F.SiteCache(P, W, lim)
    cm = P["tank_charge_max"]
    rows = []
    for k in D.k_grid(P, lim):
        a = cache.k_arrays(k, True)
        nd, na, sp, wa = a["need_dry"], a["need_ad"], a["spare"], a["w_ad"]
        ey = np.array([E.sequent_peak(na[y], sp[y])[0] for y in range(na.shape[0])])
        hs = float(np.sort(ey)[::-1][m])
        tanks = [2.0, 4.0, 8.0] + ([hs + 1e-7] if 1e-9 < hs <= 8.0 and not any(abs(hs - g) < 1e-6 for g in (2.0, 4.0, 8.0)) else [])
        for th in tanks:
            req = np.array([sf_min_chiller(nd[y], na[y], sp[y], wa[y], th, cm) for y in range(nd.shape[0])])
            cc = float(np.sort(req)[::-1][m])
            cc = cc if cc > 1e-9 else 0.0
            yr = [sf_year(nd[y], na[y], sp[y], wa[y], th, cm, cc) for y in range(nd.shape[0])]
            n_viol = int(sum(1 for r in yr if r[0] > 1e-9))
            rows.append({"k": float(k), "tank_h": float(th), "policy": "storage_first", "c_ch": cc, "n_viol": n_viol,
                         "compliant": n_viol <= m, "water_m3_mean": float(np.mean([r[1] for r in yr])),
                         "water_m3_max_year": float(max(r[1] for r in yr)), "water_peak_m3h": float(max(r[2] for r in yr)),
                         "pad_hours_mean": float(np.mean([r[3] for r in yr]))})
        cache._k.pop(float(k), None)  # free this k's hourly arrays (memory only; results unchanged)
    rec = {"cell": cell, "climate": climate, "dT": dT, "m": m, "commit": COMMIT, "designs": rows,
           "elapsed_s": round(time.time() - t0, 1)}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, climate, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weather-dir", required=True)
    a = ap.parse_args()
    out_dir = ROOT / "experiments/phase1/out/ff_water" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    d = pd.read_csv(ROOT / f"experiments/phase1/out/ff_sites_{a.tag}.csv")
    cols = [c for c in d.columns if c.startswith(("cphys_w", "cphys_peak50")) and not c.endswith("_h0")]
    pos = d[(d[cols].fillna(0) > 1e-9).any(axis=1)]
    todo = sorted(set(zip(pos.cell_id, pos.climate)))
    items = [(c, cl, CLIMATES[cl], a.weather_dir, str(out_dir)) for c, cl in todo if not (out_dir / f"{c}__{cl}.json").exists()]
    (out_dir / "expected.txt").write_text(str(len(todo)), encoding="utf-8")
    print(f"{len(items)} items pending of {len(todo)}; commit {COMMIT}; tag {a.tag}", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)} {r}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
