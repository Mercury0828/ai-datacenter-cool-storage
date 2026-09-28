"""PR-AS part A: assumption stress on headline A (F_dry at 8 h). Per cell × climate (uncorrected weather), h*(k_max) under the
registered variants: charge-rate limit, standby losses, usable fraction, warmer nights on hot days, dry-cooler approach floor,
plate-HX approach, and the combined adverse / favourable cases. Dry route only (t_db and pressure drive it).
Usage: python experiments/phase1/as_stress.py [--workers 3]
Output: experiments/phase1/out/as/<cell>__<climate>.json
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
from model import components as C  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
CLIMATES = {"historical": 0.0, "stress_plus1.2K": 1.2}
OUT = ROOT / "experiments/phase1/out/as"
WX = pathlib.Path("D:/_caches/era5/weather_hub")


@njit(cache=False)
def lossy_ok(need, spare, e_max, lam):
    e = e_max
    for h in range(need.shape[0]):
        e *= (1.0 - lam)
        if need[h] > 0.0:
            if need[h] > e + 1e-9:
                return False
            e -= need[h]
        elif spare[h] > 0.0:
            e = min(e_max, e + spare[h])
    return True


@njit(cache=False)
def lossy_min_store(need, spare, lam, hi):
    if lossy_ok(need, spare, 0.0, lam):
        return 0.0
    if not lossy_ok(need, spare, hi, lam):
        return np.inf
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if lossy_ok(need, spare, mid, lam):
            hi = mid
        else:
            lo = mid
    return hi


def need_spare(P, W, k, lim, charge_max):
    ch = C.DryChain(P, k, lim)
    tdb, p = W["t_db"], W["p"]
    qmax = ch.q_max(tdb, p)
    need = ch.trim_need(1.0, tdb, p)
    fwr = lim + 1.0 / ch.C_f
    spare = np.where(qmax >= 1.0, np.clip(ch.duty_fixed_inlet(tdb, p, fwr) - 1.0, 0.0, charge_max), 0.0)
    return need, spare


def hstar(need, spare, m, loss_per_day=0.0):
    ny = need.shape[0]
    if loss_per_day <= 0:
        ey = np.array([E.sequent_peak(need[y], spare[y])[0] for y in range(ny)])
    else:
        lam = 1.0 - (1.0 - loss_per_day) ** (1.0 / 24.0)
        ey = np.array([lossy_min_store(np.ascontiguousarray(need[y]), np.ascontiguousarray(spare[y]), lam, 2000.0) for y in range(ny)])
    return float(np.sort(ey)[::-1][m])


def warm_nights(W, lon, dT):
    """+dT on hours below the local-day mean, on local days whose maximum is >= the record's 95th percentile of daily maxima."""
    t = W["t_db"].reshape(-1).copy()
    off = int(round(lon / 15.0))
    tl = np.roll(t, off)
    n = tl.shape[0] // 24
    days = tl[: n * 24].reshape(n, 24)
    dmax, dmean = days.max(axis=1), days.mean(axis=1)
    hot = dmax >= np.quantile(dmax, 0.95)
    add = np.zeros_like(days)
    add[hot] = np.where(days[hot] < dmean[hot, None], dT, 0.0)
    tl2 = tl.copy()
    tl2[: n * 24] += add.reshape(-1)
    out = dict(W)
    out["t_db"] = np.roll(tl2, -off).reshape(W["t_db"].shape)
    return out


def run_item(args):
    cell, climate, dT, lon = args
    out = OUT / f"{cell}__{climate}.json"
    if out.exists():
        return cell, climate, "skip"
    t0 = time.time()
    P0 = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P0)
    W = E.load_weather(WX / f"{cell}.npz")
    if dT:
        W = E.shift_weather(W, dT)
    m = D.allowed_violations(P0["eta_risk"], W["t_db"].shape[0])
    k0 = max(D.k_grid(P0, lim))
    r = {}
    need, sp1 = need_spare(P0, W, k0, lim, 1.0)
    r["base"] = hstar(need, sp1, m)
    for cm in (0.5, 0.25):
        r[f"charge_{cm:g}"] = hstar(need, np.minimum(sp1, cm), m)
    for lp in (0.01, 0.03):
        r[f"loss_{lp:g}_per_day"] = hstar(need, sp1, m, lp)
    r["usable_0.8"] = r["base"] / 0.8
    for dn in (1.0, 2.0):
        n2, s2 = need_spare(P0, warm_nights(W, lon, dn), k0, lim, 1.0)
        r[f"nights_plus{dn:g}K"] = hstar(n2, s2, m)
    for fl in (3.0, 5.0):
        P = PR.params(ch_etaII=0.5076, dc_approach_floor=fl)
        k = D.k_max(P, lim)
        n2, s2 = need_spare(P, W, k, lim, 1.0)
        r[f"approach_floor_{fl:g}K"] = hstar(n2, s2, m)
        r[f"approach_floor_{fl:g}K_kmax"] = float(k)
    for hx in (1.0, 3.0):
        P = PR.params(ch_etaII=0.5076, dT_HX_design=hx)
        n2, s2 = need_spare(P, W, k0, lim, 1.0)
        r[f"hx_{hx:g}K"] = hstar(n2, s2, m)
    # combined adverse: charge 0.5, losses 1 %/day, usable 0.9, nights +1 K, approach floor 5 K, HX 3 K
    P = PR.params(ch_etaII=0.5076, dc_approach_floor=5.0, dT_HX_design=3.0)
    k = D.k_max(P, lim)
    n2, s2 = need_spare(P, warm_nights(W, lon, 1.0), k, lim, 0.5)
    r["combined_adverse"] = hstar(n2, s2, m, 0.01) / 0.9
    # combined favourable: approach floor 3 K, HX 1 K
    P = PR.params(ch_etaII=0.5076, dc_approach_floor=3.0, dT_HX_design=1.0)
    k = D.k_max(P, lim)
    n2, s2 = need_spare(P, W, k, lim, 1.0)
    r["combined_favourable"] = hstar(n2, s2, m)
    rec = {"cell": cell, "climate": climate, "commit": COMMIT, "h_star": r, "elapsed_s": round(time.time() - t0, 1)}
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
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv")
    items = [(c.site_id, cl, dT, float(c.era5_lon)) for c in cells.itertuples() for cl, dT in CLIMATES.items()
             if not (OUT / f"{c.site_id}__{cl}.json").exists()]
    print(f"{len(items)} items pending; commit {COMMIT}", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)} {r}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
