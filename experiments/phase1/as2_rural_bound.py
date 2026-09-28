"""PR-AS2 AS2-1b: rural-site bound. On hot local days (daily max ≥ the cell record's 95th percentile of daily maxima, AS-4's
definition), hours below the day mean −1.70 K and hours above it +0.38 K (the median ERA5 − USCRN diurnal-range differences), dry
route, h*(k_max), all cells, both climates (uncorrected weather).
Usage: python experiments/phase1/as2_rural_bound.py [--workers 3]
Output: experiments/phase1/out/as2/rural/<cell>__<climate>.json
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
from as_stress import COMMIT, hstar, need_spare  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

CLIMATES = {"historical": 0.0, "stress_plus1.2K": 1.2}
OUT = ROOT / "experiments/phase1/out/as2/rural"
WX = pathlib.Path("D:/_caches/era5/weather_hub")
NIGHT, DAY = -1.70, +0.38


def rural_diurnal(W, lon):
    t = W["t_db"].reshape(-1).copy()
    off = int(round(lon / 15.0))
    tl = np.roll(t, off)
    n = tl.shape[0] // 24
    days = tl[: n * 24].reshape(n, 24)
    dmax, dmean = days.max(axis=1), days.mean(axis=1)
    hot = dmax >= np.quantile(dmax, 0.95)
    add = np.zeros_like(days)
    add[hot] = np.where(days[hot] < dmean[hot, None], NIGHT, DAY)
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
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = E.load_weather(WX / f"{cell}.npz")
    if dT:
        W = E.shift_weather(W, dT)
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    k = max(D.k_grid(P, lim))
    n0, s0 = need_spare(P, W, k, lim, P["tank_charge_max"])
    n1, s1 = need_spare(P, rural_diurnal(W, lon), k, lim, P["tank_charge_max"])
    rec = {"cell": cell, "climate": climate, "commit": COMMIT, "h_star_base": hstar(n0, s0, m), "h_star_rural": hstar(n1, s1, m)}
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
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(n, r, flush=True)
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    res = {}
    for cl in CLIMATES:
        recs = {json.loads(f.read_text(encoding="utf-8"))["cell"]: json.loads(f.read_text(encoding="utf-8"))
                for f in OUT.glob(f"*__{cl}.json")}
        w = sites.B_full_MW.astype(float)
        hb = sites.cell_id.map(lambda c: recs[c]["h_star_base"])
        hr = sites.cell_id.map(lambda c: recs[c]["h_star_rural"])
        res[cl] = {"F_dry_8h_GW_base": float(w[hb <= 8 + 1e-9].sum() / 1e3), "F_dry_8h_GW_rural": float(w[hr <= 8 + 1e-9].sum() / 1e3),
                   "gained_sites": sites.site_id[(hb > 8 + 1e-9) & (hr <= 8 + 1e-9)].tolist(),
                   "lost_sites": sites.site_id[(hb <= 8 + 1e-9) & (hr > 8 + 1e-9)].tolist()}
    (OUT.parent / "rural_bound_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
