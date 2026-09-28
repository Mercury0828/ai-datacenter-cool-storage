"""PR-R2F requirement arrays (ledger PR-R2F; DD-018). Yearly chiller requirement of every DD-010 design at the 15 R2 sites under each
climate world's SSP2-4.5 2051-2070 monthly field (operator G = cmip_scenarios.shift_hourly, RH-preserving, on detrended years).
  --part test : the unused ERA5 block 1960-1989 (CDS, native sp), each year detrended to 2004.5 with the site's linear annual-mean
                trend fitted on 1960-2014 (1960-1989 block + CDS 1990-2014), then the world's field; one item per (site, world).
  --part train: the R2 training years 1990-2014 (R2 weather tables), RAW (no detrending slope fitted or applied; PR-R2F
                amendment 2, F2 / F3), then the world's field through G; one item per (site, world).
  --part shift: Stage 2 (amendment 3): the 0.1 K grid of uniform RH-preserving shifts of the raw training years (= G with a constant
                monthly field) up to the largest shift reachable at the site (grid_max in out/r2f/shift_grid.json); grid values that
                exist in the R2 delta grid are reused from the R2 precompute and not recomputed.
  --part bmon : Stage 2: the four fixed-q B-MON fields per outer fold (family-balanced per-month lower q-quantile of the fold's
                training models' 2051-2070 deltas), through G on the raw training years; identical fields are computed once.
Output: experiments/phase1/out/r2f/<part>/<site>__<model>.npz (R[j, year], years, delta12, slope); shift/<site>__<sigma>.npz;
        bmon/<site>__<fieldhash>.npz with bmon_index.json
Usage: python experiments/phase1/r2f_precompute.py --part test|train|shift|bmon [--workers 3]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import functools  # noqa: E402
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
from cds_extract import point_key  # noqa: E402
from cmip_scenarios import MONTH, scenario_weather, shift_hourly  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402
from r2_dro import ETA2, design_list, req_and_opex  # noqa: E402

OUT = ROOT / "experiments/phase1/out/r2f"
T60 = pathlib.Path("D:/_caches/era5/cds/tables_1960")
CDS = pathlib.Path("D:/_caches/era5/cds/tables")
REF = 2004.5


def r2_sites():
    return sorted(p.stem for p in (ROOT / "experiments/phase0/out/r2/sites").glob("*.npz") if not p.stem.endswith(("partial", "tmp")))


def subset(W, keep):
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(W["years"]) else v) for k, v in W.items()}


@functools.lru_cache(maxsize=2)
def train_weather(site):
    W = E.load_weather(ROOT / f"data/weather/{site}.npz")
    return subset(W, (W["years"] >= 1990) & (W["years"] <= 2014))


@functools.lru_cache(maxsize=2)
def test_weather(site):
    """1960-1989 block and its 1960-2014 reference slope (annual means of the block + the CDS 1990-2014 series of the same point)."""
    W = E.load_weather(T60 / f"{site}.npz")
    assert W["years"][0] == 1960 and W["years"][-1] == 1989 and len(W["years"]) == 30
    s = pd.read_csv(ROOT / "data/sites/phase0_sites.csv").set_index("site_id").loc[site]
    Wc = E.load_weather(CDS / f"{point_key(round(float(s.era5_lat), 2), round(float(s.era5_lon), 2))}.npz")
    Wc = subset(Wc, (Wc["years"] >= 1990) & (Wc["years"] <= 2014))
    yrs = np.concatenate([W["years"], Wc["years"]]).astype(float)
    ann = np.concatenate([W["t_db"].mean(axis=1), Wc["t_db"].mean(axis=1)])
    slope = float(np.polyfit(yrs, ann, 1)[0])
    return W, slope


def fields():
    s = json.loads((OUT / "deltas_2060_all.json").read_text(encoding="utf-8"))
    return s, s["valid_models"]


R2_GRID = (0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.4, 3.0, 4.0, 5.0)
BMON_Q = (0.5, 0.75, 0.9, 1.0)


def shift_grid():
    return json.loads((OUT / "shift_grid.json").read_text(encoding="utf-8"))


def run_field_item(args):
    part, site, key, field12 = args
    out = OUT / part / f"{site}__{key}.npz"
    if out.exists():
        return site, key, "skip"
    P = PR.params(ch_etaII=ETA2)
    lim = PR.fws_max(P)
    W = scenario_weather(train_weather(site), np.asarray(field12, float), False)
    ds = design_list(P, lim)
    R = np.zeros((len(ds), W["t_db"].shape[0]))
    for j, dsg in enumerate(ds):
        R[j] = req_and_opex(P, W, dsg, lim, want_opex=False)[0]
    tmp = out.parent / f"{site}__{key}.tmp.npz"
    np.savez_compressed(tmp, R=R, years=W["years"], field12=np.asarray(field12, float))
    os.replace(tmp, out)
    return site, key, "done"


def bmon_fields():
    """{(site, fold, q): field12} and the deduplicated {(site, key): field12} (key = rounded-field hash)."""
    import hashlib
    s, models = fields()
    fams = s["families"]
    fam_of = {m: f for f, ms in fams.items() for m in ms}
    out, uniq = {}, {}
    for site in r2_sites():
        ci = s["cells"].index(s["site_cell"][site])
        for f in fams:
            ms = [m for m in models if fam_of[m] != f]
            nf = {g: sum(fam_of[m] == g for m in ms) for g in fams if g != f}
            w = np.array([1.0 / (len(nf) * nf[fam_of[m]]) for m in ms])
            A = np.stack([np.array(s["delta_2060"][m])[:, ci] for m in ms])
            for q in BMON_Q:
                fld = []
                for l in range(12):
                    o = np.argsort(A[:, l], kind="stable")
                    cw = np.cumsum(w[o])
                    fld.append(float(A[o, l][min(np.searchsorted(cw, q * cw[-1] - 1e-12), len(ms) - 1)]))
                key = hashlib.sha256(np.round(fld, 9).tobytes()).hexdigest()[:16]
                out[(site, f, q)] = key
                uniq[(site, key)] = fld
    return out, uniq


def run_item(args):
    part, site, model = args
    out = OUT / part / f"{site}__{model}.npz"
    if out.exists():
        return site, model, "skip"
    s, _ = fields()
    cell = s["site_cell"][site]
    delta12 = np.array(s["delta_2060"][model])[:, s["cells"].index(cell)]
    P = PR.params(ch_etaII=ETA2)
    lim = PR.fws_max(P)
    if part == "train":
        Wb = train_weather(site)
        W = scenario_weather(Wb, delta12, False)
        slope = 0.0  # raw training years: no slope is fitted or applied
    else:
        Wb, slope = test_weather(site)
        yrs = np.asarray(Wb["years"], float)
        d = np.tile(delta12[MONTH], (len(yrs), 1)) - (slope * (yrs - REF))[:, None]
        W = shift_hourly(Wb, d)
    ds = design_list(P, lim)
    R = np.zeros((len(ds), W["t_db"].shape[0]))
    for j, dsg in enumerate(ds):
        R[j] = req_and_opex(P, W, dsg, lim, want_opex=False)[0]
    tmp = out.parent / f"{site}__{model}.tmp.npz"
    np.savez_compressed(tmp, R=R, years=W["years"], delta12=delta12, slope=slope, designs=np.array([json.dumps(x) for x in ds]))
    os.replace(tmp, out)
    return site, model, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("test", "train", "shift", "bmon"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    (OUT / a.part).mkdir(parents=True, exist_ok=True)
    (OUT / a.part / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    _, models = fields()
    assert len(models) == 33
    if a.part in ("test", "train"):
        items = [(a.part, st, m) for st in r2_sites() for m in models if not (OUT / a.part / f"{st}__{m}.npz").exists()]
        fn = run_item
    elif a.part == "shift":
        g = shift_grid()
        items = []
        for st in r2_sites():
            for k in range(1, int(round(g[st]["grid_max_K"] * 10)) + 1):
                sig = round(0.1 * k, 1)
                if any(abs(sig - x) < 1e-9 for x in R2_GRID):
                    continue
                if not (OUT / "shift" / f"{st}__{sig:.1f}.npz").exists():
                    items.append(("shift", st, f"{sig:.1f}", [sig] * 12))
        # signed B-SCAL atoms below the grid (internal review stage2 R1, F1): rounded JJA deltas < 0.05 K other than 0
        s, _ = fields()
        for st in r2_sites():
            ci = s["cells"].index(s["site_cell"][st])
            for m in s["valid_models"]:
                sig = float(np.floor(float(np.mean(np.array(s["delta_2060"][m])[5:8, ci])) * 10 + 0.5) / 10)
                if sig < -1e-9 and not (OUT / "shift" / f"{st}__{sig:.1f}.npz").exists() and                         ("shift", st, f"{sig:.1f}", [sig] * 12) not in items:
                    items.append(("shift", st, f"{sig:.1f}", [sig] * 12))
        fn = run_field_item
    else:
        idx, uniq = bmon_fields()
        (OUT / "bmon_index.json").write_text(json.dumps({f"{s}|{f}|{q:g}": k for (s, f, q), k in idx.items()}), encoding="utf-8")
        items = [("bmon", st, key, fld) for (st, key), fld in sorted(uniq.items()) if not (OUT / "bmon" / f"{st}__{key}.npz").exists()]
        fn = run_field_item
    print(f"{len(items)} items pending ({a.part})", flush=True)
    with mp.Pool(a.workers, maxtasksperchild=40) as pool:
        for n, r in enumerate(pool.imap(fn, items), 1):
            print(n, r, flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
