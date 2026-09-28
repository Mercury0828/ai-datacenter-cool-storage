"""PR-CMIP step 1: monthly-climatology tas deltas (2031–2050 minus 1995–2014) at the 80 hub cells, per CMIP6 model and SSP, from the
Pangeo CMIP6 cloud (gs://cmip6, anonymous). Nearest model grid cell to each ERA5 cell. Only point values are kept.
Usage: python experiments/phase1/cmip_extract.py [--threads 4]
Output: experiments/phase1/out/cmip/deltas_<source_id>.json (per model; resumable) and deltas_all.json
"""
import argparse
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/phase1/out/cmip"
CAT = pathlib.Path("D:/_caches/cmip6/pangeo_cmip6.csv")
EXPS = ("historical", "ssp245", "ssp585")
BASE = ("1995-01-01", "2014-12-31")
FUT = ("2031-01-01", "2050-12-31")


def pick(cat, model, exp):
    q = cat[(cat.source_id == model) & (cat.experiment_id == exp)]
    if q.empty:
        return None
    q = q.sort_values(["grid_label", "version"], key=lambda s: s.map(lambda x: (0 if x == "gn" else 1)) if s.name == "grid_label" else s,
                      ascending=[True, False])
    return q.iloc[0].zstore


def open_tas(fs, store):
    ds = xr.open_zarr(fs.get_mapper(store.replace("gs://", "")), consolidated=True, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    tas = ds["tas"]
    lat = "lat" if "lat" in tas.dims else [d for d in tas.dims if "lat" in d][0]
    lon = "lon" if "lon" in tas.dims else [d for d in tas.dims if "lon" in d][0]
    return tas, lat, lon


def clim(tas, lat, lon, cells, period):
    # some stores carry non-monotonic or duplicated time stamps and 360-day calendars: sort, drop duplicates, select by year
    tas = tas.sortby("time")
    _, keep = np.unique(tas["time"].values, return_index=True)
    tas = tas.isel(time=np.sort(keep))
    y0, y1 = int(period[0][:4]), int(period[1][:4])
    yrs = np.array([t.year for t in tas["time"].values])
    sub = tas.isel(time=np.flatnonzero((yrs >= y0) & (yrs <= y1)))
    la = sub[lat].values
    lo = sub[lon].values % 360
    ii = [int(np.argmin(np.abs(la - c.era5_lat))) for c in cells.itertuples()]
    jj = [int(np.argmin(np.abs(((lo - (c.era5_lon % 360)) + 180) % 360 - 180))) for c in cells.itertuples()]
    pts = sub.isel({lat: xr.DataArray(ii, dims="cell"), lon: xr.DataArray(jj, dims="cell")}).load()
    months = np.array([t.month for t in pts["time"].values])
    out = np.stack([pts.values[months == m].mean(axis=0) for m in range(1, 13)])  # (12, cells)
    return out, [float(la[i]) for i in ii], [float(lo[j]) for j in jj]


def run_model(args):
    model, cat, cells, fs = args
    out = OUT / f"deltas_{model}.json"
    if out.exists():
        return model, "skip"
    t0 = time.time()
    try:
        stores = {e: pick(cat, model, e) for e in EXPS}
        th, la, lo = open_tas(fs, stores["historical"])
        base, glat, glon = clim(th, la, lo, cells, BASE)
        rec = {"model": model, "stores": stores, "grid_lat": glat, "grid_lon": glon}
        for e in ("ssp245", "ssp585"):
            tf, la2, lo2 = open_tas(fs, stores[e])
            fut, _, _ = clim(tf, la2, lo2, cells, FUT)
            rec[e] = (fut - base).tolist()  # (12, cells) K
        rec["elapsed_s"] = round(time.time() - t0, 1)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        tmp.replace(out)
        return model, "done"
    except Exception as ex:  # record and continue; failed models are listed in the summary
        (OUT / f"failed_{model}.txt").write_text(repr(ex), encoding="utf-8")
        return model, f"failed: {ex!r}"[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(CAT)
    cat = cat[(cat.table_id == "Amon") & (cat.variable_id == "tas") & (cat.member_id == "r1i1p1f1") & cat.experiment_id.isin(EXPS)]
    have = cat.groupby("source_id").experiment_id.agg(lambda s: set(s))
    models = sorted(m for m, s in have.items() if set(EXPS) <= s)
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv")
    fs = gcsfs.GCSFileSystem(token="anon")
    print(f"{len(models)} models", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        for n, r in enumerate(ex.map(run_model, [(m, cat, cells, fs) for m in models]), 1):
            print(n, r, flush=True)
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("deltas_*.json")) if f.name != "deltas_all.json"]
    arr = {e: np.array([r[e] for r in recs]) for e in ("ssp245", "ssp585")}  # (models, 12, cells)
    summ = {"models": [r["model"] for r in recs], "failed": [f.stem.replace("failed_", "") for f in OUT.glob("failed_*.txt")],
            "cells": cells.site_id.tolist()}
    for e, x in arr.items():
        summ[e] = {q: np.nanquantile(x, p, axis=0).tolist() for q, p in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))}
        summ[e]["n_models_finite"] = np.isfinite(x).sum(axis=0).min().item()
        summ[e]["JJA_median_capacity_weighted"] = float(np.average(np.nanmedian(x, axis=0)[5:8].mean(axis=0), weights=cells.B_full_MW))
    (OUT / "deltas_all.json").write_text(json.dumps(summ), encoding="utf-8")
    print({e: round(summ[e]["JJA_median_capacity_weighted"], 2) for e in arr}, "models", len(recs), "failed", summ["failed"])


if __name__ == "__main__":
    main()
