"""DD-018 / PR-R2F: per-model monthly tas deltas, SSP2-4.5 2051-2070 minus historical 1995-2014, at the 15 R2 sites' hub cells
(nearest model grid point, as PR-CMIP), for the 33 PR-R2F models; window validation (20 values per calendar month in each window,
all finite). Inputs only: no cooling outcome is computed here.
Usage: python experiments/phase1/r2f_cmip_2060.py [--threads 4]
Output: experiments/phase1/out/r2f/deltas_2060_<model>.json and deltas_2060_all.json
"""
import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import gcsfs
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from cmip_extract import CAT, open_tas, pick  # noqa: E402

OUT = ROOT / "experiments/phase1/out/r2f"
BASE = (1995, 2014)
FUT = (2051, 2070)
FAMILIES = {
    "CAM": ["CESM2-WACCM", "CMCC-CM2-SR5", "CMCC-ESM2", "FIO-ESM-2-0", "NorESM2-LM", "NorESM2-MM", "TaiESM1", "CIESM", "BCC-CSM2-MR"],
    "ECHAM": ["AWI-CM-1-1-MR", "CAMS-CSM1-0", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "NESM3"],
    "IFS": ["EC-Earth3", "EC-Earth3-CC", "EC-Earth3-Veg", "EC-Earth3-Veg-LR"],
    "UM": ["ACCESS-CM2", "KACE-1-0-G"],
    "GFDL-AM": ["GFDL-CM4", "GFDL-ESM4", "KIOST-ESM"],
    "CAS-IAP": ["CAS-ESM2-0", "FGOALS-f3-L", "FGOALS-g3"],
    "INM": ["INM-CM4-8", "INM-CM5-0"],
    "LMDZ": ["IPSL-CM6A-LR"],
    "MIROC": ["MIROC6"],
    "MRI": ["MRI-ESM2-0"],
    "CanAM": ["CanESM5"],
    "GFS": ["IITM-ESM"],
}
MODELS = [m for ms in FAMILIES.values() for m in ms]


def r2_cells():
    r2 = sorted(p.stem for p in (ROOT / "experiments/phase0/out/r2/sites").glob("*.npz") if not p.stem.endswith(("partial", "tmp")))
    hub = pd.read_csv(ROOT / "data/sites/hub_sites.csv").set_index("site_id")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").set_index("site_id")
    ids = sorted(set(hub.loc[r2, "cell_id"]))
    return r2, hub.loc[r2, "cell_id"].to_dict(), cells.loc[ids].reset_index()


def clim_counted(tas, lat, lon, cells, y0, y1):
    """Monthly climatology at the nearest grid points plus the number of finite values per calendar month (window validation)."""
    import xarray as xr
    tas = tas.sortby("time")
    _, keep = np.unique(tas["time"].values, return_index=True)
    tas = tas.isel(time=np.sort(keep))
    yrs = np.array([t.year for t in tas["time"].values])
    sub = tas.isel(time=np.flatnonzero((yrs >= y0) & (yrs <= y1)))
    la = sub[lat].values
    lo = sub[lon].values % 360
    ii = [int(np.argmin(np.abs(la - c.era5_lat))) for c in cells.itertuples()]
    jj = [int(np.argmin(np.abs(((lo - (c.era5_lon % 360)) + 180) % 360 - 180))) for c in cells.itertuples()]
    pts = sub.isel({lat: xr.DataArray(ii, dims="cell"), lon: xr.DataArray(jj, dims="cell")}).load()
    months = np.array([t.month for t in pts["time"].values])
    vals = pts.values
    clim = np.stack([np.nanmean(vals[months == m], axis=0) for m in range(1, 13)])
    count = np.stack([np.isfinite(vals[months == m]).sum(axis=0) for m in range(1, 13)])  # (12, cells)
    return clim, count, [float(la[i]) for i in ii], [float(lo[j]) for j in jj]


def run_model(args):
    model, cat, cells, fs = args
    out = OUT / f"deltas_2060_{model}.json"
    if out.exists():
        return model, "skip"
    t0 = time.time()
    try:
        stores = {e: pick(cat, model, e) for e in ("historical", "ssp245")}
        th, la, lo = open_tas(fs, stores["historical"])
        base, nb, glat, glon = clim_counted(th, la, lo, cells, *BASE)
        tf, la2, lo2 = open_tas(fs, stores["ssp245"])
        fut, nf, _, _ = clim_counted(tf, la2, lo2, cells, *FUT)
        d = fut - base
        valid = bool((nb == BASE[1] - BASE[0] + 1).all() and (nf == FUT[1] - FUT[0] + 1).all() and np.isfinite(d).all())
        rec = {"model": model, "stores": stores, "cells": cells.site_id.tolist(), "grid_lat": glat, "grid_lon": glon,
               "delta_2060": d.tolist(), "count_base": nb.tolist(), "count_fut": nf.tolist(), "valid": valid,
               "elapsed_s": round(time.time() - t0, 1)}
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        tmp.replace(out)
        return model, f"done valid={valid}"
    except Exception as ex:  # record and continue; failed models are listed in the summary
        (OUT / f"failed_2060_{model}.txt").write_text(repr(ex), encoding="utf-8")
        return model, f"failed: {ex!r}"[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert len(MODELS) == 33 and len(set(MODELS)) == 33
    cat = pd.read_csv(CAT)
    cat = cat[(cat.table_id == "Amon") & (cat.variable_id == "tas") & (cat.member_id == "r1i1p1f1")
              & cat.experiment_id.isin(("historical", "ssp245"))]
    r2, site_cell, cells = r2_cells()
    fs = gcsfs.GCSFileSystem(token="anon")
    print(f"{len(MODELS)} models, {len(cells)} cells for {len(r2)} R2 sites", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        for n, r in enumerate(ex.map(run_model, [(m, cat, cells, fs) for m in MODELS]), 1):
            print(n, r, flush=True)
    recs = {m: json.loads((OUT / f"deltas_2060_{m}.json").read_text(encoding="utf-8")) for m in MODELS
            if (OUT / f"deltas_2060_{m}.json").exists()}
    summ = {"window": {"base": BASE, "future": FUT}, "families": FAMILIES, "site_cell": site_cell, "cells": cells.site_id.tolist(),
            "valid_models": [m for m in MODELS if m in recs and recs[m]["valid"]],
            "invalid_or_failed": [m for m in MODELS if m not in recs or not recs[m]["valid"]],
            "delta_2060": {m: recs[m]["delta_2060"] for m in MODELS if m in recs and recs[m]["valid"]}}
    (OUT / "deltas_2060_all.json").write_text(json.dumps(summ), encoding="utf-8")
    print("valid", len(summ["valid_models"]), "invalid/failed", summ["invalid_or_failed"], flush=True)


if __name__ == "__main__":
    main()
