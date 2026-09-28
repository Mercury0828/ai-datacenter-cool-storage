"""Extract hourly ERA5 point series for the Phase-0 sites from the Open-Meteo open-data mirror (DD-005).

Reads raw ERA5 grid-point values (no elevation downscaling) from
  s3://openmeteo/data/copernicus_era5/<var>/year_YYYY.om   (1990-2021, one file per year)
  s3://openmeteo/data/copernicus_era5/<var>/chunk_K.om     (K = 904..973: 504-hour chunks, 2021-12-23 .. 2026-01-01)
Every file is one task; each task reads all sites' land grid points and writes one .npz atomically under
D:/_caches/era5/om_points/<var>/. A task whose output already holds every requested site is skipped, so the
run resumes after a kill (guide §7 checkpoint-and-resume: manifest + atomic per-item writes + skip-done).

Usage:  python tools/fetch_era5_om.py [--workers 8] [--fresh]
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import threading
import time

import numpy as np
import pandas as pd
import s3fs
from omfiles import OmFileReader

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path("D:/_caches/era5/om_points")
VARS = ["temperature_2m", "dew_point_2m", "pressure_msl"]
YEARS = range(1990, 2022)
CHUNKS = range(904, 974)
_local = threading.local()


def fs():
    if not hasattr(_local, "fs"):
        _local.fs = s3fs.S3FileSystem(anon=True)
    return _local.fs


def tasks():
    for v in VARS:
        for y in YEARS:
            yield v, f"year_{y}"
        for k in CHUNKS:
            yield v, f"chunk_{k}"


def out_path(var, fname):
    return OUT / var / f"{fname}.npz"


def done(var, fname, site_ids):
    p = out_path(var, fname)
    if not p.exists():
        return False
    try:
        with np.load(p) as z:
            return set(site_ids) <= set(z["site_ids"].tolist())
    except Exception:
        return False


def run_task(var, fname, sites):
    key = f"openmeteo/data/copernicus_era5/{var}/{fname}.om"
    for attempt in range(4):
        try:
            with fs().open(key, "rb", block_size=2**20, cache_type="blockcache") as f, OmFileReader(f) as r:
                data = np.stack([np.asarray(r[int(i), int(j), :], dtype=np.float32)
                                 for i, j in zip(sites.era5_ilat, sites.era5_ilon)])
            p = out_path(var, fname)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, data=data, site_ids=np.array(sites.site_id.tolist()),
                                ilat=sites.era5_ilat.to_numpy(), ilon=sites.era5_ilon.to_numpy())
            os.replace(tmp, p)  # atomic
            return var, fname, data.shape, None
        except Exception as e:  # network hiccups: back off and retry
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return var, fname, None, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--sites", default="data/sites/phase0_sites.csv")
    ap.add_argument("--out", default="D:/_caches/era5/om_points")
    a = ap.parse_args()
    global OUT
    OUT = pathlib.Path(a.out)
    sites = pd.read_csv(ROOT / a.sites)
    ids = sites.site_id.tolist()
    todo = [t for t in tasks() if a.fresh or not done(*t, ids)]
    manifest = OUT / "manifest.json"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} tasks pending of {len(list(tasks()))}", flush=True)
    failed, t0 = [], time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_task, v, f, sites) for v, f in todo]
        for n, fu in enumerate(cf.as_completed(futs), 1):
            v, f, shape, err = fu.result()
            if err:
                failed.append((v, f, err))
            if n % 20 == 0 or err:
                print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s {v}/{f} {shape or err}", flush=True)
    state = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "sites": ids,
             "done": [f"{v}/{f}" for v, f in tasks() if done(v, f, ids)], "failed": failed}
    tmp = manifest.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, manifest)
    print(f"finished in {time.time() - t0:.0f}s; done {len(state['done'])}; failed {len(failed)}", flush=True)


if __name__ == "__main__":
    main()
