"""M2 provenance spot-check (DD-015): the ERA5 tables used in the analysis (Open-Meteo `copernicus_era5` mirror) vs ARCO-ERA5
(Google Research's analysis-ready Zarr copy of ECMWF ERA5), at all 87 grid points used (80 hub cells + 7 station cells), on the
declared sample hours. Reads one global field per variable and hour (anonymous GCS), keeps only the 87 point values.
Output: experiments/phase1/out/m2/arco_spotcheck.json (+ per-hour point values in arco_points.npz). Resumable per hour batch.
Usage: python experiments/phase1/m2_arco_spotcheck.py [--threads 8]
"""
import argparse
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import gcsfs
import numpy as np
import pandas as pd
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/phase1/out/m2"
STORE = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
T0 = np.datetime64("1990-01-01T00", "h")
N = 315576
HOT_CELLS = ["cell_490_321", "cell_487_293", "cell_494_271", "cell_457_938", "cell_500_360", "cell_531_368"]
VARS = {"2m_temperature": "t2m_c100", "2m_dewpoint_temperature": "td_c100", "surface_pressure": "psfc_pa10"}


def points():
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv")
    st = pd.read_csv(ROOT / "data/sites/bv3_stations.csv")
    pts = [(c.site_id, c.era5_lat, c.era5_lon, pathlib.Path("D:/_caches/era5/weather_hub") / f"{c.site_id}.npz") for c in cells.itertuples()]
    pts += [(f"station_{s.site_id}", s.era5_lat, s.era5_lon, pathlib.Path("D:/_caches/era5/weather_stations") / f"{s.site_id}.npz")
            for s in st.itertuples()]
    return pts


def sample_hours(tables):
    hrs = set()
    for c in HOT_CELLS:
        t = tables[c]["t2m_c100"]
        i = int(np.argmax(t))
        hrs.update(range(max(0, i - 84), min(N, i + 84)))
    rng = np.random.default_rng(20260925)
    years = (T0 + np.arange(N).astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970
    for y in range(1990, 2025):
        idx = np.flatnonzero(years == y)
        hrs.update(int(x) for x in rng.choice(idx, 10, replace=False))
    return sorted(hrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pts = points()
    tables = {p[0]: np.load(p[3]) for p in pts}
    hours = sample_hours(tables)
    ilat = np.array([int(round((90.0 - p[1]) / 0.25)) for p in pts])
    ilon = np.array([int(round((p[2] % 360.0) / 0.25)) for p in pts])
    fs = gcsfs.GCSFileSystem(token="anon")
    g = zarr.open_group(fs.get_mapper(STORE), mode="r")
    off = int((T0 - np.datetime64("1900-01-01T00", "h")) / np.timedelta64(1, "h"))
    assert int(g["time"][off]) == off, "ARCO time axis is not hours since 1900-01-01 at the expected index"
    part = OUT / "arco_points_partial.npz"
    done = dict(np.load(part)) if part.exists() else {}
    vals = {v: done.get(v, np.full((len(hours), len(pts)), np.nan)) for v in VARS}
    todo = [(k, h) for k, h in enumerate(hours) if not all(np.isfinite(vals[v][k]).all() for v in VARS)]
    print(f"{len(pts)} points, {len(hours)} sample hours, {len(todo)} to read", flush=True)

    def read(job):
        k, h = job
        out = {}
        for v in VARS:
            f = g[v][off + h]
            out[v] = f[ilat, ilon]
        return k, out

    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for n, (k, out) in enumerate(ex.map(read, todo), 1):
            for v in VARS:
                vals[v][k] = out[v]
            if n % 100 == 0 or n == len(todo):
                np.savez(OUT / "arco_points_partial.tmp.npz", **vals)
                (OUT / "arco_points_partial.tmp.npz").replace(part)
                print(f"[{time.time() - t0:6.0f}s] {n}/{len(todo)} hours", flush=True)
    # compare with the tables
    conv = {"2m_temperature": (lambda x: x / 100.0, lambda k: k - 273.15), "2m_dewpoint_temperature": (lambda x: x / 100.0, lambda k: k - 273.15),
            "surface_pressure": (lambda x: x * 10.0, lambda p: p)}
    res = {"store": STORE, "n_points": len(pts), "n_hours": len(hours), "hours_utc_first_last": [str(T0 + hours[0]), str(T0 + hours[-1])],
           "variables": {}}
    hrs = np.array(hours)
    for v, col in VARS.items():
        tab = np.array([conv[v][0](tables[p[0]][col][hrs].astype(float)) for p in pts]).T
        arc = conv[v][1](vals[v].astype(float))
        d = tab - arc
        ad = np.abs(d)
        worst = np.unravel_index(np.nanargmax(ad), ad.shape)
        res["variables"][v] = {"max_abs": float(np.nanmax(ad)), "mean": float(np.nanmean(d)), "p999_abs": float(np.nanquantile(ad, 0.999)),
                               "worst": {"point": pts[worst[1]][0], "hour_utc": str(T0 + hrs[worst[0]]),
                                         "table": float(tab[worst]), "arco": float(arc[worst])},
                               "per_point_max_abs": {p[0]: float(np.nanmax(ad[:, j])) for j, p in enumerate(pts)}}
    tol = {"2m_temperature": 0.05, "2m_dewpoint_temperature": 0.05, "surface_pressure": 15.0}
    res["pass"] = {v: res["variables"][v]["max_abs"] <= tol[v] for v in VARS}
    res["PASS"] = all(res["pass"].values())
    (OUT / "arco_spotcheck.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "arco_points.npz", hours=hrs, **{v: vals[v] for v in VARS})
    for v, r in res["variables"].items():
        print(f"{v:26s} max|d| {r['max_abs']:.4f}  mean d {r['mean']:+.5f}  p99.9 {r['p999_abs']:.4f}  worst {r['worst']}")
    print("PASS" if res["PASS"] else "NOT PASS", res["pass"])


if __name__ == "__main__":
    main()
