"""DD-017: ERA5 hourly time series from the Copernicus CDS (`reanalysis-era5-single-levels-timeseries`) for every unique ERA5 point
used by any analysis (106), 1990-2025, 2m temperature / 2m dew point / surface pressure. Raw zips in D:/_caches/era5/cds/raw/, tables
in the mirror format (t0, n_hours, t2m_c100, td_c100, psfc_pa10 = native sp) in D:/_caches/era5/cds/tables/<point>.npz, and a point
map experiments/phase1/out/m2/cds_points.csv (point -> every site / cell name). Resumable; requests split by decade if needed.
Usage: python experiments/phase1/cds_extract.py [--threads 3]
"""
import argparse
import io
import json
import os
import pathlib
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = pathlib.Path("D:/_caches/era5/cds")
RAW, TAB = BASE / "raw", BASE / "tables"
DATASET = "reanalysis-era5-single-levels-timeseries"
VARS = ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure"]
T0 = np.datetime64("1990-01-01T00", "h")
N = int((np.datetime64("2026-01-01T00", "h") - T0) / np.timedelta64(1, "h"))  # 315,576
TABLES = (("hub_cells.csv", "site_id"), ("bv3_stations.csv", "site_id"), ("as2_uscrn_stations.csv", "site_id"),
          ("md_hubs.csv", "site_id"), ("nonepoch_hot_sites.csv", "site_id"), ("phase0_sites.csv", "site_id"))


def point_key(lat, lon):
    return f"p{lat:+07.2f}_{lon:+08.2f}".replace("+", "n").replace("-", "s")


def points():
    rows = {}
    for tab, col in TABLES:
        s = pd.read_csv(ROOT / "data/sites" / tab)
        for r in s.itertuples():
            lat, lon = round(float(r.era5_lat), 2), round(float(r.era5_lon), 2)
            k = point_key(lat, lon)
            rows.setdefault(k, {"point": k, "lat": lat, "lon": lon, "names": []})["names"].append(f"{tab.split('.')[0]}:{getattr(r, col)}")
    df = pd.DataFrame(rows.values())
    df["names"] = df.names.map(lambda x: ";".join(x))
    return df


def fetch(client, lat, lon, start, end, out):
    req = {"variable": VARS, "location": {"longitude": lon, "latitude": lat}, "date": [f"{start}/{end}"], "data_format": "csv"}
    tmp = out.with_suffix(".part")
    client.retrieve(DATASET, req, str(tmp))
    os.replace(tmp, out)


def read_zip(path):
    z = zipfile.ZipFile(path)
    frames = [pd.read_csv(io.BytesIO(z.read(n)), parse_dates=["valid_time"]) for n in z.namelist() if n.endswith(".csv")]
    return pd.concat(frames, ignore_index=True)


def to_table(df, lat, lon, out):
    df = df.drop_duplicates("valid_time").sort_values("valid_time")
    assert np.allclose(df.latitude.unique(), [lat]) and np.allclose(((df.longitude.unique() - lon + 180) % 360) - 180, [0.0]), \
        (df.latitude.unique(), df.longitude.unique(), lat, lon)
    t = df.valid_time.values.astype("datetime64[h]")
    idx = ((t - T0) / np.timedelta64(1, "h")).astype(np.int64)
    ok = (idx >= 0) & (idx < N)
    t2m = np.full(N, np.nan); d2m = np.full(N, np.nan); sp = np.full(N, np.nan)
    t2m[idx[ok]] = df.t2m.values[ok] - 273.15
    d2m[idx[ok]] = df.d2m.values[ok] - 273.15
    sp[idx[ok]] = df.sp.values[ok]
    missing = int(np.isnan(t2m).sum() + np.isnan(d2m).sum() + np.isnan(sp).sum())
    assert missing == 0, f"{missing} missing values"
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t0=np.array(str(T0)), n_hours=np.array(N), lat=np.array(lat), lon=np.array(lon),
                        t2m_c100=np.round(t2m * 100).astype(np.int16), td_c100=np.round(d2m * 100).astype(np.int16),
                        psfc_pa10=np.round(sp / 10).astype(np.int16), source=np.array(DATASET))
    os.replace(tmp, out)


def run_point(args):
    try:
        return _run_point(args)
    except Exception as e:  # keep the other points going; a failed point is retried on the next run
        return args[0]["point"], f"ERROR {e!r}"[:300]


def _run_point(args):
    import cdsapi
    row, = args
    out = TAB / f"{row['point']}.npz"
    if out.exists():
        return row["point"], "skip"
    client = cdsapi.Client(quiet=True, progress=False)
    t0 = time.time()
    whole = RAW / f"{row['point']}__1990-2025.zip"
    try:
        if not whole.exists():
            fetch(client, row["lat"], row["lon"], "1990-01-01", "2025-12-31", whole)
        df = read_zip(whole)
    except Exception as e:  # fall back to decades if the single request is refused
        parts = []
        for a, b in (("1990-01-01", "1999-12-31"), ("2000-01-01", "2009-12-31"), ("2010-01-01", "2019-12-31"), ("2020-01-01", "2025-12-31")):
            p = RAW / f"{row['point']}__{a[:4]}-{b[:4]}.zip"
            if not p.exists():
                fetch(client, row["lat"], row["lon"], a, b, p)
            parts.append(read_zip(p))
        df = pd.concat(parts, ignore_index=True)
        print(row["point"], "decade split after", repr(e)[:120], flush=True)
    to_table(df, row["lat"], row["lon"], out)
    return row["point"], f"done {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    for d in (RAW, TAB, ROOT / "experiments/phase1/out/m2"):
        d.mkdir(parents=True, exist_ok=True)
    pts = points()
    pts.to_csv(ROOT / "experiments/phase1/out/m2/cds_points.csv", index=False)
    todo = [r for r in pts.to_dict(orient="records") if not (TAB / f"{r['point']}.npz").exists()]
    print(f"{len(pts)} points, {len(todo)} pending", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        for n, r in enumerate(ex.map(run_point, [(r,) for r in todo]), 1):
            print(n, r, flush=True)
    (ROOT / "experiments/phase1/out/m2/cds_extract_done.json").write_text(json.dumps({"points": len(pts)}), encoding="utf-8")


if __name__ == "__main__":
    main()
