"""DD-018 / PR-R2F: the unused ERA5 test block, 1960-01-01 .. 1989-12-31 hourly, at the 15 R2 sites' ERA5 points (the R2 weather
tables' coordinates), from the CDS time-series product (same variables and request form as DD-017). Tables in the mirror format
(t0 = 1960-01-01T00Z, native sp) in D:/_caches/era5/cds/tables_1960/<site>.npz; complete hours required. Inputs only.
Usage: python experiments/phase1/r2f_extract_1960.py [--threads 3]
"""
import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from cds_extract import DATASET, fetch, read_zip  # noqa: E402

BASE = pathlib.Path("D:/_caches/era5/cds")
RAW, TAB = BASE / "raw_1960", BASE / "tables_1960"
T0 = np.datetime64("1960-01-01T00", "h")
N = int((np.datetime64("1990-01-01T00", "h") - T0) / np.timedelta64(1, "h"))


def r2_points():
    r2 = sorted(p.stem for p in (ROOT / "experiments/phase0/out/r2/sites").glob("*.npz") if not p.stem.endswith(("partial", "tmp")))
    s = pd.read_csv(ROOT / "data/sites/phase0_sites.csv").set_index("site_id")
    return [{"site": x, "lat": round(float(s.loc[x, "era5_lat"]), 2), "lon": round(float(s.loc[x, "era5_lon"]), 2)} for x in r2]


def to_table(df, lat, lon, out):
    df = df.drop_duplicates("valid_time").sort_values("valid_time")
    assert np.allclose(df.latitude.unique(), [lat]) and np.allclose(((df.longitude.unique() - lon + 180) % 360) - 180, [0.0]), \
        (df.latitude.unique(), df.longitude.unique(), lat, lon)
    t = df.valid_time.values.astype("datetime64[h]")
    idx = ((t - T0) / np.timedelta64(1, "h")).astype(np.int64)
    ok = (idx >= 0) & (idx < N)
    arr = {k: np.full(N, np.nan) for k in ("t2m", "d2m", "sp")}
    for k in arr:
        arr[k][idx[ok]] = df[k].values[ok]
    missing = int(sum(np.isnan(v).sum() for v in arr.values()))
    assert missing == 0, f"{missing} missing values"
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t0=np.array(str(T0)), n_hours=np.array(N), lat=np.array(lat), lon=np.array(lon),
                        t2m_c100=np.round((arr["t2m"] - 273.15) * 100).astype(np.int16),
                        td_c100=np.round((arr["d2m"] - 273.15) * 100).astype(np.int16),
                        psfc_pa10=np.round(arr["sp"] / 10).astype(np.int16), source=np.array(DATASET))
    os.replace(tmp, out)


def run_point(row):
    try:
        import cdsapi
        out = TAB / f"{row['site']}.npz"
        if out.exists():
            return row["site"], "skip"
        client = cdsapi.Client(quiet=True, progress=False)
        t0 = time.time()
        raw = RAW / f"{row['site']}__1960-1989.zip"
        if not raw.exists():
            fetch(client, row["lat"], row["lon"], "1960-01-01", "1989-12-31", raw)
        to_table(read_zip(raw), row["lat"], row["lon"], out)
        return row["site"], f"done {time.time() - t0:.0f}s"
    except Exception as e:  # keep the other points going; a failed point is retried on the next run
        return row["site"], f"ERROR {e!r}"[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    for d in (RAW, TAB):
        d.mkdir(parents=True, exist_ok=True)
    pts = r2_points()
    print(f"{len(pts)} points", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        for n, r in enumerate(ex.map(run_point, pts), 1):
            print(n, r, flush=True)
    done = sorted(p.stem for p in TAB.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    (ROOT / "experiments/phase1/out/r2f").mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments/phase1/out/r2f/era5_1960_extract.json").write_text(
        json.dumps({"points": pts, "tables_done": done, "hours": N}, indent=1), encoding="utf-8")
    print(len(done), "tables", flush=True)


if __name__ == "__main__":
    main()
