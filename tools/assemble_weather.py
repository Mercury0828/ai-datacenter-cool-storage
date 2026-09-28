"""Assemble per-site hourly ERA5 series 1990-01-01T00Z .. 2025-12-31T23Z from the extracted .om point files.

Checks (printed and stored in data/weather/weather_qc.json):
  * chunk alignment: chunk_K covers hours [K*504, (K+1)*504) since 1970-01-01T00Z; the 2021-12-23..31 overlap
    between year_2021 and chunk_904 must agree exactly;
  * completeness: no NaN in the assembled window;
  * surface pressure p_sfc = p_msl * (1 - 0.0065 z / (T + 0.0065 z + 273.15))**5.257 with z = ERA5 orography.
Output per site: data/weather/<site_id>.npz with int16 arrays
  t2m_c100 (T2m, 0.01 degC), td_c100 (dew point, 0.01 degC), psfc_pa10 (surface pressure, 10 Pa),
  plus t0 = '1990-01-01T00:00Z' and n_hours. The .npz is small enough to version in git.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PTS = pathlib.Path("D:/_caches/era5/om_points")
EPOCH = np.datetime64("1970-01-01T00", "h")
T0 = np.datetime64("1990-01-01T00", "h")
T1 = np.datetime64("2026-01-01T00", "h")  # exclusive
VARS = ["temperature_2m", "dew_point_2m", "pressure_msl"]


def load(var, fname, ids):
    with np.load(PTS / var / f"{fname}.npz") as z:
        order = [z["site_ids"].tolist().index(s) for s in ids]
        return z["data"][order]


def series(var, ids):
    n = int((T1 - T0) / np.timedelta64(1, "h"))
    out = np.full((len(ids), n), np.nan, dtype=np.float32)
    for y in range(1990, 2022):
        a = load(var, f"year_{y}", ids)
        s = int((np.datetime64(f"{y}-01-01T00", "h") - T0) / np.timedelta64(1, "h"))
        out[:, s:s + a.shape[1]] = a
    overlap = {}
    for k in range(904, 974):
        a = load(var, f"chunk_{k}", ids)
        start = EPOCH + np.timedelta64(k * 504, "h")
        s = int((start - T0) / np.timedelta64(1, "h"))
        e = min(s + a.shape[1], n)
        if k == 904:  # overlap with year_2021 -> alignment check
            ov_end = int((np.datetime64("2022-01-01T00", "h") - T0) / np.timedelta64(1, "h"))
            prev = out[:, s:ov_end].copy()
            cur = a[:, :ov_end - s]
            overlap = {"hours": int(ov_end - s), "max_abs_diff": float(np.nanmax(np.abs(prev - cur)))}
        out[:, s:e] = a[:, :e - s]
    return out, overlap


def main():
    global PTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="data/sites/phase0_sites.csv")
    ap.add_argument("--pts", default="D:/_caches/era5/om_points")
    ap.add_argument("--outdir", default="data/weather")
    a = ap.parse_args()
    PTS = pathlib.Path(a.pts)
    sites = pd.read_csv(ROOT / a.sites)
    ids = sites.site_id.tolist()
    data, qc = {}, {"window": "1990-01-01T00Z..2025-12-31T23Z", "source": "ERA5 raw grid point via s3://openmeteo (DD-005)"}
    for v in VARS:
        data[v], ov = series(v, ids)
        qc[f"{v}_chunk904_overlap"] = ov
        qc[f"{v}_nan_count"] = int(np.isnan(data[v]).sum())
    z = sites.era5_orog_m.to_numpy()[:, None]
    t = data["temperature_2m"]
    psfc = data["pressure_msl"] * (1 - 0.0065 * z / (t + 0.0065 * z + 273.15)) ** 5.257
    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    summ = []
    years = (T0 + np.arange(t.shape[1]).astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970
    for i, sid in enumerate(ids):
        np.savez_compressed(outdir / f"{sid}.npz", t0=str(T0), n_hours=t.shape[1],
                            t2m_c100=np.round(t[i] * 100).astype(np.int16),
                            td_c100=np.round(data["dew_point_2m"][i] * 100).astype(np.int16),
                            psfc_pa10=np.round(psfc[i] / 10).astype(np.int16))
        ann = pd.Series(t[i]).groupby(years).max()
        summ.append({"site_id": sid, "Tmax_all": float(t[i].max()), "Tdmax_all": float(data["dew_point_2m"][i].max()),
                     "h_gt35_per_yr": float((t[i] > 35).sum() / 36), "h_gt38_per_yr": float((t[i] > 38).sum() / 36),
                     "h_gt40_per_yr": float((t[i] > 40).sum() / 36),
                     "ann_max_min": float(ann.min()), "ann_max_median": float(ann.median()),
                     "psfc_mean_hPa": float(psfc[i].mean() / 100)})
    s = pd.DataFrame(summ)
    s.to_csv(outdir / "weather_summary.csv", index=False)
    (outdir / "weather_qc.json").write_text(json.dumps(qc, indent=1), encoding="utf-8")
    print(json.dumps(qc, indent=1))
    print(s.round(1).to_string())


if __name__ == "__main__":
    main()
