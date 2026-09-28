"""PR-AS2 AS2-1: ERA5 (nearest land cell, the tables' source) vs NOAA USCRN rural reference stations near the boundary cluster.
Hourly T_CALC (5-min mean ending at the UTC hour) matched to ERA5's instantaneous 2 m temperature at that UTC hour. Metrics
(ERA5 − station): dry-bulb 99 / 99.6 / 99.9th percentiles, annual maxima, mean daily minimum on hot local days (station daily max ≥
its 99th percentile of daily maxima; days with ≥ 20 matched hours) and on all JJA days. Raw differences decide the registered rule;
a lapse-adjusted version (6.5 K/km to the station elevation) is reported as a diagnostic.
Output: experiments/phase1/out/as2/uscrn_nights.json
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
USCRN = pathlib.Path("D:/_caches/uscrn/hourly02")
WX = pathlib.Path("D:/_caches/era5/weather_uscrn")
OUT = ROOT / "experiments/phase1/out/as2"
T0 = np.datetime64("1990-01-01T00", "h")
N = 315576


def load_uscrn(name):
    t = np.full(N, np.nan)
    for f in sorted(USCRN.glob(f"CRNH0203-*-{name}.txt")):
        a = np.loadtxt(f, usecols=(1, 2, 8), dtype=float, ndmin=2)
        d = a[:, 0].astype(np.int64)
        hh = (a[:, 1].astype(np.int64) // 100)
        dates = (d // 10000 - 1970).astype("datetime64[Y]").astype("datetime64[M]") + ((d // 100) % 100 - 1).astype("timedelta64[M]")
        hrs = (dates.astype("datetime64[D]") + (d % 100 - 1).astype("timedelta64[D]")).astype("datetime64[h]") + hh.astype("timedelta64[h]")
        idx = ((hrs - T0) / np.timedelta64(1, "h")).astype(np.int64)
        ok = (idx >= 0) & (idx < N) & (a[:, 2] > -9000)
        t[idx[ok]] = a[ok, 2]
    return t


def local_days(x, valid, lon):
    off = int(round(lon / 15.0))
    xs = np.roll(np.where(valid, x, np.nan), off)
    vs = np.roll(valid, off)
    n = N // 24
    xd, vd = xs[: n * 24].reshape(n, 24), vs[: n * 24].reshape(n, 24)
    ok = vd.sum(axis=1) >= 20
    ok[0] = ok[-1] = False
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mx, mn = np.nanmax(xd, axis=1), np.nanmin(xd, axis=1)
    day0 = (T0 + np.timedelta64(-off, "h")).astype("datetime64[D]")
    months = ((day0 + np.arange(n).astype("timedelta64[D]")).astype("datetime64[M]").astype(int) % 12) + 1
    return np.where(ok, mx, np.nan), np.where(ok, mn, np.nan), months


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(ROOT / "data/sites/as2_uscrn_stations.csv")
    years = (T0 + np.arange(N).astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970
    res = {"stations": {}}
    for s in st.itertuples():
        ts = load_uscrn(s.site_id)
        z = np.load(WX / f"{s.site_id}.npz")
        te = z["t2m_c100"] / 100.0
        valid = np.isfinite(ts)
        first = int(np.flatnonzero(valid)[0]) if valid.any() else 0
        cov = float(valid[first:].mean())
        lapse = -0.0065 * (s.station_elev_m - s.era5_orog_m)  # K added to ERA5 to move it to the station elevation
        rec = {"coverage_since_first_obs": cov, "matched_hours": int(valid.sum()), "first_obs": str(T0 + first),
               "elev_station_m": float(s.station_elev_m), "elev_era5_m": float(s.era5_orog_m), "lapse_adj_K": lapse,
               "era5_dist_km": float(s.era5_dist_km)}
        if cov < 0.8:
            rec["dropped"] = True
            res["stations"][s.site_id] = rec
            continue
        a, b = ts[valid], te[valid]
        q = [0.99, 0.996, 0.999]
        rec["dry_bulb_q_era5_minus_station"] = {f"{x:g}": float(np.quantile(b, x) - np.quantile(a, x)) for x in q}
        am = []
        for y in range(int(years[first]), 2026):
            m = (years == y) & valid
            if m.sum() >= 0.8 * np.sum(years == y):
                am.append(float(te[m].max() - ts[m].max()))
        rec["annual_max_mean_era5_minus_station"] = float(np.mean(am)) if am else None
        rec["annual_max_years"] = len(am)
        dmx_s, dmn_s, mon = local_days(ts, valid, s.lon)
        dmx_e, dmn_e, _ = local_days(te, valid, s.lon)
        ok = np.isfinite(dmx_s) & np.isfinite(dmx_e)
        hot = ok & (dmx_s >= np.quantile(dmx_s[ok], 0.99))
        jja = ok & np.isin(mon, (6, 7, 8))
        rec["hot_days"] = int(hot.sum())
        rec["night_min_hot_days_era5_minus_station"] = float(np.mean(dmn_e[hot] - dmn_s[hot]))
        rec["night_min_hot_days_lapse_adjusted"] = rec["night_min_hot_days_era5_minus_station"] + lapse
        rec["day_max_hot_days_era5_minus_station"] = float(np.mean(dmx_e[hot] - dmx_s[hot]))
        rec["night_min_jja_era5_minus_station"] = float(np.mean(dmn_e[jja] - dmn_s[jja]))
        rec["night_min_jja_lapse_adjusted"] = rec["night_min_jja_era5_minus_station"] + lapse
        res["stations"][s.site_id] = rec
        print(f"{s.site_id:20s} cov {cov:.3f} p99.6 {rec['dry_bulb_q_era5_minus_station']['0.996']:+.2f} annmax "
              f"{rec['annual_max_mean_era5_minus_station']:+.2f} | hot-day night min {rec['night_min_hot_days_era5_minus_station']:+.2f} "
              f"(lapse-adj {rec['night_min_hot_days_lapse_adjusted']:+.2f}) | JJA night {rec['night_min_jja_era5_minus_station']:+.2f} "
              f"(adj {rec['night_min_jja_lapse_adjusted']:+.2f}) | hot days {rec['hot_days']}", flush=True)
    kept = [v for v in res["stations"].values() if not v.get("dropped")]
    med = float(np.median([v["night_min_hot_days_era5_minus_station"] for v in kept]))
    med_adj = float(np.median([v["night_min_hot_days_lapse_adjusted"] for v in kept]))
    res["median_hot_day_night_min_era5_minus_station"] = med
    res["median_hot_day_night_min_lapse_adjusted"] = med_adj
    res["rule"] = ("closed: no ERA5 night bias at rural reference sites (|median| <= 0.5 K)" if abs(med) <= 0.5 else
                   ("apply the median rural night bias to the cluster cells" if med < -0.5 else "ERA5 nights warmer than rural stations (> +0.5 K)"))
    (OUT / "uscrn_nights.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("median hot-day night-min ERA5 - station:", round(med, 2), "K (lapse-adjusted", round(med_adj, 2), ") ->", res["rule"])


if __name__ == "__main__":
    main()
