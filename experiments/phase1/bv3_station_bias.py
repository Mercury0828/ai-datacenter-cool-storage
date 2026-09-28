"""BV-3 (pre-registered in ledger PR-BV, commit 8bb6703; details declared in DD-012): ERA5 vs NOAA ISD-lite station bias
and the declared additive hot-tail correction.

1. Station hourly dry-bulb and dew point (ISD-lite 1990-2025) vs ERA5 at the nearest land cell (`data/sites/bv3_stations.csv`),
   matched on UTC hours; stations below 80 % coverage are dropped (listed).
2. Metrics (ERA5 − station): percentiles 99 / 99.6 / 99.9 and annual maxima of dry-bulb, dew point, wet-bulb; hours per year
   above the station's 99th percentile and run lengths; night recovery (minimum dry-bulb of hot days).
3. Correction: Δ_v(q) = station − ERA5 quantile at the DD-012 knots, linear in q, zero below 0.95, constant above the last knot;
   dry-bulb and dew point by their own ranks; site-specific within 50 km, station-median elsewhere. Corrected hub tables are
   written to D:/_caches/era5/weather_hub_corrected/.
Outputs: experiments/phase1/out/bv3/bv3_station_bias.json (+ printed summary).
"""
import gzip
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import psychro as ps  # noqa: E402

ISD = pathlib.Path("D:/_caches/isd/lite")
WX_ST = pathlib.Path("D:/_caches/era5/weather_stations")
WX_HUB = pathlib.Path("D:/_caches/era5/weather_hub")
WX_OUT = pathlib.Path("D:/_caches/era5/weather_hub_corrected")
OUT = ROOT / "experiments/phase1/out/bv3"
T0 = np.datetime64("1990-01-01T00", "h")
N = 315576  # hours 1990-01-01T00 .. 2025-12-31T23
PCTS = [0.99, 0.996, 0.999]
KNOTS = [0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.9999]
VARS = ("dry_bulb", "dew_point", "wet_bulb")
YEARS = (T0 + np.arange(N).astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970


def load_isd(isd_id, elev):
    t, td, slp = (np.full(N, np.nan) for _ in range(3))
    for y in range(1990, 2026):
        f = ISD / f"{isd_id}-{y}.gz"
        if not f.exists():
            continue
        with gzip.open(f, "rt") as fh:
            a = np.loadtxt(fh, dtype=np.int64, usecols=range(7), ndmin=2)
        mon = (a[:, 0] - 1970).astype("datetime64[Y]").astype("datetime64[M]") + (a[:, 1] - 1).astype("timedelta64[M]")
        hrs = (mon.astype("datetime64[D]") + (a[:, 2] - 1).astype("timedelta64[D]")).astype("datetime64[h]") \
            + a[:, 3].astype("timedelta64[h]")
        idx = ((hrs - T0) / np.timedelta64(1, "h")).astype(np.int64)
        ok = (idx >= 0) & (idx < N)
        for arr, col, scale in ((t, 4, 0.1), (td, 5, 0.1), (slp, 6, 10.0)):
            good = ok & (a[:, col] != -9999)
            arr[idx[good]] = a[good, col] * scale
    p_std = 101325.0 * (1 - 2.25577e-5 * elev) ** 5.25588
    p = np.where(np.isfinite(slp) & np.isfinite(t),
                 slp * (1 - 0.0065 * elev / (np.nan_to_num(t) + 0.0065 * elev + 273.15)) ** 5.257, p_std)
    return t, td, p


def load_cell(path):
    z = np.load(path)
    assert str(z["t0"]).startswith("1990-01-01T00") and int(z["n_hours"]) == N
    return z["t2m_c100"] / 100.0, z["td_c100"] / 100.0, z["psfc_pa10"] * 10.0


def qdict(x, qs):
    v = np.quantile(x, qs)
    return {f"{q:g}": float(a) for q, a in zip(qs, v)}


def runs(mask):
    """Lengths of runs of True in a boolean array."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def local_days(x, valid, lon):
    """Daily max / min over local solar days (UTC + round(lon/15) h) with >= 20 valid hours."""
    off = int(round(lon / 15.0))
    xs = np.roll(np.where(valid, x, np.nan), off)  # local hour h = UTC hour h - off -> index shift
    vs = np.roll(valid, off)
    n = N // 24
    xd, vd = xs[: n * 24].reshape(n, 24), vs[: n * 24].reshape(n, 24)
    ok = vd.sum(axis=1) >= 20
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mx, mn = np.nanmax(xd, axis=1), np.nanmin(xd, axis=1)
    ok[0] = ok[-1] = False  # roll wrap-around
    return np.where(ok, mx, np.nan), np.where(ok, mn, np.nan)


def corr_fn(dk):
    qk = np.array(KNOTS)
    d = np.array([dk[f"{q:g}"] for q in KNOTS])
    return lambda q: np.where(q < qk[0], 0.0, np.interp(q, qk, d))


def rank_q(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = (np.arange(len(x)) + 0.5) / len(x)
    return r


def correct(t, td, cd):
    t_new = t + corr_fn(cd["dry_bulb"])(rank_q(t))
    td_new = np.minimum(td + corr_fn(cd["dew_point"])(rank_q(td)), t_new)
    return t_new, td_new


def wb(t, td, p):
    return ps.wet_bulb(t, np.minimum(td, t), p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WX_OUT.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(ROOT / "data/sites/bv3_stations.csv").set_index("site_id")
    res = {"declared": "PR-BV (ledger, commit 8bb6703) + DD-012", "stations": {}, "dropped": []}
    matched = {}
    for sid, s in st.iterrows():
        t_s, td_s, p_s = load_isd(s.isd_id, s.station_elev_m)
        t_e, td_e, p_e = load_cell(WX_ST / f"{sid}.npz")
        valid = np.isfinite(t_s) & np.isfinite(td_s)
        cov = float(valid.mean())
        if cov < 0.80:
            res["dropped"].append({"station": sid, "coverage": cov})
            print(sid, "DROPPED coverage", round(cov, 3), flush=True)
            continue
        vs = {"dry_bulb": t_s[valid], "dew_point": np.minimum(td_s[valid], t_s[valid])}
        ve = {"dry_bulb": t_e[valid], "dew_point": np.minimum(td_e[valid], t_e[valid])}
        vs["wet_bulb"] = wb(vs["dry_bulb"], vs["dew_point"], p_s[valid])
        ve["wet_bulb"] = wb(ve["dry_bulb"], ve["dew_point"], p_e[valid])
        matched[sid] = (valid, vs, ve, p_e)
        rec = {"coverage": cov, "matched_hours": int(valid.sum()), "isd_id": s.isd_id, "era5_cell": [s.era5_lat, s.era5_lon],
               "era5_dist_km": float(s.era5_dist_km), "elev_station_m": float(s.station_elev_m),
               "elev_era5_m": float(s.era5_orog_m)}
        for v in VARS:
            qs, qe = qdict(vs[v], PCTS), qdict(ve[v], PCTS)
            rec[v] = {"station": qs, "era5": qe, "era5_minus_station": {k: qe[k] - qs[k] for k in qs}}
            if v != "wet_bulb":
                ks, ke = qdict(vs[v], KNOTS), qdict(ve[v], KNOTS)
                rec[v]["delta_station_minus_era5_knots"] = {k: ks[k] - ke[k] for k in ks}
        # annual maxima: calendar years with >= 80 % matched hours
        yv = YEARS[valid]
        am = {v: [] for v in VARS}
        for y in range(1990, 2026):
            m = yv == y
            if m.sum() >= 0.8 * np.sum(YEARS == y):
                for v in VARS:
                    am[v].append((float(vs[v][m].max()), float(ve[v][m].max())))
        rec["annual_max"] = {v: {"n_years": len(am[v]), "mean_station": float(np.mean([a for a, _ in am[v]])),
                                 "mean_era5": float(np.mean([b for _, b in am[v]])),
                                 "mean_era5_minus_station": float(np.mean([b - a for a, b in am[v]]))} for v in VARS}
        # persistence: hours/yr above the station's p99 dry-bulb, run lengths (missing hours end runs)
        thr = rec["dry_bulb"]["station"]["0.99"]
        yrs = valid.sum() / 8766.0
        full_s, full_e = np.zeros(N, bool), np.zeros(N, bool)
        full_s[valid] = vs["dry_bulb"] > thr
        full_e[valid] = ve["dry_bulb"] > thr
        rs, re_ = runs(full_s), runs(full_e)
        rec["persistence"] = {"threshold_C": thr,
                              "hours_per_yr": {"station": float(full_s.sum() / yrs), "era5": float(full_e.sum() / yrs)},
                              "run_mean_h": {"station": float(rs.mean()), "era5": float(re_.mean()) if len(re_) else 0.0},
                              "run_p95_h": {"station": float(np.quantile(rs, 0.95)),
                                            "era5": float(np.quantile(re_, 0.95)) if len(re_) else 0.0},
                              "run_max_h": {"station": int(rs.max()), "era5": int(re_.max()) if len(re_) else 0}}
        # night recovery: min dry-bulb of hot local days (station daily max >= station p99 of daily maxima)
        ts_full = np.where(valid, t_s, np.nan)
        dmx_s, dmn_s = local_days(ts_full, valid, s.lon)
        dmx_e, dmn_e = local_days(np.where(valid, t_e, np.nan), valid, s.lon)
        ok = np.isfinite(dmx_s) & np.isfinite(dmx_e)
        hot = ok & (dmx_s >= np.quantile(dmx_s[ok], 0.99))
        rec["night_recovery"] = {"n_hot_days": int(hot.sum()), "hot_day_threshold_C": float(np.quantile(dmx_s[ok], 0.99)),
                                 "mean_min_station": float(dmn_s[hot].mean()), "mean_min_era5": float(dmn_e[hot].mean()),
                                 "era5_minus_station": float((dmn_e[hot] - dmn_s[hot]).mean()),
                                 "mean_max_era5_minus_station": float((dmx_e[hot] - dmx_s[hot]).mean())}
        res["stations"][sid] = rec
        print(sid, f"cov {cov:.3f}", " ".join(f"{v[:3]} p99.6 {rec[v]['era5_minus_station']['0.996']:+.2f}" for v in VARS),
              f"| annmax db {rec['annual_max']['dry_bulb']['mean_era5_minus_station']:+.2f}"
              f" wb {rec['annual_max']['wet_bulb']['mean_era5_minus_station']:+.2f}"
              f" | h>p99/yr st {rec['persistence']['hours_per_yr']['station']:.0f} era5 {rec['persistence']['hours_per_yr']['era5']:.0f}"
              f" | night {rec['night_recovery']['era5_minus_station']:+.2f}", flush=True)
    kept = list(res["stations"])
    med = {v: {f"{q:g}": float(np.median([res["stations"][k][v]["delta_station_minus_era5_knots"][f"{q:g}"] for k in kept]))
               for q in KNOTS} for v in ("dry_bulb", "dew_point")}
    res["median_correction"] = med

    # diagnostics: in-sample residual (own correction) and leave-one-out residual (median of the others)
    diag = {}
    for sid in kept:
        valid, vs, ve, p_e = matched[sid]
        t_e, td_e, _ = load_cell(WX_ST / f"{sid}.npz")
        out = {}
        others = [k for k in kept if k != sid]
        loo = {v: {f"{q:g}": float(np.median([res["stations"][k][v]["delta_station_minus_era5_knots"][f"{q:g}"] for k in others]))
                   for q in KNOTS} for v in ("dry_bulb", "dew_point")}
        for lab, cd in (("own", {v: res["stations"][sid][v]["delta_station_minus_era5_knots"] for v in ("dry_bulb", "dew_point")}),
                        ("loo_median", loo)):
            tc, tdc = correct(t_e, td_e, cd)
            vc = {"dry_bulb": tc[valid], "dew_point": tdc[valid]}
            vc["wet_bulb"] = wb(vc["dry_bulb"], vc["dew_point"], p_e[valid])
            out[lab] = {v: {k: a - b for (k, a), b in zip(qdict(vc[v], PCTS).items(), qdict(vs[v], PCTS).values())}
                        for v in VARS}
        diag[sid] = out
        print(sid, "residual after correction p99.6 (own | loo):",
              " ".join(f"{v[:3]} {out['own'][v]['0.996']:+.2f}|{out['loo_median'][v]['0.996']:+.2f}" for v in VARS), flush=True)
    res["diagnostics_residual_corrected_minus_station"] = diag

    # apply to the 80 hub cells
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv")
    applied = []
    for _, c in cells.iterrows():
        dist = {k: 111.2 * np.hypot(c.era5_lat - st.loc[k, "lat"], (c.era5_lon - st.loc[k, "lon"]) * np.cos(np.radians(c.era5_lat)))
                for k in kept}
        near = min(dist, key=dist.get)
        src = near if dist[near] <= 50.0 else "median"
        cd = ({v: res["stations"][near][v]["delta_station_minus_era5_knots"] for v in ("dry_bulb", "dew_point")}
              if src != "median" else med)
        z = np.load(WX_HUB / f"{c.site_id}.npz")
        t, td = z["t2m_c100"] / 100.0, z["td_c100"] / 100.0
        tc, tdc = correct(t, td, cd)
        tmp = WX_OUT / f"{c.site_id}.tmp.npz"
        np.savez_compressed(tmp, t0=z["t0"], n_hours=z["n_hours"], t2m_c100=np.round(tc * 100).astype(np.int16),
                            td_c100=np.round(tdc * 100).astype(np.int16), psfc_pa10=z["psfc_pa10"])
        tmp.replace(WX_OUT / f"{c.site_id}.npz")
        applied.append({"cell": c.site_id, "source": src, "nearest_station": near, "dist_km": round(float(dist[near]), 1),
                        "B_full_MW": float(c.B_full_MW), "dT_p99.6": float(np.quantile(tc, 0.996) - np.quantile(t, 0.996)),
                        "dTmax": float(tc.max() - t.max())})
    res["applied"] = applied
    (OUT / "bv3_station_bias.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("median Δ dry-bulb:", {k: round(v, 2) for k, v in med["dry_bulb"].items()})
    print("median Δ dew point:", {k: round(v, 2) for k, v in med["dew_point"].items()})
    for a in applied:
        if a["source"] != "median":
            print("site-specific:", a)
    print("cells site-specific:", sum(a["source"] != "median" for a in applied), "of", len(applied))


if __name__ == "__main__":
    main()
