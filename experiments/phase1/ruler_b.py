"""PR-RULER-b (ledger 2026-09-26, declared after internal review climate-ruler R1, before any run): practitioner inputs for the ruler.
(A) ASHRAE-convention coordinates: x_A = x; y_A = hottest-month mean daily range (calendar month with the highest mean dry-bulb over
    the record; mean of daily max - min on local days of that month in every year). Same logistic form, frame points, LOCO and bar as
    PR-RULER; external check on the 15 disjoint cells.
(B) Station inputs at 15 stations (8 USCRN, 7 BV-3 ISD): station hourly dry-bulb, gaps <= 6 h filled linearly, a year kept only if
    no longer gap remains; standard-atmosphere pressure at the station elevation. On the kept years, for the station series and the
    co-located ERA5 cell: coordinates (x, y, y_A), h*(k_max) with m = floor(n/20), and the ruler bands (PR-RULER model; model A).
Writes experiments/phase1/out/ruler_b/{points_A.csv, external_A.csv, stations.csv, ruler_b_summary.json}.
Usage: python experiments/phase1/ruler_b.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from as2_uscrn_nights import N, T0, load_uscrn  # noqa: E402
from bv1_min_storage import need_spare  # noqa: E402
from bv3_station_bias import load_isd  # noqa: E402
from cmip_scenarios import MONTH, SCEN  # noqa: E402
from model import design as D  # noqa: E402
from model import params as PR  # noqa: E402
from ruler import (TOL, WX, band, climate_coords, fit, load_dry, metrics, predict, scen_tdb, shifted_dry,  # noqa: E402
                   t_lim)

OUT = ROOT / "experiments/phase1/out/ruler_b"
RULER = ROOT / "experiments/phase1/out/ruler"
DAY_MONTH = np.repeat(np.arange(12), [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])  # month of each day of a 365-day year


def hottest_month_range(t_db, lon):
    """ASHRAE-convention hottest-month mean daily range of a (years, 8760) dry-bulb array (local days as AS-4)."""
    hm = int(np.argmax([t_db[:, MONTH == m].mean() for m in range(12)]))
    t = t_db.reshape(-1)
    tl = np.roll(t, int(round(lon / 15.0)))
    n = tl.shape[0] // 24
    days = tl[: n * 24].reshape(n, 24)
    sel = DAY_MONTH[np.arange(n) % 365] == hm
    return float(np.mean((days.max(axis=1) - days.min(axis=1))[sel])), hm


def h_star_years(P, W):
    lim = PR.fws_max(P)
    n = W["t_db"].shape[0]
    m = D.allowed_violations(P["eta_risk"], n)
    _, need, spare, _ = need_spare(P, W, max(D.k_grid(P, lim)), lim)
    from model import evaluate as E
    ey = np.array([E.sequent_peak(need[y], spare[y])[0] for y in range(n)])
    return float(np.sort(ey)[::-1][m]), m


def part_a():
    fr = pd.read_csv(RULER / "points.csv")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").set_index("site_id")
    dall = json.loads((ROOT / "experiments/phase1/out/cmip/deltas_all.json").read_text(encoding="utf-8"))
    exp, q, detrend = SCEN["ssp245_median"]
    yA = {}
    for c in fr.cell.unique():
        W = load_dry(WX / f"{c}.npz")
        lon = float(cells.loc[c, "era5_lon"])
        d12 = np.array(dall[exp][q])[:, dall["cells"].index(c)]
        for cl, t in (("historical", W["t_db"]), ("stress_plus1.2K", W["t_db"] + 1.2), ("ssp245_median", scen_tdb(W, d12, detrend))):
            yA[(c, cl)] = hottest_month_range(t, lon)[0]
    fr["y_A"] = [yA[(c, cl)] for c, cl in zip(fr.cell, fr.climate)]
    X = fr[["x", "y_A"]].to_numpy()
    res = {}
    for lab, thr in (("h8", 8.0), ("h0", 0.0)):
        truth = (fr.h_star <= thr + TOL).to_numpy()
        yb = truth.astype(float)
        full = fit(X, yb)
        p = np.zeros(len(fr))
        for c in fr.cell.unique():
            te = (fr.cell == c).to_numpy()
            p[te] = predict(fit(X[~te], yb[~te]), X[te])
        fr[f"pA_loco_{lab}"] = p
        w = fr.w_MW.to_numpy()
        m = metrics(p, truth, w)
        m["by_climate"] = {cl: metrics(p[(fr.climate == cl).to_numpy()], truth[(fr.climate == cl).to_numpy()], w[(fr.climate == cl).to_numpy()])
                           for cl in fr.climate.unique()}
        res[lab] = {"model_all_points": full, "loco": m}
    res["bar_met_h8"] = bool(res["h8"]["loco"]["error_among_confident_weighted"] <= 0.05 and res["h8"]["loco"]["marginal_share_weighted"] <= 0.30)
    fr.to_csv(OUT / "points_A.csv", index=False)
    # external (15 disjoint cells)
    ex = pd.read_csv(RULER / "external.csv")
    tabs = pd.concat([pd.read_csv(ROOT / "data/sites/md_hubs.csv"), pd.read_csv(ROOT / "data/sites/nonepoch_hot_sites.csv")]).set_index("site_id")
    ya = []
    for r in ex.itertuples():
        wdir = "weather_mdhubs" if r.set == "weather_mdhubs" else "weather_nonepoch"
        W = load_dry(ROOT / "data" / wdir / f"{r.site}.npz")
        t = W["t_db"] + (1.2 if r.climate == "stress_plus1.2K" else 0.0)
        ya.append(hottest_month_range(t, float(tabs.loc[r.site, "era5_lon"]))[0])
    ex["y_A"] = ya
    ex["pA_h8"] = predict(res["h8"]["model_all_points"], ex[["x", "y_A"]].to_numpy())
    ex.to_csv(OUT / "external_A.csv", index=False)
    res["external"] = {"h8": metrics(ex.pA_h8.to_numpy(), (ex.h_star <= 8 + TOL).to_numpy(), np.ones(len(ex)))}
    res["y_A_minus_y"] = {"mean": float((fr.y_A - fr.y).mean()), "min": float((fr.y_A - fr.y).min()), "max": float((fr.y_A - fr.y).max())}
    return res


def station_series(kind, row):
    if kind == "uscrn":
        t = load_uscrn(row.site_id)
    else:
        t = load_isd(row.isd_id, row.station_elev_m)[0]
    times = T0 + np.arange(N).astype("timedelta64[h]")
    md = (times.astype("datetime64[D]") - times.astype("datetime64[M]")).astype(int) + 1
    mo = times.astype("datetime64[M]").astype(int) % 12 + 1
    keep = ~((mo == 2) & (md == 29))
    t = t[keep]
    years = times[keep].astype("datetime64[Y]").astype(int) + 1970
    # fill gaps <= 6 h by linear interpolation along the continuous series
    isn = np.isnan(t)
    d = np.diff(np.concatenate([[0], isn.astype(int), [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    tf = t.copy()
    for a, b in zip(s, e):
        if b - a <= 6 and a > 0 and b < len(t):
            tf[a:b] = np.interp(np.arange(a, b), [a - 1, b], [t[a - 1], t[b]])
    ny = len(np.unique(years))
    T = tf.reshape(ny, 8760)
    raw = t.reshape(ny, 8760)
    ok = ~np.isnan(T).any(axis=1) & (~np.isnan(raw)).any(axis=1)
    return T, np.unique(years), ok


def part_b(model8, modelA):
    P = PR.params(ch_etaII=0.5076)
    rows = []
    for kind, tab, wx in (("uscrn", "as2_uscrn_stations.csv", pathlib.Path("D:/_caches/era5/weather_uscrn")),
                          ("isd", "bv3_stations.csv", pathlib.Path("D:/_caches/era5/weather_stations"))):
        st = pd.read_csv(ROOT / "data/sites" / tab)
        for r in st.itertuples():
            Ts, yrs, ok = station_series(kind, r)
            We = load_dry(wx / f"{r.site_id}.npz")
            assert np.array_equal(We["years"], yrs)
            K = np.where(ok)[0]
            p_std = 101325.0 * (1 - 2.25577e-5 * r.station_elev_m) ** 5.25588
            rec = {"station": r.site_id, "kind": kind, "years_kept": int(len(K)), "first": int(yrs[K[0]]) if len(K) else None}
            for src, tdb, p in (("station", Ts[K], np.full((len(K), 8760), p_std)), ("era5", We["t_db"][K], We["p"][K])):
                t04, y = climate_coords({"t_db": tdb}, r.lon)
                yA, _ = hottest_month_range(tdb, r.lon)
                tl = t_lim(P, float(np.mean(p)))
                h, m = h_star_years(P, {"t_db": tdb, "p": p, "years": yrs[K]})
                x = t04 - tl
                rec.update({f"{src}_T04": t04, f"{src}_Tlim": tl, f"{src}_x": x, f"{src}_y": y, f"{src}_yA": yA, f"{src}_h_star": h,
                            f"{src}_m": m, f"{src}_p8": float(predict(model8, np.array([[x, y]]))[0]),
                            f"{src}_pA8": float(predict(modelA, np.array([[x, yA]]))[0])})
            rows.append(rec)
            print(r.site_id, len(K), flush=True)
    df = pd.DataFrame(rows)
    out = {"n_stations": len(df)}
    for mdl, col in (("ruler", "p8"), ("model_A", "pA8")):
        bs, be = band(df[f"station_{col}"].to_numpy()), band(df[f"era5_{col}"].to_numpy())
        ts, te_ = (df.station_h_star <= 8 + TOL).to_numpy(), (df.era5_h_star <= 8 + TOL).to_numpy()
        conf_s = bs != "marginal"
        conf_e = be != "marginal"
        out[mdl] = {"band_changes_station_vs_era5": int((bs != be).sum()),
                    "station_inputs_vs_station_h": {"confident": int(conf_s.sum()),
                                                    "errors": int((conf_s & (((bs == "in") & ~ts) | ((bs == "out") & ts))).sum())},
                    "era5_inputs_vs_era5_h": {"confident": int(conf_e.sum()),
                                              "errors": int((conf_e & (((be == "in") & ~te_) | ((be == "out") & te_))).sum())}}
    out["coordinate_shift_station_minus_era5"] = {k: {"median": float((df[f"station_{k}"] - df[f"era5_{k}"]).median()),
                                                      "min": float((df[f"station_{k}"] - df[f"era5_{k}"]).min()),
                                                      "max": float((df[f"station_{k}"] - df[f"era5_{k}"]).max())}
                                                  for k in ("T04", "Tlim", "x", "y", "yA", "h_star")}
    out["class_changes_station_vs_era5_h8"] = int(((df.station_h_star <= 8 + TOL) != (df.era5_h_star <= 8 + TOL)).sum())
    df.to_csv(OUT / "stations.csv", index=False)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"A": part_a()}
    model8 = json.loads((RULER / "ruler_summary.json").read_text(encoding="utf-8"))["h8"]["model_all_points"]
    res["B"] = part_b(model8, res["A"]["h8"]["model_all_points"])
    (OUT / "ruler_b_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"A": {k: v for k, v in res["A"].items() if k in ("bar_met_h8", "external", "y_A_minus_y")},
                      "A_loco_h8": {k: v for k, v in res["A"]["h8"]["loco"].items() if k != "by_climate"}, "B": res["B"]}, indent=1))


if __name__ == "__main__":
    main()
