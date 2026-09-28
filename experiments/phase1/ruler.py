"""PR-RULER (ledger 2026-09-26, written before any run): the climate-space ruler. Coordinates x = T_0.4 - T_lim (99.6th percentile
of hourly dry-bulb minus the dry-bulb at which the k_max dry chain starts to need trim at the site's mean pressure) and y = mean daily
range on the days reaching T_0.4 (local days as AS-4); response h*(k_max) (headline kernel). Logistic screening model for
1{h* <= 8 h} and 1{h* = 0} (weak ridge), leave-one-cell-out evaluation on the 80 hub cells x {historical, +1.2 K, SSP2-4.5 2040s},
external check on 16 non-frame ERA5 cells, equipment-axis check on the AS2-2 approach grid.
Usage: python experiments/phase1/ruler.py
Output: experiments/phase1/out/ruler/{points.csv, external.csv, equipment.csv, ruler_summary.json, ruler.png}
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from bv1_min_storage import need_spare  # noqa: E402
from cmip_scenarios import MONTH, SCEN, scenario_weather  # noqa: E402
from model import components as C  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / "experiments/phase1/out/ruler"
WX = pathlib.Path("D:/_caches/era5/weather_hub")
BV1 = ROOT / "experiments/phase1/out/bv1/uncorrected"
SCN = ROOT / "experiments/phase1/out/cmip/scen"
AS2 = ROOT / "experiments/phase1/out/as2/ruler"
LAM = 0.01
TOL = 1e-9
T_GRID = np.round(np.arange(0.0, 60.0, 0.01), 2)


def load_dry(path):
    """Dry-bulb and pressure tables exactly as evaluate.load_weather (Feb 29 dropped), without the wet-bulb step the ruler
    does not use (the dry-route kernel reads t_db and p only)."""
    z = np.load(path)
    t0 = np.datetime64(str(z["t0"]).replace("Z", ""), "h")
    n = int(z["n_hours"])
    time = t0 + np.arange(n).astype("timedelta64[h]")
    t = z["t2m_c100"].astype(float) / 100.0
    p = z["psfc_pa10"].astype(float) * 10.0
    md = (time.astype("datetime64[D]") - time.astype("datetime64[M]")).astype(int) + 1
    mo = time.astype("datetime64[M]").astype(int) % 12 + 1
    keep = ~((mo == 2) & (md == 29))
    t, p, time = t[keep], p[keep], time[keep]
    years = time.astype("datetime64[Y]").astype(int) + 1970
    uy = np.unique(years)
    ny = len(uy)
    return {"years": uy, "t_db": t.reshape(ny, 8760), "p": p.reshape(ny, 8760)}


def shifted_dry(W, t_db):
    return {"years": W["years"], "t_db": t_db, "p": W["p"]}


def scen_tdb(W, delta12, detrend):
    """Dry-bulb of cmip_scenarios.scenario_weather (same delta and detrending), without the psychrometric step."""
    ny = W["t_db"].shape[0]
    d = np.tile(np.asarray(delta12)[MONTH], (ny, 1))
    if detrend:
        yrs = np.asarray(W["years"], float)
        slope = np.polyfit(yrs, W["t_db"].mean(axis=1), 1)[0]
        d = d - (slope * (yrs - 2004.5))[:, None]
    return W["t_db"] + d


def check_fast_path(cell, delta12):
    """Built-in check: the fast dry path reproduces load_weather / shift_weather / scenario_weather dry-bulb and pressure."""
    Wf = load_dry(WX / f"{cell}.npz")
    W = E.load_weather(WX / f"{cell}.npz")
    assert np.array_equal(Wf["t_db"], W["t_db"]) and np.array_equal(Wf["p"], W["p"]) and np.array_equal(Wf["years"], W["years"])
    assert np.allclose(Wf["t_db"] + 1.2, E.shift_weather(W, 1.2)["t_db"], atol=1e-12)
    assert np.allclose(scen_tdb(Wf, delta12, True), scenario_weather(W, delta12, True)["t_db"], atol=1e-12)


def t_lim(P, p_mean):
    lim = PR.fws_max(P)
    k = max(D.k_grid(P, lim))
    need = C.DryChain(P, k, lim).trim_need(1.0, T_GRID, np.full_like(T_GRID, p_mean))
    return float(T_GRID[np.argmax(need > 0)])


def climate_coords(W, lon):
    """(T_0.4, y) of one weather record: 99.6th percentile of all hours; mean daily range on local days reaching T_0.4."""
    t = W["t_db"].reshape(-1)
    t04 = float(np.quantile(t, 0.996))
    tl = np.roll(t, int(round(lon / 15.0)))
    n = tl.shape[0] // 24
    days = tl[: n * 24].reshape(n, 24)
    dmax, dmin = days.max(axis=1), days.min(axis=1)
    sel = dmax >= t04
    return t04, float(np.mean((dmax - dmin)[sel]))


def h_star_kernel(P, W):
    lim = PR.fws_max(P)
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    _, need, spare, _ = need_spare(P, W, max(D.k_grid(P, lim)), lim)
    ey = np.array([E.sequent_peak(need[y], spare[y])[0] for y in range(need.shape[0])])
    return float(np.sort(ey)[::-1][m])


def fit(X, yb):
    """Logistic regression with intercept; ridge LAM on standardized features (intercept not penalized)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])

    def nll(b):
        z = Z @ b
        return float(np.sum(np.logaddexp(0.0, z) - yb * z) + LAM * np.sum(b[1:] ** 2))

    def grad(b):
        pr = 1.0 / (1.0 + np.exp(-(Z @ b)))
        g = Z.T @ (pr - yb)
        g[1:] += 2 * LAM * b[1:]
        return g

    b = minimize(nll, np.zeros(Z.shape[1]), jac=grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 10000}).x
    return {"b": b.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}


def predict(model, X):
    b, mu, sd = np.array(model["b"]), np.array(model["mu"]), np.array(model["sd"])
    z = b[0] + ((X - mu) / sd) @ b[1:]
    return 1.0 / (1.0 + np.exp(-z))


def band(p):
    return np.where(p >= 0.9, "in", np.where(p <= 0.1, "out", "marginal"))


def metrics(p, truth, w):
    b = band(p)
    conf = b != "marginal"
    err = conf & (((b == "in") & ~truth) | ((b == "out") & truth))
    return {"error_among_confident_weighted": float(w[err].sum() / w[conf].sum()) if w[conf].sum() > 0 else None,
            "error_among_confident_unweighted": float(err.sum() / conf.sum()) if conf.sum() > 0 else None,
            "marginal_share_weighted": float(w[~conf].sum() / w.sum()), "marginal_share_unweighted": float((~conf).mean()),
            "n": int(len(p)), "n_errors": int(err.sum()), "n_marginal": int((~conf).sum())}


def frame_points():
    P = PR.params(ch_etaII=0.5076)
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv")
    dall = json.loads((ROOT / "experiments/phase1/out/cmip/deltas_all.json").read_text(encoding="utf-8"))
    exp, q, detrend = SCEN["ssp245_median"]
    rows = []
    c0 = cells.site_id.iloc[0]
    check_fast_path(c0, np.array(dall[exp][q])[:, dall["cells"].index(c0)])
    for c in cells.itertuples():
        W = load_dry(WX / f"{c.site_id}.npz")
        tl = t_lim(P, float(np.mean(W["p"])))
        d12 = np.array(dall[exp][q])[:, dall["cells"].index(c.site_id)]
        Ws = {"historical": W, "stress_plus1.2K": shifted_dry(W, W["t_db"] + 1.2),
              "ssp245_median": shifted_dry(W, scen_tdb(W, d12, detrend))}
        for cl, Wx in Ws.items():
            t04, y = climate_coords(Wx, c.era5_lon)
            if cl == "ssp245_median":
                h = json.loads((SCN / f"{cl}__{c.site_id}.json").read_text(encoding="utf-8"))["h_star"]
            else:
                h = json.loads((BV1 / f"{c.site_id}__{cl}.json").read_text(encoding="utf-8"))["k"]["1.4948"]["h_star"]
            rows.append({"cell": c.site_id, "climate": cl, "w_MW": float(c.B_full_MW), "p_mean": float(np.mean(Wx["p"])), "T_lim": tl,
                         "T_04": t04, "x": t04 - tl, "y": y, "h_star": h})
        print(c.site_id, flush=True)
    return pd.DataFrame(rows)


def external_points():
    """External cells, disjoint from the frame by ERA5 grid index (corrected 2026-09-26 after internal review climate-ruler R1: md_cheyenne
    shares ERA5 cell 525/301 with the frame's Meta Cheyenne and is excluded)."""
    P = PR.params(ch_etaII=0.5076)
    hub = pd.read_csv(ROOT / "data/sites/hub_cells.csv")
    frame_idx = set(zip(hub.era5_ilat, hub.era5_ilon))
    rows = []
    for tab, wdir in (("md_hubs.csv", "weather_mdhubs"), ("nonepoch_hot_sites.csv", "weather_nonepoch")):
        s = pd.read_csv(ROOT / "data/sites" / tab)
        for r in s.itertuples():
            if (r.era5_ilat, r.era5_ilon) in frame_idx:
                continue
            W = load_dry(ROOT / "data" / wdir / f"{r.site_id}.npz")
            tl = t_lim(P, float(np.mean(W["p"])))
            for cl, Wx in (("historical", W), ("stress_plus1.2K", shifted_dry(W, W["t_db"] + 1.2))):
                t04, y = climate_coords(Wx, r.era5_lon)
                rows.append({"site": r.site_id, "set": wdir, "climate": cl, "T_lim": tl, "T_04": t04, "x": t04 - tl, "y": y,
                             "h_star": h_star_kernel(P, Wx)})
    return pd.DataFrame(rows)


def equipment_points(fr):
    rows = []
    base = fr.set_index(["cell", "climate"])
    tl_cache = {}
    for f in sorted(AS2.glob("cell_*.json")):
        it = json.loads(f.read_text(encoding="utf-8"))
        b = base.loc[(it["cell"], it["climate"])]
        for key, g in it["grid"].items():
            if key == "floor4_hx2":
                continue
            fl, hx = (float(v) for v in key.replace("floor", "").split("_hx"))
            k = (fl, hx, round(b.p_mean, 1))
            if k not in tl_cache:
                P = PR.params(ch_etaII=0.5076, dc_approach_floor=fl, dT_HX_design=hx)
                lim = PR.fws_max(P)
                kk = D.k_max(P, lim)
                need = C.DryChain(P, kk, lim).trim_need(1.0, T_GRID, np.full_like(T_GRID, b.p_mean))
                tl_cache[k] = float(T_GRID[np.argmax(need > 0)])
            rows.append({"cell": it["cell"], "climate": it["climate"], "variant": key, "w_MW": b.w_MW, "T_lim": tl_cache[k],
                         "x": b.T_04 - tl_cache[k], "y": b.y, "h_star": g["h_star"]})
    return pd.DataFrame(rows)


def external_only():
    """Recompute the external check with the disjointness fix, using the stored full-frame models (no refit)."""
    res = json.loads((OUT / "ruler_summary.json").read_text(encoding="utf-8"))
    if "external_original_including_duplicate" not in res:
        res["external_original_including_duplicate"] = res["external"]
    ex = external_points()
    ex["p_h8"] = predict(res["h8"]["model_all_points"], ex[["x", "y"]].to_numpy())
    ex["p_h0"] = predict(res["h0"]["model_all_points"], ex[["x", "y"]].to_numpy())
    ex.to_csv(OUT / "external.csv", index=False)
    one = np.ones(len(ex))
    res["external"] = {"h8": metrics(ex.p_h8.to_numpy(), (ex.h_star <= 8 + TOL).to_numpy(), one),
                       "h0": metrics(ex.p_h0.to_numpy(), (ex.h_star <= TOL).to_numpy(), one), "n_cells": int(ex.site.nunique())}
    (OUT / "ruler_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["external"], indent=1))


def main():
    if "--external-only" in sys.argv:
        return external_only()
    OUT.mkdir(parents=True, exist_ok=True)
    fr = frame_points()
    fr.to_csv(OUT / "points.csv", index=False)
    X = fr[["x", "y"]].to_numpy()
    res = {"n_points": len(fr), "lambda": LAM}
    for lab, thr in (("h8", 8.0), ("h0", 0.0)):
        truth = (fr.h_star <= thr + TOL).to_numpy()
        yb = truth.astype(float)
        full = fit(X, yb)
        p_loco = np.zeros(len(fr))
        for c in fr.cell.unique():
            te = (fr.cell == c).to_numpy()
            p_loco[te] = predict(fit(X[~te], yb[~te]), X[te])
        fr[f"p_loco_{lab}"] = p_loco
        w = fr.w_MW.to_numpy()
        m = metrics(p_loco, truth, w)
        m["by_climate"] = {cl: metrics(p_loco[(fr.climate == cl).to_numpy()], truth[(fr.climate == cl).to_numpy()],
                                       w[(fr.climate == cl).to_numpy()]) for cl in fr.climate.unique()}
        res[lab] = {"model_all_points": full, "loco": m}
    res["bar_met_h8"] = bool(res["h8"]["loco"]["error_among_confident_weighted"] <= 0.05
                             and res["h8"]["loco"]["marginal_share_weighted"] <= 0.30)
    fr.to_csv(OUT / "points.csv", index=False)
    ex = external_points()
    ex["p_h8"] = predict(res["h8"]["model_all_points"], ex[["x", "y"]].to_numpy())
    ex["p_h0"] = predict(res["h0"]["model_all_points"], ex[["x", "y"]].to_numpy())
    ex.to_csv(OUT / "external.csv", index=False)
    one = np.ones(len(ex))
    res["external"] = {"h8": metrics(ex.p_h8.to_numpy(), (ex.h_star <= 8 + TOL).to_numpy(), one),
                       "h0": metrics(ex.p_h0.to_numpy(), (ex.h_star <= TOL).to_numpy(), one)}
    eq = equipment_points(fr)
    eq["p_h8"] = predict(res["h8"]["model_all_points"], eq[["x", "y"]].to_numpy())
    eq.to_csv(OUT / "equipment.csv", index=False)
    res["equipment"] = {"h8": metrics(eq.p_h8.to_numpy(), (eq.h_star <= 8 + TOL).to_numpy(), eq.w_MW.to_numpy())}
    (OUT / "ruler_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    figure(fr, ex, res)
    print(json.dumps({k: v for k, v in res.items() if k != "h0"}, indent=1)[:5000])


def figure(fr, ex, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xs = np.linspace(min(fr.x.min(), ex.x.min()) - 1, max(fr.x.max(), ex.x.max()) + 1, 300)
    ys = np.linspace(min(fr.y.min(), ex.y.min()) - 1, max(fr.y.max(), ex.y.max()) + 1, 300)
    gx, gy = np.meshgrid(xs, ys)
    G = np.column_stack([gx.ravel(), gy.ravel()])
    for lab, ls in (("h8", "-"), ("h0", "--")):
        pz = predict(res[lab]["model_all_points"], G).reshape(gx.shape)
        cs = ax.contour(gx, gy, pz, levels=[0.1, 0.5, 0.9], colors="k", linestyles=ls, linewidths=[0.6, 1.2, 0.6])
        ax.clabel(cs, fmt={0.1: f"{lab} 0.1", 0.5: f"{lab} 0.5", 0.9: f"{lab} 0.9"}, fontsize=6)
    cls = np.select([fr.h_star <= TOL, fr.h_star <= 8 + TOL, fr.h_star <= 24 + TOL], ["0 h", "(0, 8] h", "(8, 24] h"], "> 24 h")
    for c, col in (("0 h", "#2b83ba"), ("(0, 8] h", "#abdda4"), ("(8, 24] h", "#fdae61"), ("> 24 h", "#d7191c")):
        s = cls == c
        ax.scatter(fr.x[s], fr.y[s], s=np.sqrt(fr.w_MW[s]) + 4, c=col, edgecolors="k", linewidths=0.3, label=f"frame h* {c}")
    ax.scatter(ex.x, ex.y, marker="^", s=30, facecolors="none", edgecolors="purple", label="external cells")
    ax.set_xlabel("x = T_0.4 − T_lim (K)")
    ax.set_ylabel("y = mean daily range on days reaching T_0.4 (K)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "ruler.png", dpi=200)


if __name__ == "__main__":
    main()
