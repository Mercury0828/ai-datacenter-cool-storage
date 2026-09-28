"""PR-CMIP step 2: CMIP6-based dated scenarios (trend-preserving monthly delta) at the 80 hub cells; per cell × scenario: h*(k_max)
(BV-1 kernel), the Phase-0 exact screen (F_dry at 8 h, F_nc), and h*(k_max) for the 5 K total-approach design (dry-cooler floor 4 K,
plate-HX 1 K) — the ruler's "1 K recovers the loss" check.
Scenarios: ssp245_median (central), ssp585_median, ssp585_p90 (high bound), ssp245_p10 (low bound), ssp245_median_plain (no detrend).
Usage: python experiments/phase1/cmip_scenarios.py [--workers 3]
Output: experiments/phase1/out/cmip/scen/<scenario>__<cell>.json and cmip_summary.json
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from as_stress import COMMIT, hstar, need_spare  # noqa: E402
from hw_screen import screen_path  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402
from model import psychro as ps  # noqa: E402

OUT = ROOT / "experiments/phase1/out/cmip/scen"
WX = pathlib.Path("D:/_caches/era5/weather_hub")
SCEN = {"ssp245_median": ("ssp245", "median", True), "ssp585_median": ("ssp585", "median", True),
        "ssp585_p90": ("ssp585", "p90", True), "ssp245_p10": ("ssp245", "p10", True), "ssp245_median_plain": ("ssp245", "median", False)}
# month of each hour of a 365-day year (Feb 29 dropped in the tables)
MONTH = np.repeat(np.arange(12), [31 * 24, 28 * 24, 31 * 24, 30 * 24, 31 * 24, 30 * 24, 31 * 24, 31 * 24, 30 * 24, 31 * 24, 30 * 24, 31 * 24])


def shift_hourly(W, d):
    """RH-preserving shift by an hourly array d (years × 8760), wet-bulb recomputed (as evaluate.shift_weather)."""
    t = W["t_db"]
    t_new = t + d
    rh = np.clip(ps.rel_humidity(t, W["t_dp"]), 1e-4, 1.0)
    pw = rh * ps.p_ws(t_new)
    td_new = np.minimum(ps.dewpoint_from_w(ps.w_from_pw(pw, W["p"]), W["p"]), t_new)
    out = dict(W)
    out.update(t_db=t_new, t_dp=td_new, t_wb=ps.wet_bulb(t_new, td_new, W["p"]))
    return out


def scenario_weather(W, delta12, detrend):
    ny = W["t_db"].shape[0]
    d = np.tile(np.asarray(delta12)[MONTH], (ny, 1))
    if detrend:
        yrs = np.asarray(W["years"], float)
        ann = W["t_db"].mean(axis=1)
        slope = np.polyfit(yrs, ann, 1)[0]
        d = d - (slope * (yrs - 2004.5))[:, None]
    return shift_hourly(W, d)


def run_item(args):
    scen, cell, delta12 = args
    out = OUT / f"{scen}__{cell}.json"
    if out.exists():
        return scen, cell, "skip"
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = scenario_weather(E.load_weather(WX / f"{cell}.npz"), delta12, SCEN[scen][2])
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    r = screen_path(P, W, lim, m)
    P5 = PR.params(ch_etaII=0.5076, dT_HX_design=1.0)
    n5, s5 = need_spare(P5, W, D.k_max(P5, lim), lim, P5["tank_charge_max"])
    rec = {"scenario": scen, "cell": cell, "commit": COMMIT, "delta12": list(map(float, delta12)), **r,
           "h_star_total_approach_5K": hstar(n5, s5, m)}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return scen, cell, "done"


def summarize():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    w = sites.B_full_MW.astype(float)
    res = {}
    for scen in SCEN:
        recs = {}
        for f in OUT.glob(f"{scen}__*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            recs[r["cell"]] = r
        if len(recs) < sites.cell_id.nunique():
            res[scen] = {"missing": sites.cell_id.nunique() - len(recs)}
            continue
        h = sites.cell_id.map(lambda c: recs[c]["h_star"])
        h5 = sites.cell_id.map(lambda c: recs[c]["h_star_total_approach_5K"])
        fnc = sites.cell_id.map(lambda c: recs[c]["F_nc"]).astype(bool)
        jja = np.average([np.mean(recs[c]["delta12"][5:8]) for c in sites.cell_id], weights=w)
        res[scen] = {"JJA_delta_capacity_weighted_K": float(jja),
                     "F_dry_curve_GW": {str(x): float(w[h <= x + 1e-9].sum() / 1e3) for x in (0, 2, 4, 8, 12, 24, 48)},
                     "F_dry_8h_GW_total_approach_5K": float(w[h5 <= 8 + 1e-9].sum() / 1e3),
                     "F_nc_GW": float(w[fnc].sum() / 1e3), "h_star_median_weighted": float(np.median(np.repeat(h.values, (w / 10).astype(int))))}
    (OUT.parent / "cmip_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    dall = json.loads((OUT.parent / "deltas_all.json").read_text(encoding="utf-8"))
    cells = dall["cells"]
    items = []
    for scen, (exp, q, _) in SCEN.items():
        arr = np.array(dall[exp][q])  # (12, cells)
        for j, c in enumerate(cells):
            if not (OUT / f"{scen}__{c}.json").exists():
                items.append((scen, c, arr[:, j].tolist()))
    print(f"{len(items)} items pending; commit {COMMIT}", flush=True)
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(n, r, flush=True)
    summarize()


if __name__ == "__main__":
    main()
