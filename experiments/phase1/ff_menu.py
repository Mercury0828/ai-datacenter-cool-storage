"""Package 2 (full-frame consequences; PR-FF in the ledger — DRAFT until registered): cost the full frozen design menu at every
cell × climate × weather tag, so that every package-2 estimand (least-cost routes with / without water, physical minimum
compressor capacity under storage and water budgets, resource cost of eliminating compressors, flip points, near-ties, violation
severity) is a post-processing step over one table per item.

Per item: every design (dry k × adiabatic × tank, tower kct) × chiller option (0 and the risk-sized capacity): yearly chiller
requirement, compliance, capex components, and operating quantities (means over site-years; annual-peak mean and max) at the
base prices and — for adiabatic designs, whose dispatch depends on the water / electricity price ratio — at each declared ratio
level; violating-year severity for compliant designs. Uses the cached evaluator (model/fleet.py, bit-identical to evaluate.py).
Usage: python experiments/phase1/ff_menu.py [--workers 3] [--weather-dir DIR] [--tag uncorrected]
Output: experiments/phase1/out/ff/<tag>/<cell>__<climate>.json (atomic; resumable).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import fleet as F  # noqa: E402
from model import params as PR  # noqa: E402

COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
CLIMATES = {"historical": 0.0, "stress_plus1.2K": 1.2}
RATIO_MULT = [0.0, 0.25, 0.5, 2.0 / 3.0, 1.0, 1.6, 2.0, 3.0, 5.0, 10.0, 20.0, 1e9]  # water/electricity price-ratio multipliers (one-at-a-time
# sweeps); 1e9 = water-minimizing dispatch (adiabatic only when dry mode cannot hold the limit), used for water budgets
QKEYS = ("fan_MWh", "ch_MWh", "water_m3", "ad_hours", "trim_hours", "elec_MWh")


def summarize(ox):
    s = {k: float(np.mean(ox[k])) for k in QKEYS}
    s["peak_MW_mean"] = float(np.mean(ox["peak_MW"]))
    s["peak_MW_max"] = float(np.max(ox["peak_MW"]))
    if "water_peak_m3h" in ox:
        s["water_peak_m3h_max"] = float(np.max(ox["water_peak_m3h"]))
    return s


def severity(d, viol_years):
    """Unmet duty in violating years (S0 is not zero lost compute)."""
    out = []
    for y in viol_years:
        u = d["unmet_h"][y]
        out.append({"year_idx": int(y), "hours": int(np.sum(u > E.TOL)), "unmet_MWh": float(u.sum()), "peak_unmet": float(u.max())})
    return out


def dry_rows(cache, P, m, k, ad, th, base_ratio, only_c0=False):
    ev = F.eval_dry_c(cache, k, ad, th)
    cc = D.chiller_for_risk(ev["cch_req"], m)
    out = []
    for c_ch in ((0.0, cc) if (cc > 1e-9 and not only_c0) else (0.0,)):
        n_viol = int(np.sum(ev["cch_req"] > c_ch + 1e-9))
        r = {"family": "dry", "k": float(k), "ad": bool(ad), "tank_h": float(th), "kct": 0.0, "c_ch": c_ch,
             "n_viol": n_viol, "compliant": n_viol <= m,
             "capex": {"dc": k * P["capex_dc_ref"], "ad": k * P["capex_dc_ref"] * P["capex_ad_frac"] if ad else 0.0,
                       "tank_MWh": float(th), "chiller": c_ch * P["capex_chiller"], "tower": 0.0},
             "cch_req_years": [float(x) for x in ev["cch_req"]]}
        if r["compliant"]:
            vy = [y for y in range(len(ev["cch_req"])) if ev["cch_req"][y] > c_ch + 1e-9]
            d = F.opex_dry_c(cache, ev, c_ch, detail=bool(vy))
            r["ox"] = summarize(d)
            r["severity"] = severity(d, vy) if vy else []
            if ad:
                r["ox_ratio"] = {f"{mult:.4g}": summarize(F.opex_dry_c(
                    cache, ev, c_ch, prices=(P["elec_price"], P["elec_price"] * base_ratio * mult))) for mult in RATIO_MULT}
        out.append(r)
    return out


HOURLY = pathlib.Path("D:/_caches/hotloop/ff_hourly")


def save_hourly(cache, P, W, lim, rec, tag, cell, climate):
    """Hourly cooling electricity and water of the base-price selections (least cost with / without water, compressor-free
    with / without water, tower counterfactual) for the coincident-peak estimand E6; float32 npz on D: (not in the repo).
    Selections use the aggregator's own functions, so they are the designs the aggregator reports."""
    import ff_aggregate as A  # noqa: E402  (same selection code as the aggregation)
    pr = A.base_prices(rec)
    rk = A.ratio_key(rec, pr)
    sel = {"lc": A.best(rec["designs"], pr, rk)[0], "ban": A.best(rec["designs"], pr, rk, A.is_ban)[0],
           "cf": A.best(rec["designs"], pr, rk, lambda d: d["c_ch"] <= 1e-9)[0],
           "cfdry": A.best(rec["designs"], pr, rk, lambda d: d["c_ch"] <= 1e-9 and A.is_ban(d))[0],
           "tower": A.best(rec["designs"], pr, rk, lambda d: d["family"] == "tower")[0]}
    arrays = {}
    for key, d in sel.items():
        if d is None:
            continue
        if d["family"] == "dry":
            o = F.opex_dry_c(cache, F.eval_dry_c(cache, d["k"], d["ad"], d["tank_h"]), d["c_ch"], detail=True)
            elec, water = o["elec_h"], o["water_h"]
        else:
            ev = E.eval_tower(P, W, d["kct"], lim)
            trim = np.minimum(ev["need"], d["c_ch"])
            elec = ev["fan"] + np.where(trim > 0, trim / cache.cop, 0.0)
            water = ev["water_m3h"]
        arrays[f"{key}_elec"] = elec.astype(np.float32)
        arrays[f"{key}_water"] = water.astype(np.float32)
    out = HOURLY / tag
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{cell}__{climate}.npz"
    tmp = out / f"{cell}__{climate}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, f)
    return str(f)


def subset_years(W, y0, y1):
    """Weather restricted to the calendar years y0..y1 (every per-year array; PR-OOS-b training-only menu)."""
    yrs = np.asarray(W["years"])
    sel = (yrs >= y0) & (yrs <= y1)
    return {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == yrs.shape[0] else v) for k, v in W.items()}


def run_item(args):
    cell, climate, dT, wx_dir, out_dir = args[:5]
    years, hourly = (args[5], args[6]) if len(args) > 5 else (None, True)
    out = pathlib.Path(out_dir) / f"{cell}__{climate}.json"
    if out.exists():
        return cell, climate, "skip"
    t0 = time.time()
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = E.load_weather(pathlib.Path(wx_dir) / f"{cell}.npz")
    if years:
        W = subset_years(W, *years)
    if dT is None:  # PR-CMIP scenario weather (DD-016): the climate label is the scenario name
        from cmip_scenarios import SCEN, scenario_weather
        exp, q, detrend = SCEN[climate]
        dall = json.loads((ROOT / "experiments/phase1/out/cmip/deltas_all.json").read_text(encoding="utf-8"))
        W = scenario_weather(W, np.array(dall[exp][q])[:, dall["cells"].index(cell)], detrend)
    elif dT:
        W = E.shift_weather(W, dT)
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    cache = F.SiteCache(P, W, lim)
    base_ratio = P["water_price"] / P["elec_price"]
    rows = []
    for k in D.k_grid(P, lim):
        for ad in (False, True):
            for th in P["tank_hours_grid"]:
                rows += dry_rows(cache, P, m, k, ad, th, base_ratio)
    # continuous (exact) minimal storage for compressor-free dry-family designs: tank = h*(k) (sequent peak, m = 1), added
    # when 0 < h* <= 8 h and not already on the grid (PR-FF; a menu refinement that favours compressor-free designs)
    for k in D.k_grid(P, lim):
        a = cache.k_arrays(k, True)
        for ad in (False, True):
            need = a["need_ad"] if ad else a["need_dry"]
            ey = np.array([E.sequent_peak(need[y], a["spare"][y])[0] for y in range(need.shape[0])])
            hs = float(np.sort(ey)[::-1][m])
            if 1e-9 < hs <= 8.0 and not any(abs(hs - g) < 1e-6 for g in P["tank_hours_grid"]):
                rows += [dict(r, exact_h_star=hs) for r in dry_rows(cache, P, m, k, ad, hs + 1e-7, base_ratio, only_c0=True)]
    for kct in P["kct_grid"]:
        ev = E.eval_tower(P, W, kct, lim)
        cc = D.chiller_for_risk(ev["cch_req"], m)
        for c_ch in ((0.0, cc) if cc > 1e-9 else (0.0,)):
            n_viol = int(np.sum(ev["cch_req"] > c_ch + 1e-9))
            r = {"family": "tower", "k": 0.0, "ad": False, "tank_h": 0.0, "kct": float(kct), "c_ch": c_ch, "n_viol": n_viol,
                 "compliant": n_viol <= m,
                 "capex": {"dc": 0.0, "ad": 0.0, "tank_MWh": 0.0, "chiller": c_ch * P["capex_chiller"],
                           "tower": kct * P["capex_tower"]},
                 "cch_req_years": [float(x) for x in ev["cch_req"]]}
            if r["compliant"]:
                ox = E.opex_tower(P, W, ev, c_ch, lim)
                r["ox"] = summarize(ox)
                vy = [y for y in range(len(ev["cch_req"])) if ev["cch_req"][y] > c_ch + 1e-9]
                r["severity"] = [{"year_idx": int(y), "hours": int(np.sum(np.maximum(ev["need"][y] - c_ch, 0) > E.TOL)),
                                  "unmet_MWh": float(np.maximum(ev["need"][y] - c_ch, 0).sum()),
                                  "peak_unmet": float(np.maximum(ev["need"][y] - c_ch, 0).max())} for y in vy]
                r["tower_peak_water_m3h"] = float(ev["water_m3h"].max())
            rows.append(r)
    rec = {"cell": cell, "climate": climate, "dT": dT, "m": m, "commit": COMMIT, "config_hash": PR.config_hash(P),
           "prices": {"elec": P["elec_price"], "water": P["water_price"], "crf": P["crf"],
                      "tank_per_MWh": P["capex_tank_per_mwh"]},
           "ratio_mult": RATIO_MULT, "designs": rows}
    rec["years"] = [int(W["years"][0]), int(W["years"][-1])]
    rec["hourly_file"] = save_hourly(cache, P, W, lim, rec, pathlib.Path(out_dir).name, cell, climate) if hourly else None
    rec["elapsed_s"] = round(time.time() - t0, 1)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, climate, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--weather-dir", default="D:/_caches/era5/weather_hub")
    ap.add_argument("--tag", default="uncorrected")
    ap.add_argument("--scenario", default=None, help="PR-CMIP scenario (cmip_scenarios.SCEN) run as the only climate (DD-016)")
    ap.add_argument("--years", default=None, help="Y0-Y1: restrict the weather to these years; historical climate only (PR-OOS-b)")
    ap.add_argument("--no-hourly", action="store_true", help="skip the hourly selection files (E6 coincident peaks)")
    a = ap.parse_args()
    climates = {a.scenario: None} if a.scenario else CLIMATES
    years = tuple(int(x) for x in a.years.split("-")) if a.years else None
    if years:
        climates = {"historical": 0.0}
    out_dir = ROOT / "experiments/phase1/out/ff" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    items = [(c, cl, dT, a.weather_dir, str(out_dir), years, not a.no_hourly) for c in cells for cl, dT in climates.items()
             if not (out_dir / f"{c}__{cl}.json").exists()]
    print(f"{len(items)} items pending of {len(climates) * len(cells)}; commit {COMMIT}; tag {a.tag}", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(run_item, items), 1):
            print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
