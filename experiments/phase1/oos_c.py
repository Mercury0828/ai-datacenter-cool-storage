"""PR-OOS-c (ledger 2026-09-26, declared after the PR-OOS-b result, before any run): drift-aware (additive temperature) margins.
Drift per cell from training data only: b_c = Theil-Sen slope of the JJA-mean dry-bulb over 1990-2014; delta_c = max(0, b_c) x
(use-period midpoint - 2002): 18 yr for the observed 2015-2025 test and its +1.2 K variant, 38.5 yr for the SSP2-4.5 2040s scenario.
(a) water-free design: h*_drift = 2nd-largest continuous-state E*_y over the 25 training years with dry-bulb + delta_c; retained if
    <= 8 h; a test year violates iff its E*_y > h*_drift.
(b)/(c) the PR-OOS-b hardware kept; its chiller re-sized to the 2nd-largest requirement over the training years shifted by delta_c
    (RH-preserving, DD-010); tests as PR-OOS-b; added annualized chiller cost (CRF x capex).
Writes experiments/phase1/out/oos_c/<cell>.json and experiments/phase1/out/oos_c_summary.json.
Usage: python experiments/phase1/oos_c.py [--workers 2]
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
from scipy.stats import theilslopes  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from bv1_min_storage import need_spare  # noqa: E402
from bv2_continuous import sequent_peak_continuous  # noqa: E402
from cmip_scenarios import SCEN, scenario_weather  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import fleet as F  # noqa: E402
from model import params as PR  # noqa: E402
from oos_b import TR, TOL, WX, evaluate_design, hottest3_share, second, test_sets  # noqa: E402

OUT = ROOT / "experiments/phase1/out/oos_c"
OOSB = ROOT / "experiments/phase1/out/oos_b"
HORIZON = {"obs": 2020.0 - 2002.0, "stress": 2020.0 - 2002.0, "scen": 2040.5 - 2002.0}


def drift_slope(W):
    yrs = np.asarray(W["years"])
    tr = (yrs >= 1990) & (yrs <= 2014)
    jja = W["t_db"][:, 151 * 24:243 * 24].mean(axis=1)
    return float(theilslopes(jja[tr], yrs[tr])[0])


def train_shift_dry(W, d):
    return {"years": W["years"][TR], "t_db": W["t_db"][TR] + d, "p": W["p"][TR]}


def run_cell(cell):
    out = OUT / f"{cell}.json"
    if out.exists():
        return cell, "skip"
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = E.load_weather(WX / f"{cell}.npz")
    dall = json.loads((ROOT / "experiments/phase1/out/cmip/deltas_all.json").read_text(encoding="utf-8"))
    exp, q, detrend = SCEN["ssp245_median"]
    Wsc = scenario_weather(W, np.array(dall[exp][q])[:, dall["cells"].index(cell)], detrend)
    sets = test_sets(W, E.shift_weather(W, 1.2), Wsc)
    b = drift_slope(W)
    delta = {k: max(0.0, b) * h for k, h in HORIZON.items()}
    prev = json.loads((OOSB / f"{cell}.json").read_text(encoding="utf-8"))
    k = max(D.k_grid(P, lim))
    rec = {"cell": cell, "slope_K_per_yr": b, "delta_K": delta, "a": {}, "b": {}, "c": {}}
    # (a) drift-aware water-free store
    tests_E = {"obs": np.array(prev["a"]["E_hist"]), "stress": np.array(prev["a"]["E_stress"]), "scen": np.array(prev["a"]["E_scen"])}
    for name, Wx, idx in sets:
        _, n_tr, s_tr, _ = need_spare(P, train_shift_dry(W, delta[name]), k, lim)
        ey_tr, _ = sequent_peak_continuous(np.ascontiguousarray(n_tr), np.ascontiguousarray(s_tr))
        S = second(ey_tr)
        e = tests_E[name][idx]
        v = e > S + TOL
        un = []
        if v.any():
            _, need, spare, _ = need_spare(P, Wx, k, lim)
            un = [float(F._unmet_hours(need[y], spare[y], S, P["tank_charge_max"], 0.0).sum()) for y in idx[v]]
        rec["a"][name] = {"h_drift": S, "viol": v.astype(int).tolist(), "storage_shortfall": (e[v] - S).tolist(), "unmet": un,
                          "tmax": Wx["t_db"][idx].max(axis=1).tolist()}
    # (b), (c) drift-aware chiller on the PR-OOS-b hardware
    W_train = {kk: (v[TR] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == W["t_db"].shape[0] else v) for kk, v in W.items()}
    shifted_train = {}  # training weather shifted once per distinct delta
    for lab in ("b", "c"):
        dsg = dict(prev[lab]["design"])
        c_old = float(dsg["c_ch"])
        rec[lab]["c_old"] = c_old
        req_cache = {}
        for name, Wx, idx in sets:
            dk = round(delta[name], 12)
            if dk not in shifted_train:
                shifted_train[dk] = E.shift_weather(W_train, dk) if dk > 0 else W_train
            if dk not in req_cache:
                req_cache[dk] = evaluate_design(P, lim, dsg, shifted_train[dk])[0]
            c_new = max(c_old, second(req_cache[dk]))
            c_new = c_new if c_new > 1e-9 else 0.0
            d2 = dict(dsg, c_ch=c_new)
            req, unmet = evaluate_design(P, lim, d2, Wx)
            v = req[idx] > c_new + TOL
            rec[lab][name] = {"c_new": c_new, "viol": v.astype(int).tolist(), "shortfall": np.maximum(req[idx] - c_new, 0.0).tolist(),
                              "unmet": unmet[idx].tolist(), "tmax": Wx["t_db"][idx].max(axis=1).tolist()}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, "done"


def summarize():
    P = PR.params(ch_etaII=0.5076)
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    recs = {json.loads(f.read_text(encoding="utf-8"))["cell"]: json.loads(f.read_text(encoding="utf-8")) for f in OUT.glob("cell_*.json")}
    sites = sites[sites.cell_id.isin(recs)]
    res = {"n_cells": len(recs), "n_sites": len(sites),
           "delta_K_obs": {"min": min(r["delta_K"]["obs"] for r in recs.values()), "max": max(r["delta_K"]["obs"] for r in recs.values()),
                           "capacity_weighted_mean": float(np.average([recs[c]["delta_K"]["obs"] for c in sites.cell_id], weights=sites.B_full_MW))}}
    out = {}
    for name in ("obs", "stress", "scen"):
        num = den = 0.0
        numu = denu = 0
        gw, ge2, top_v, all_v, un = 0.0, [], 0, 0, []
        for s in sites.itertuples():
            r = recs[s.cell_id]["a"][name]
            if r["h_drift"] > 8 + TOL:
                continue
            v = np.array(r["viol"], bool)
            num += s.B_full_MW * v.sum(); den += s.B_full_MW * len(v); numu += int(v.sum()); denu += len(v)
            gw += s.B_full_MW / 1e3
            if v.sum() >= 2:
                ge2.append(s.site_id)
            t3, tv = hottest3_share(v, r["tmax"]); top_v += t3; all_v += tv
            un += r["unmet"]
        out[name] = {"pooled_weighted": num / den, "pooled_unweighted": numu / denu, "retained_GW": gw, "sites_ge2": ge2,
                     "MW_ge2": float(sites[sites.site_id.isin(ge2)].B_full_MW.sum()), "share_in_3_hottest_years": top_v / all_v if all_v else None,
                     "unmet_heat_h_mean_in_violating_years": float(np.mean(un)) if un else 0.0,
                     "store_GWh_th": float(sum(s.B_full_MW * recs[s.cell_id]["a"][name]["h_drift"] for s in sites.itertuples()
                                               if recs[s.cell_id]["a"][name]["h_drift"] <= 8 + TOL) / 1e3)}
    res["a"] = out
    for lab in ("b", "c"):
        o = {}
        for name in ("obs", "stress", "scen"):
            num = den = 0.0
            numu = denu = 0
            ge2, top_v, all_v, un, add = [], 0, 0, [], 0.0
            for s in sites.itertuples():
                rr = recs[s.cell_id][lab]
                r = rr[name]
                v = np.array(r["viol"], bool)
                num += s.B_full_MW * v.sum(); den += s.B_full_MW * len(v); numu += int(v.sum()); denu += len(v)
                if v.sum() >= 2:
                    ge2.append(s.site_id)
                t3, tv = hottest3_share(v, r["tmax"]); top_v += t3; all_v += tv
                un += [u for u, x in zip(r["unmet"], v) if x]
                add += s.B_full_MW * (r["c_new"] - rr["c_old"])
            o[name] = {"pooled_weighted": num / den, "pooled_unweighted": numu / denu, "sites_ge2": ge2,
                       "MW_ge2": float(sites[sites.site_id.isin(ge2)].B_full_MW.sum()),
                       "share_in_3_hottest_years": top_v / all_v if all_v else None,
                       "unmet_heat_h_mean_in_violating_years": float(np.mean(un)) if un else 0.0,
                       "added_chiller_GW_th": add / 1e3, "added_chiller_cost_MUSD_yr": P["crf"] * add * P["capex_chiller"] / 1e6}
        res[lab] = o
    (OUT.parent / "oos_c_summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float)[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.summarize_only:
        (OUT / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
        cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
        todo = [c for c in cells if not (OUT / f"{c}.json").exists()]
        print(f"{len(todo)} cells pending", flush=True)
        with mp.Pool(a.workers) as pool:
            for n, r in enumerate(pool.imap_unordered(run_cell, todo), 1):
                print(n, r, flush=True)
    summarize()


if __name__ == "__main__":
    main()
