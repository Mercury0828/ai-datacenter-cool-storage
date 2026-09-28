"""PR-OOS-b (ledger 2026-09-26, declared after internal review out-of-sample-transfer R1, before any run): training-only selection and a
prospective store margin.
(a) dry k_max water-free compressor-free design: forward-validated margin alpha_fwd (inner sizing 1990-2009, inner validation
    2010-2014, smallest alpha on {1.0, ..., 2.0} with pooled capacity-weighted inner violation <= 5 % on its retained cohort); stores
    alpha_fwd * h*_train (1990-2014) tested on 2015-2025 observed, the same years under +1.2 K, and the 36 years of the SSP2-4.5
    2040s central scenario (PR-CMIP weather; continuous-state E*_y).
(b)/(c) least-cost water-allowed / water-ban designs selected from the training-only PR-FF menu (ff_menu.py --years 1990-2014
    --tag train_1990_2014) with the aggregator's functions at base prices; the selected hardware is evaluated on the three test sets.
Writes experiments/phase1/out/oos_b/<cell>.json and experiments/phase1/out/oos_b_summary.json.
Usage: python experiments/phase1/oos_b.py [--workers 2] [--summarize-only]
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
import ff_aggregate as A  # noqa: E402
from bv1_min_storage import need_spare  # noqa: E402
from bv2_continuous import sequent_peak_continuous  # noqa: E402
from cmip_scenarios import SCEN, scenario_weather  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import fleet as F  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / "experiments/phase1/out/oos_b"
MENU = ROOT / "experiments/phase1/out/ff/train_1990_2014"
BV2 = ROOT / "experiments/phase1/out/bv2_continuous/uncorrected"
WX = pathlib.Path("D:/_caches/era5/weather_hub")
TR, INNER, VAL, TE = slice(0, 25), slice(0, 20), slice(20, 25), slice(25, 36)
ALPHAS = [round(1.0 + 0.1 * i, 1) for i in range(11)]
M = 1  # <= 1 violating year for 20, 25 and 36 years (floor(n / 20))
TOL = 1e-9


def second(x):
    return float(np.sort(np.asarray(x))[::-1][M])


def test_sets(W, Wst, Wsc):
    """(name, weather, year indices) of the three test sets."""
    return (("obs", W, np.arange(25, 36)), ("stress", Wst, np.arange(25, 36)), ("scen", Wsc, np.arange(36)))


def part_a(P, lim, cell, sets):
    k = max(D.k_grid(P, lim))
    eh = np.array(json.loads((BV2 / f"{cell}__historical.json").read_text(encoding="utf-8"))["E_y_continuous"])
    es = np.array(json.loads((BV2 / f"{cell}__stress_plus1.2K.json").read_text(encoding="utf-8"))["E_y_continuous"])
    rec = {"E_hist": eh.tolist(), "E_stress": es.tolist(), "h_inner": second(eh[INNER]), "h_train": second(eh[TR])}
    stores = sorted({round(a * rec["h_train"], 12) for a in ALPHAS} | {8.0})
    for name, Wx, idx in sets:
        _, need, spare, _ = need_spare(P, Wx, k, lim)
        need, spare = np.ascontiguousarray(need), np.ascontiguousarray(spare)
        ey, carry = sequent_peak_continuous(need, spare)
        if name in ("obs", "stress"):
            assert np.allclose(ey, eh if name == "obs" else es, atol=1e-9), "continuous E*_y disagrees with the stored BV-2 values"
        if name == "scen":
            rec["E_scen"], rec["carry_scen_max"] = ey.tolist(), float(carry.max())
        rec[f"tmax_{name}"] = Wx["t_db"][idx].max(axis=1).tolist()
        # unmet heat (full-IT-heat hours = MWh_th per MW) in violating test years, per candidate store (greedy dispatch, full at Jan 1)
        un = {}
        for S in stores:
            un[f"{S:.12g}"] = {int(y): float(F._unmet_hours(need[y], spare[y], S, P["tank_charge_max"], 0.0).sum())
                               for y in idx if ey[y] > S + TOL}
        rec[f"unmet_{name}"] = un
    return rec


def evaluate_design(P, lim, d, Wx):
    """Yearly chiller requirement and per-year unmet heat at the installed chiller for the selected hardware."""
    if d["family"] == "tower":
        ev = E.eval_tower(P, Wx, d["kct"], lim)
        req = np.asarray(ev["cch_req"])
        unmet = np.maximum(ev["need"] - d["c_ch"], 0.0).sum(axis=1)
        return req, unmet
    cache = F.SiteCache(P, Wx, lim)
    ev = F.eval_dry_c(cache, d["k"], d["ad"], d["tank_h"])
    req = np.asarray(ev["cch_req"])
    a = cache.k_arrays(d["k"], d["ad"])
    need = a["need_ad"] if d["ad"] else a["need_dry"]
    unmet = np.array([F._unmet_hours(need[y], a["spare"][y], d["tank_h"], P["tank_charge_max"], d["c_ch"]).sum()
                      if req[y] > d["c_ch"] + TOL else 0.0 for y in range(req.shape[0])])
    return req, unmet


def part_bc(P, lim, cell, sets):
    it = json.loads((MENU / f"{cell}__historical.json").read_text(encoding="utf-8"))
    assert it["years"] == [1990, 2014] and it["m"] == M
    pr = A.base_prices(it)
    rk = A.ratio_key(it, pr)
    out = {}
    for lab, ok in (("b", lambda d: True), ("c", A.is_ban)):
        d, cost = A.best(it["designs"], pr, rk, ok)
        r = {"design": {k: d[k] for k in ("family", "k", "ad", "tank_h", "kct", "c_ch")}, "train_cost": cost,
             "exact_store": "exact_h_star" in d, "zone": A.zone(d)}
        for name, Wx, idx in sets:
            req, unmet = evaluate_design(P, lim, d, Wx)
            if name == "obs":
                # built-in check: the full-weather requirement on the training years equals the training menu's stored one
                assert np.allclose(req[TR], np.array(d["cch_req_years"]), atol=1e-9), "train-year requirement mismatch"
                r["train_viol"] = int(np.sum(req[TR] > d["c_ch"] + TOL))
            v = req[idx] > d["c_ch"] + TOL
            r[name] = {"viol": v.astype(int).tolist(), "shortfall": np.maximum(req[idx] - d["c_ch"], 0.0).tolist(),
                       "unmet": unmet[idx].tolist(), "tmax": Wx["t_db"][idx].max(axis=1).tolist()}
        out[lab] = r
    return out


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
    rec = {"cell": cell, "a": part_a(P, lim, cell, sets), **part_bc(P, lim, cell, sets)}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, "done"


def hottest3_share(viol, tmax):
    """Share of a site's violating test years that fall in its three hottest test years (annual maximum dry-bulb)."""
    v = np.asarray(viol, bool)
    top = np.argsort(np.asarray(tmax))[::-1][:3]
    return int(v[top].sum()), int(v.sum())


def summarize():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    recs = {json.loads(f.read_text(encoding="utf-8"))["cell"]: json.loads(f.read_text(encoding="utf-8")) for f in OUT.glob("cell_*.json")}
    sites = sites[sites.cell_id.isin(recs)]
    res = {"n_cells": len(recs), "n_sites": len(sites)}
    # (a) forward-validated margin
    ladder = {}
    for a in ALPHAS:
        num = den = 0.0
        for s in sites.itertuples():
            r = recs[s.cell_id]["a"]
            if a * r["h_inner"] > 8 + TOL:
                continue
            e = np.array(r["E_hist"])[VAL]
            num += s.B_full_MW * np.sum(e > a * r["h_inner"] + TOL)
            den += s.B_full_MW * len(e)
        ladder[f"{a:.1f}"] = {"inner_validation_violation": num / den if den else None, "cohort_GW": den / 5 / 1e3}
    ok = [a for a in ALPHAS if ladder[f"{a:.1f}"]["inner_validation_violation"] is not None
          and ladder[f"{a:.1f}"]["inner_validation_violation"] <= 0.05 + 1e-12]
    a_fwd = ok[0] if ok else None
    res["a"] = {"inner_ladder": ladder, "alpha_fwd": a_fwd}
    if a_fwd is not None:
        for cohort in ("retained", "fixed"):
            out = {}
            for name, key, n in (("obs", "E_hist", 11), ("stress", "E_stress", 11), ("scen", "E_scen", 36)):
                num = den = 0.0
                numu = denu = 0
                ge2, gw, top_v, all_v, unmet, short = [], 0.0, 0, 0, [], []
                for s in sites.itertuples():
                    r = recs[s.cell_id]["a"]
                    h = r["h_train"]
                    S = a_fwd * h
                    if cohort == "retained" and S > 8 + TOL:
                        continue
                    if cohort == "fixed":
                        if h > 8 + TOL:
                            continue
                        S = min(S, 8.0)
                    e = np.array(r[key])
                    idx = np.arange(25, 36) if name != "scen" else np.arange(36)
                    v = e[idx] > S + TOL
                    num += s.B_full_MW * v.sum(); den += s.B_full_MW * n; numu += int(v.sum()); denu += n
                    gw += s.B_full_MW / 1e3
                    if v.sum() >= 2:
                        ge2.append(s.site_id)
                    t3, tv = hottest3_share(v, r[f"tmax_{name}"])
                    top_v += t3; all_v += tv
                    uk = min(r[f"unmet_{name}"], key=lambda x: abs(float(x) - S))
                    assert abs(float(uk) - S) < 1e-6
                    unmet += [r[f"unmet_{name}"][uk][str(int(y))] for y in idx[v]]
                    short += list(e[idx][v] - S)
                out[name] = {"pooled_weighted": num / den if den else None, "pooled_unweighted": numu / denu if denu else None,
                             "GW": gw, "sites_ge2_violating_years": ge2, "MW_ge2": float(sites[sites.site_id.isin(ge2)].B_full_MW.sum()),
                             "share_in_3_hottest_years": top_v / all_v if all_v else None,
                             "unmet_heat_h_mean_in_violating_years": float(np.mean(unmet)) if unmet else 0.0,
                             "storage_shortfall_h_mean": float(np.mean(short)) if short else 0.0}
            res["a"][cohort] = out
    # (b), (c)
    for lab in ("b", "c"):
        out = {"train_viol_max": max(recs[s.cell_id][lab]["train_viol"] for s in sites.itertuples()),
               "exact_store_selected_GW": float(sites[[recs[c][lab]["exact_store"] for c in sites.cell_id]].B_full_MW.sum() / 1e3),
               "Z3_share": float(sites[[recs[c][lab]["zone"] == "Z3" for c in sites.cell_id]].B_full_MW.sum() / sites.B_full_MW.sum())}
        for name in ("obs", "stress", "scen"):
            num = den = 0.0
            numu = denu = 0
            ge2, top_v, all_v, unmet, short = [], 0, 0, [], []
            for s in sites.itertuples():
                r = recs[s.cell_id][lab][name]
                v = np.array(r["viol"], bool)
                n = len(v)
                num += s.B_full_MW * v.sum(); den += s.B_full_MW * n; numu += int(v.sum()); denu += n
                if v.sum() >= 2:
                    ge2.append(s.site_id)
                t3, tv = hottest3_share(v, r["tmax"])
                top_v += t3; all_v += tv
                unmet += [u for u, x in zip(r["unmet"], v) if x]
                short += [u for u, x in zip(r["shortfall"], v) if x]
            out[name] = {"pooled_weighted": num / den, "pooled_unweighted": numu / denu, "sites_ge2_violating_years": ge2,
                         "MW_ge2": float(sites[sites.site_id.isin(ge2)].B_full_MW.sum()),
                         "share_in_3_hottest_years": top_v / all_v if all_v else None,
                         "unmet_heat_h_mean_in_violating_years": float(np.mean(unmet)) if unmet else 0.0,
                         "capacity_shortfall_mean": float(np.mean(short)) if short else 0.0}
        res[lab] = out
    (OUT.parent / "oos_b_summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
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
