"""PR-OOS: selected designs outside their sizing data. Train 1990–2014 (25 site-years, ≤ 1 violating year), test 2015–2025 (11
site-years), uncorrected weather.
(a) water-free compressor-free design: dry k_max with store h*_train (continuous-state E*_y from bv2_continuous items);
(b) water-allowed and (c) water-ban least-cost designs re-selected on the train years from the PR-FF menu (train-sized chiller =
    2nd-largest train-year requirement; operating quantities from the stored compliant entry — declared approximation affecting only
    the selection cost). Observed-test violations from the stored yearly requirements; +1.2 K test violations by re-evaluating the
    selected design on the stress weather (cached evaluator).
Writes experiments/phase1/out/oos/<cell>.json and experiments/phase1/out/oos_summary.json.
Usage: python experiments/phase1/oos_designs.py [--workers 2]
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
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import fleet as F  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / "experiments/phase1/out/oos"
TR, TE = slice(0, 25), slice(25, 36)
M_TR = 1
WX = pathlib.Path("D:/_caches/era5/weather_hub")


def second(x):
    s = np.sort(np.asarray(x))[::-1]
    return float(s[M_TR]) if len(s) > M_TR else 0.0


def train_select(it, ban):
    """Least-cost design on the train years: for each (family, k, ad, tank, kct) the train-sized chiller."""
    pr = A.base_prices(it)
    rk = A.ratio_key(it, pr)
    fams = {}
    for d in it["designs"]:
        key = (d["family"], d["k"], d["ad"], round(d["tank_h"], 9), d["kct"])
        if ban and not (d["family"] == "dry" and not d["ad"]):
            continue
        cc = second(np.array(d["cch_req_years"])[TR])
        cc = cc if cc > 1e-9 else 0.0
        ox_d = d if d["compliant"] else None
        prev = fams.get(key)
        if prev is None:
            fams[key] = {"d": d, "cc": cc, "ox_d": ox_d}
        elif ox_d is not None:
            prev["ox_d"] = ox_d
    best, best_cost = None, np.inf
    for key, v in fams.items():
        if v["ox_d"] is None:
            continue
        d, cc = v["ox_d"], v["cc"]
        c = d["capex"]
        capex = c["dc"] + c["ad"] + c["tank_MWh"] * pr["tank"] + cc * 300e3 + c["tower"]
        ox = A.ox_of(d, rk)
        cost = pr["crf"] * capex + ox["elec_MWh"] * pr["elec"] + ox["water_m3"] * pr["water"]
        if cost < best_cost:
            best, best_cost = {"family": d["family"], "k": d["k"], "ad": d["ad"], "tank_h": d["tank_h"], "kct": d["kct"], "cc_train": cc,
                               "cch_req_years": v["d"]["cch_req_years"]}, cost
    return best, best_cost


def stress_req(P, W, lim, dsg):
    if dsg["family"] == "tower":
        return E.eval_tower(P, W, dsg["kct"], lim)["cch_req"]
    cache = F.SiteCache(P, W, lim)
    return F.eval_dry_c(cache, dsg["k"], dsg["ad"], dsg["tank_h"])["cch_req"]


def run_cell(cell):
    out = OUT / f"{cell}.json"
    if out.exists():
        return cell, "skip"
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    rec = {"cell": cell}
    # (a) water-free compressor-free design at k_max
    eh = np.array(json.loads((ROOT / f"experiments/phase1/out/bv2_continuous/uncorrected/{cell}__historical.json").read_text())["E_y_continuous"])
    es = np.array(json.loads((ROOT / f"experiments/phase1/out/bv2_continuous/uncorrected/{cell}__stress_plus1.2K.json").read_text())["E_y_continuous"])
    htr = second(eh[TR])
    rec["a"] = {"h_train": htr, "h_full36": second(eh), "test_E_obs": eh[TE].tolist(), "test_E_stress": es[TE].tolist()}
    # (b) and (c)
    it = json.loads((ROOT / f"experiments/phase1/out/ff/uncorrected/{cell}__historical.json").read_text())
    W = None
    for lab, ban in (("b", False), ("c", True)):
        dsg, cost = train_select(it, ban)
        req = np.array(dsg["cch_req_years"])
        if W is None:
            W = E.shift_weather(E.load_weather(WX / f"{cell}.npz"), 1.2)
        rs = stress_req(P, W, lim, dsg)
        rec[lab] = {"design": {k: v for k, v in dsg.items() if k != "cch_req_years"}, "train_cost": cost,
                    "test_viol_obs": int(np.sum(req[TE] > dsg["cc_train"] + 1e-9)),
                    "test_viol_stress": int(np.sum(rs[TE] > dsg["cc_train"] + 1e-9)),
                    "train_viol": int(np.sum(req[TR] > dsg["cc_train"] + 1e-9))}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, "done"


def summarize():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    recs = {json.loads(f.read_text())["cell"]: json.loads(f.read_text()) for f in OUT.glob("cell_*.json")}
    w_site = sites.set_index("site_id").B_full_MW.astype(float)
    res = {}
    # (a)
    rows = []
    for s in sites.itertuples():
        r = recs[s.cell_id]["a"]
        rows.append((s.site_id, s.B_full_MW, r["h_train"], r["h_full36"], np.array(r["test_E_obs"]), np.array(r["test_E_stress"])))
    def pooled(alpha, which):
        num = den = 0.0
        numu = denu = 0
        for sid, w, htr, _, eo, es in rows:
            if alpha * htr > 8 + 1e-9:
                continue  # site would not build the water-free design (store beyond 8 h)
            e = eo if which == "obs" else es
            v = int(np.sum(e > alpha * htr + 1e-9))
            num += w * v; den += w * len(e); numu += v; denu += len(e)
        return num / den, numu / denu
    res["a"] = {"F_dry_8h_GW_train": float(sum(w for _, w, h, _, _, _ in rows if h <= 8 + 1e-9) / 1e3),
                "F_dry_8h_GW_full36": float(sum(w for _, w, _, h, _, _ in rows if h <= 8 + 1e-9) / 1e3),
                "pooled_test_violation_obs": pooled(1.0, "obs"), "pooled_test_violation_stress": pooled(1.0, "stress")}
    margin = {}
    for alpha in [1.0 + 0.1 * i for i in range(11)]:
        po = pooled(alpha, "obs")
        margin[f"{alpha:.1f}"] = {"pooled_obs": po, "pooled_stress": pooled(alpha, "stress"),
                                  "F_dry_8h_GW": float(sum(w for _, w, h, _, _, _ in rows if alpha * h <= 8 + 1e-9) / 1e3)}
    res["a"]["margin"] = margin
    ok = [a for a, v in margin.items() if v["pooled_obs"][0] <= 0.05]
    res["a"]["smallest_alpha_obs_le_5pct"] = ok[0] if ok else None
    # (b), (c)
    for lab in ("b", "c"):
        num_o = num_s = den = 0.0
        many = []
        for s in sites.itertuples():
            r = recs[s.cell_id][lab]
            num_o += s.B_full_MW * r["test_viol_obs"]; num_s += s.B_full_MW * r["test_viol_stress"]; den += s.B_full_MW * 11
            if r["test_viol_obs"] >= 2:
                many.append(s.site_id)
        res[lab] = {"pooled_test_violation_obs_weighted": num_o / den, "pooled_test_violation_stress_weighted": num_s / den,
                    "sites_with_ge2_obs_test_violations": many}
    (OUT.parent / "oos_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
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
