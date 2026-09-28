"""PR-DE DE-3: joint tank–chiller refinement under a water ban. Compressor designs get continuous storage too: for every dry-cooler
size k (dry, no pads) and store h ∈ {0, 0.25, …, 8} h, the risk-sized chiller (2nd-largest yearly requirement, m = 1). The ban's
least-cost design is re-selected over the PR-FF menu (read from ff/<tag>) plus these designs. Exact capex-bound pruning: operating
cost ≥ 0, so a design whose annualized capex alone exceeds the best total so far cannot win (the bound is initialized with the
PR-FF ban least cost). Uses the cached evaluator (bit-identical to evaluate.py).
Usage: python experiments/phase1/de3_tank_chiller.py --tag TAG --weather-dir DIR [--workers 3]
Output: experiments/phase1/out/de/de3/<tag>/<cell>__<climate>.json
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
import ff_aggregate as A  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import fleet as F  # noqa: E402
from model import params as PR  # noqa: E402

COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
CLIMATES = {"historical": 0.0, "stress_plus1.2K": 1.2}
TANKS = [0.25 * i for i in range(33)]


def run_item(args):
    cell, climate, dT, wx_dir, tag, out_dir = args
    out = pathlib.Path(out_dir) / f"{cell}__{climate}.json"
    if out.exists():
        return cell, climate, "skip"
    t0 = time.time()
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    ff = json.loads((ROOT / "experiments/phase1/out/ff" / tag / f"{cell}__{climate}.json").read_text(encoding="utf-8"))
    pr = A.base_prices(ff)
    rk = A.ratio_key(ff, pr)
    ban_ff, ban_ff_cost = A.best(ff["designs"], pr, rk, A.is_ban)
    z3_ff, z3_ff_cost = A.best(ff["designs"], pr, rk, lambda d: A.is_ban(d) and d["c_ch"] > 1e-9)
    W = E.load_weather(pathlib.Path(wx_dir) / f"{cell}.npz")
    if dT:
        W = E.shift_weather(W, dT)
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    cache = F.SiteCache(P, W, lim)
    crf = P["crf"]
    cands = []
    for k in D.k_grid(P, lim):
        for th in TANKS:
            ev = F.eval_dry_c(cache, k, False, th)
            cc = D.chiller_for_risk(ev["cch_req"], m)
            if cc <= 1e-9:
                continue  # compressor-free designs are already in the PR-FF menu (grid + exact stores)
            capex = k * P["capex_dc_ref"] + th * P["capex_tank_per_mwh"] + cc * P["capex_chiller"]
            cands.append((crf * capex, k, th, cc, capex))
        cache._k.pop(float(k), None)
    cands.sort()
    best_cost = ban_ff_cost if ban_ff_cost is not None else np.inf
    best_z3 = z3_ff_cost if z3_ff_cost is not None else np.inf
    new_best, new_z3, n_opex = None, None, 0
    for lb, k, th, cc, capex in cands:
        if lb >= max(best_cost, best_z3):
            break
        ev = F.eval_dry_c(cache, k, False, th)
        ox = F.opex_dry_c(cache, ev, cc)
        n_opex += 1
        cost = crf * capex + float(np.mean(ox["elec_MWh"])) * pr["elec"] + float(np.mean(ox["water_m3"])) * pr["water"]
        rec = {"k": float(k), "tank_h": float(th), "c_ch": float(cc), "cost": cost, "elec_MWh": float(np.mean(ox["elec_MWh"])),
               "peak_MW_mean": float(np.mean(ox["peak_MW"]))}
        if cost < best_z3:
            best_z3, new_z3 = cost, rec
        if cost < best_cost:
            best_cost, new_best = cost, rec
    res = {"cell": cell, "climate": climate, "tag": tag, "commit": COMMIT,
           "ff_ban": {"design": A.design_str(ban_ff), "cost": ban_ff_cost, "zone": A.zone(ban_ff) if ban_ff else None,
                      "c_ch": ban_ff["c_ch"] if ban_ff else None},
           "ff_ban_z3_cost": z3_ff_cost, "refined_best_z3": new_z3 or (None if z3_ff is None else {"from_ff": A.design_str(z3_ff), "cost": z3_ff_cost}),
           "refined_ban_winner": new_best, "n_candidates": len(cands), "n_opex": n_opex, "elapsed_s": round(time.time() - t0, 1)}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    os.replace(tmp, out)
    return cell, climate, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weather-dir", required=True)
    a = ap.parse_args()
    out_dir = ROOT / "experiments/phase1/out/de/de3" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    items = [(c, cl, dT, a.weather_dir, a.tag, str(out_dir)) for c in cells for cl, dT in CLIMATES.items()
             if not (out_dir / f"{c}__{cl}.json").exists()]
    print(f"{len(items)} items pending; commit {COMMIT}; tag {a.tag}", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)} {r}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
