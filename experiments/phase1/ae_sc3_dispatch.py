"""SC-3 (MANIFEST.md, Applied Energy analyses): dispatch sensitivity of the water-ban coincident peak.

Per hub cell (CDS tables of record, historical, 1990-2025): the least-cost design under the ban from the frozen menu
experiments/phase1/out/ff/cds/<cell>__historical.json (aggregator selection), re-dispatched
  (a) chiller-first with its chiller (as the paper; reproduction gate),
  (b) store-first with the chiller re-sized to the second-largest yearly minimum under that rule,
  (c) store-first with the original chiller (violating site-years, information only);
and the water-allowed least-cost design (no chiller) as comparator. Fleet = UTC-aligned hourly sum weighted by B_full_MW.
Usage: python experiments/phase1/ae_sc3_dispatch.py [--workers 2]   (run through tools/runlock.py)
Writes experiments/phase1/out/ae/sc3/<cell>.npz and experiments/phase1/out/ae/sc3_dispatch.json.
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
from numba import njit  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import ff_aggregate as A  # noqa: E402
from ff_menu import subset_years  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import fleet as F  # noqa: E402
from model import params as PR  # noqa: E402

CDS = pathlib.Path("D:/_caches/era5/cds/tables")
MENU = ROOT / "experiments/phase1/out/ff/cds"
OUT = ROOT / "experiments/phase1/out/ae/sc3"
SUMMARY = ROOT / "experiments/phase1/out/ae/sc3_dispatch.json"


@njit(cache=True)
def sf_year(need, spare, e_max, charge_max, c_ch):
    """Store-first: discharge first, chiller covers the rest up to c_ch. Returns (max unmet, chiller duty series, charge)."""
    e = e_max
    worst = 0.0
    trim = np.zeros(need.shape[0])
    charge = np.zeros(need.shape[0])
    for h in range(need.shape[0]):
        if need[h] > 0.0:
            d = min(need[h], e)
            e -= d
            r = need[h] - d
            t = min(r, c_ch)
            trim[h] = t
            if r - t > worst:
                worst = r - t
        elif spare[h] > 0.0:
            c = min(spare[h], charge_max, e_max - e)
            e += c
            charge[h] = c
    return worst, trim, charge


@njit(cache=True)
def sf_min_chiller(need, spare, e_max, charge_max):
    hi = 0.0
    for h in range(need.shape[0]):
        if need[h] > hi:
            hi = need[h]
    w, _, _ = sf_year(need, spare, e_max, charge_max, 0.0)
    if w <= 1e-9:
        return 0.0
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        w, _, _ = sf_year(need, spare, e_max, charge_max, mid)
        if w <= 1e-9:
            hi = mid
        else:
            lo = mid
    return hi


def elec_from(cache, a, need, trim, charge):
    """Fan + chiller electricity per unit of design heat, hour by hour (same accounting as fleet.opex_dry_c, no pads)."""
    tdb, p = cache.W["t_db"], cache.W["p"]
    s = a["s_base"].copy()
    idx = charge > 0
    if idx.any():
        s[idx] = a["chain"].min_speed_duty(cache.q + charge[idx], tdb[idx], p[idx], a["fwr"])
    fan = a["chain"].fan_power(np.where(a["ok_dry"], s, 1.0))
    fan = np.where(need > 0, a["chain"].fan_power(1.0), fan)
    ch = np.where(trim > 0, trim / cache.cop, 0.0)
    return fan + ch


def cell_table(cell):
    pts = pd.read_csv(ROOT / "experiments/phase1/out/m2/cds_points.csv")
    row = pts[pts.names.str.contains(f"hub_cells:{cell}(?:;|$)", regex=True)]
    assert len(row) == 1, cell
    return CDS / f"{row.point.iloc[0]}.npz"


def run_cell(cell):
    out = OUT / f"{cell}.npz"
    if out.exists():
        return cell, "skip"
    rec = json.loads((MENU / f"{cell}__historical.json").read_text(encoding="utf-8"))
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = subset_years(E.load_weather(cell_table(cell)), 1990, 2025)
    cache = F.SiteCache(P, W, lim)
    pr = A.base_prices(rec)
    rk = A.ratio_key(rec, pr)
    ban, _ = A.best(rec["designs"], pr, rk, A.is_ban)
    lc, _ = A.best(rec["designs"], pr, rk)
    arrays = {}
    info = {"ban": {k: ban[k] for k in ("k", "ad", "tank_h", "c_ch")}}
    # (a) chiller-first, reproduction of the menu's accounting
    ev = F.eval_dry_c(cache, ban["k"], False, ban["tank_h"])
    oa = F.opex_dry_c(cache, ev, ban["c_ch"], detail=True)
    arrays["a_elec"] = oa["elec_h"].astype(np.float32)
    info["a_viol_years"] = int(oa["viol"].sum())
    info["a_peak_mean_menu"] = float(A.ox_of(ban, rk)["peak_MW_mean"])
    info["a_peak_mean_recalc"] = float(oa["peak_MW"].mean())
    # (b), (c) store-first
    a = cache.k_arrays(ban["k"], False)
    need, spare, e_max, cm = ev["need"], a["spare"], float(ban["tank_h"]), P["tank_charge_max"]
    ny = need.shape[0]
    if e_max > 0:
        req = np.array([sf_min_chiller(need[y], spare[y], e_max, cm) for y in range(ny)])
    else:
        req = need.max(axis=1)
    c_sf = float(np.sort(req)[-2]) if ny >= 2 else float(req.max())
    for lab, cc in (("b", c_sf), ("c", float(ban["c_ch"]))):
        trim = np.zeros_like(need)
        charge = np.zeros_like(need)
        viol = 0
        for y in range(ny):
            if e_max > 0:
                w, t, ch = sf_year(need[y], spare[y], e_max, cm, cc)
            else:
                t = np.minimum(need[y], cc)
                w = float((need[y] - t).max())
                ch = np.zeros(need.shape[1])
            trim[y], charge[y] = t, ch
            viol += int(w > E.TOL)
        arrays[f"{lab}_elec"] = elec_from(cache, a, need, trim, charge).astype(np.float32)
        info[f"{lab}_c_ch"] = cc
        info[f"{lab}_viol_years"] = viol
    # water-allowed least-cost (comparator)
    if lc["family"] == "dry":
        ol = F.opex_dry_c(cache, F.eval_dry_c(cache, lc["k"], lc["ad"], lc["tank_h"]), lc["c_ch"], detail=True)
        arrays["lc_elec"] = ol["elec_h"].astype(np.float32)
    else:
        ev2 = E.eval_tower(P, W, lc["kct"], lim)
        trim2 = np.minimum(ev2["need"], lc["c_ch"])
        arrays["lc_elec"] = (ev2["fan"] + np.where(trim2 > 0, trim2 / cache.cop, 0.0)).astype(np.float32)
    info["lc_c_ch"] = float(lc["c_ch"])
    tmp = OUT / f"{cell}.tmp.npz"
    np.savez_compressed(tmp, **arrays, info=np.array(json.dumps(info)))
    os.replace(tmp, out)
    return cell, "done"


def summarize():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    cw = sites.groupby("cell_id")["B_full_MW"].sum()
    res, gate = {}, []
    for key in ("lc", "a", "b", "c"):
        fleet = None
        chill = 0.0
        viol = 0
        n = 0
        for cell, w in cw.items():
            f = OUT / f"{cell}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            info = json.loads(str(z["info"]))
            e = z[f"{key}_elec"].astype(np.float64) * w
            fleet = e if fleet is None else fleet + e
            if key in ("a", "b", "c"):
                chill += w * (info["ban"]["c_ch"] if key == "a" else info[f"{key}_c_ch"]) / 1e3
                viol += info[f"{key}_viol_years"]
            if key == "a":
                gate.append((cell, info["a_peak_mean_menu"], info["a_peak_mean_recalc"]))
            n += 1
        coin = fleet.max(axis=1)
        res[key] = {"cells": n, "coincident_peak_GW_e_mean": float(coin.mean() / 1e3), "coincident_peak_GW_e_max": float(coin.max() / 1e3),
                    "annual_TWh_mean": float(fleet.sum(axis=1).mean() / 1e6), "chiller_GWth": chill, "violating_site_years": viol}
    dev = max(abs(m - r) / max(m, 1e-9) for _, m, r in gate)
    res["gate_max_rel_dev_site_peak_mean"] = dev
    res["ratio_ban_to_lc"] = {k: res[k]["coincident_peak_GW_e_mean"] / res["lc"]["coincident_peak_GW_e_mean"] for k in ("a", "b")}
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part in ("cells", "all"):
        cells = sorted(f.name.split("__")[0] for f in MENU.glob("cell_*__historical.json"))
        todo = [c for c in cells if not (OUT / f"{c}.npz").exists()]
        print(f"{len(cells)} cells, {len(todo)} pending", flush=True)
        with mp.Pool(a.workers) as pool:
            for i, (c, s) in enumerate(pool.imap_unordered(run_cell, todo), 1):
                print(f"{i}/{len(todo)} {c} {s}", flush=True)
    if a.part in ("summary", "all"):
        summarize()


if __name__ == "__main__":
    main()
