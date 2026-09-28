"""R2 precompute (protocol DD-010 / ledger PR-R2, commit 007f9c0): per site and design, the yearly chiller requirement
on the warming-shift grid, the hot-tail family, operating costs, and BL1 design-condition needs.

One .npz per site in experiments/phase0/out/r2/sites/, written atomically; manifest rewritten after each site;
resumes by skipping sites whose file exists (`--fresh` to restart). The DR / SAA / BL1 selection, the leave-years-out
radius rule and rules (a)/(b) are computed by r2_aggregate.py from these arrays.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker: the pool provides the parallelism (shared workstation)

import argparse
import json
import math
import multiprocessing as mp
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import components as C  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / "experiments/phase0/out/r2"
SITES_DIR = OUT / "sites"
DELTAS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.4, 3.0, 4.0, 5.0]  # DD-010
TRAIN = (1990, 2014)
TEST = (2015, 2025)
BL1_VARIANTS = ["2%", "1%", "0.4%", "5y", "10y", "20y", "50y", "100y"]
ETA2 = 0.5076
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def design_list(P, lim):
    ds = []
    for k in D.k_grid(P, lim):
        for ad in (False, True):
            for th in P["tank_hours_grid"]:
                ds.append(("dry", float(k), ad, float(th), 0.0))
    for kct in P["kct_grid"]:
        ds.append(("tower", 0.0, False, 0.0, float(kct)))
    return ds


def req_and_opex(P, W, d, lim, want_opex):
    """Yearly chiller requirement; operating quantities with no chiller (tank dispatched; unmet need uses no energy)
    and with an unlimited chiller (serves all need) — the selection uses the one matching the design's C (DD-010)."""
    fam, k, ad, th, kct = d
    if fam == "dry":
        ev = E.eval_dry(P, W, k, ad, th, lim)
        ox = (E.opex_dry(P, W, ev, 0.0, lim), E.opex_dry(P, W, ev, 1e9, lim)) if want_opex else None
    else:
        ev = E.eval_tower(P, W, kct, lim)
        ox = (E.opex_tower(P, W, ev, 0.0, lim), E.opex_tower(P, W, ev, 1e9, lim)) if want_opex else None
    return ev["cch_req"], ox


def gumbel_return(ann_max, n):
    """ASHRAE method-of-moments Gumbel return value (Handbook—Fundamentals, Ch. 14)."""
    mu, sd = float(np.mean(ann_max)), float(np.std(ann_max, ddof=1))
    f = -(math.sqrt(6.0) / math.pi) * (0.5772 + math.log(math.log(n / (n - 1.0))))
    return mu + sd * f


def design_conditions(W, years_mask):
    tdb = W["t_db"][years_mask]
    twb = W["t_wb"][years_mask]
    flat_db, flat_wb = tdb.ravel(), twb.ravel()
    cond = {}
    for pct in (2.0, 1.0, 0.4):
        db = float(np.percentile(flat_db, 100 - pct))
        wb = float(np.percentile(flat_wb, 100 - pct))
        mcwb = float(np.mean(flat_wb[np.abs(flat_db - db) <= 0.5]))
        mcdb = float(np.mean(flat_db[np.abs(flat_wb - wb) <= 0.5]))
        cond[f"{pct:g}%"] = {"dry": (db, mcwb), "wet": (mcdb, wb)}
    base = cond["0.4%"]
    for n in (5, 10, 20, 50, 100):
        db_n = gumbel_return(tdb.max(axis=1), n)
        wb_n = gumbel_return(twb.max(axis=1), n)
        cond[f"{n}y"] = {"dry": (db_n, base["dry"][1]), "wet": (base["wet"][0], wb_n)}
    return cond


def bl1_need(P, d, cond, p, lim):
    """Steady-state trim need of design d at its family's design condition(s) (no storage credit)."""
    fam, k, ad, th, kct = d
    if fam == "tower":
        _, wb = cond["wet"]
        return float(C.TowerChain(P, kct, lim).trim_need(1.0, np.array([wb]), np.array([p]))[0])
    ch = C.DryChain(P, k, lim)
    db, mcwb = cond["dry"]
    if not ad:
        return float(ch.trim_need(1.0, np.array([db]), np.array([p]))[0])
    mcdb, wb = cond["wet"]
    t1 = C.adiabatic_air_temp(db, mcwb, P["eta_ad"])
    t2 = C.adiabatic_air_temp(mcdb, wb, P["eta_ad"])
    return float(max(ch.trim_need(1.0, np.array([t1]), np.array([p]))[0], ch.trim_need(1.0, np.array([t2]), np.array([p]))[0]))


def run_site(site):
    t0 = time.time()
    P = PR.params(ch_etaII=ETA2)
    lim = PR.fws_max(P)
    W0 = E.load_weather(ROOT / f"data/weather/{site}.npz")
    years = W0["years"]
    tr = (years >= TRAIN[0]) & (years <= TRAIN[1])
    te = (years >= TEST[0]) & (years <= TEST[1])
    ds = design_list(P, lim)
    R = np.zeros((len(ds), len(DELTAS), len(years)))
    R4 = np.zeros((len(ds), len(years)))
    op_tr = np.zeros((len(ds), 2))  # [:, 0] no chiller, [:, 1] unlimited chiller
    op_te = np.zeros((len(ds), 2))
    ox_detail = []
    # Exact skips (DD-011): the yearly chiller requirement is monotone non-decreasing in the warming shift and
    # non-increasing in dry-cooler size k (same adiabatic/tank), tank size and tower size. So (i) a design with zero
    # requirement in every year at the largest shift has zero requirement at every smaller shift and in the hot-tail
    # family F4 (pointwise <= +1.8 K); (ii) at a given shift, a design that dominates one with zero requirement in every
    # year also has zero requirement. Operating costs at delta = 0 are always computed for every design.
    def dominates(a, b):  # a is at least as capable as b
        if a[0] != b[0]:
            return False
        if a[0] == "dry":
            return a[2] == b[2] and a[1] >= b[1] and a[3] >= b[3]
        return a[4] >= b[4]

    order = sorted(range(len(DELTAS)), key=lambda g: -DELTAS[g])  # largest shift first
    robust = set()
    # per-shift checkpoint (guide §7: full resumable state): resume a partially computed site after a stop
    part = SITES_DIR / f"{site}.partial.npz"
    done_g = []
    if part.exists():
        zp = np.load(part)
        R[:], op_tr[:], op_te[:] = zp["R"], zp["op_tr"], zp["op_te"]
        done_g = zp["done_g"].tolist()
        robust = set(zp["robust"].tolist())
        ox_detail = json.loads(str(zp["ox_detail"]))
    for g in order:
        if g in done_g:
            continue
        dT = DELTAS[g]
        W = E.shift_weather(W0, dT) if dT else W0
        zero_here = []
        for j, d in enumerate(ds):
            want = dT == 0.0
            if (j in robust or any(dominates(d, ds[i]) for i in zero_here)) and not want:
                R[j, g] = 0.0
                zero_here.append(j)
                continue
            if j in robust and want:
                _, ox = req_and_opex(P, W, d, lim, want_opex=True)
                req = np.zeros(len(years))
            else:
                req, ox = req_and_opex(P, W, d, lim, want_opex=want)
            R[j, g] = req
            if np.all(req <= 1e-12):
                zero_here.append(j)
                if g == order[0]:
                    robust.add(j)
            if ox is not None:
                for c, o in enumerate(ox):
                    cost = o["elec_MWh"] * P["elec_price"] + o["water_m3"] * P["water_price"]
                    op_tr[j, c] = float(np.mean(cost[tr]))
                    op_te[j, c] = float(np.mean(cost[te]))
                ox_detail.append({kk: float(np.mean(v[te])) for kk, v in ox[1].items() if kk != "viol"})
        done_g.append(g)
        tmpp = SITES_DIR / f"{site}.partial.tmp.npz"
        np.savez_compressed(tmpp, R=R, op_tr=op_tr, op_te=op_te, done_g=np.array(done_g), robust=np.array(sorted(robust)),
                            ox_detail=json.dumps(ox_detail))
        os.replace(tmpp, part)
    W4 = E.shift_weather(W0, 1.2, hot_amp=0.5)
    for j, d in enumerate(ds):
        R4[j] = 0.0 if j in robust else req_and_opex(P, W4, d, lim, want_opex=False)[0]
    cond = design_conditions(W0, tr)
    p_mean = float(np.mean(W0["p"]))
    NB = np.array([[bl1_need(P, d, cond[v], p_mean, lim) for v in BL1_VARIANTS] for d in ds])
    capex_parts = np.array([[d[1] * P["capex_dc_ref"] * (1 + (P["capex_ad_frac"] if d[2] else 0.0))
                             + d[3] * P["capex_tank_per_mwh"] if d[0] == "dry" else d[4] * P["capex_tower"]]
                            for d in ds])[:, 0]
    out = SITES_DIR / f"{site}.npz"
    tmp = SITES_DIR / f"{site}.tmp.npz"
    np.savez_compressed(tmp, R=R, R4=R4, op_tr=op_tr, op_te=op_te, NB=NB, capex_nochiller=capex_parts,
                        designs=np.array([json.dumps(d) for d in ds]), deltas=np.array(DELTAS), years=years,
                        bl1_variants=np.array(BL1_VARIANTS), cond=json.dumps(cond), commit=COMMIT,
                        ox_detail=json.dumps(ox_detail), config_hash=PR.config_hash(P))
    os.replace(tmp, out)
    part.unlink(missing_ok=True)
    return site, round(time.time() - t0, 1)


def manifest(sites):
    st = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "done": [s for s in sites if (SITES_DIR / f"{s}.npz").exists()],
          "pending": [s for s in sites if not (SITES_DIR / f"{s}.npz").exists()]}
    tmp = OUT / "manifest.tmp"
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, OUT / "manifest.json")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    SITES_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    sites = pd.read_csv(ROOT / "data/sites/phase0_sites.csv")
    sites = sites[sites.weighted].site_id.tolist()
    if a.only:
        sites = [s for s in sites if s == a.only]
    if a.fresh:
        for s in sites:
            (SITES_DIR / f"{s}.npz").unlink(missing_ok=True)
    st = manifest(sites)
    todo = st["pending"]
    print(f"{len(st['done'])} sites done, {len(todo)} pending (resume); commit {COMMIT}", flush=True)
    t0 = time.time()
    with mp.Pool(min(a.workers, max(1, len(todo)))) as pool:
        for site, dt in pool.imap_unordered(run_site, todo):
            st = manifest(sites)
            print(f"[{time.time() - t0:6.0f}s] {site} done in {dt}s; {len(st['done'])}/{len(sites)}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
