"""Full-hub physical-feasibility screen (all 93 Epoch sites, 80 ERA5 land cells): F_dry and F_nc under the frozen menu,
bounds and risk standard (η = 1/20, SAA, m = 1 of 36), headline prototype (45 °C, f_air = 0), S0.

Exact shortcut: violations are monotone non-increasing in dry-cooler size k, tank size and tower size, so
  F_dry(cell) = compliant(dry k_max, no adiabatic, 8 h tank, no chiller)
  F_nc(cell)  = F_dry or compliant(dry k_max, adiabatic, 8 h tank, no chiller) or compliant(tower kct_max, no chiller).
Outputs experiments/phase0/out/hub/hub_screen.json (per cell × climate; resumable per item).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / os.environ.get("HUB_OUT", "experiments/phase0/out/hub")
WX = pathlib.Path(os.environ.get("HUB_WX", "D:/_caches/era5/weather_hub"))
CELLS = os.environ.get("HUB_CELLS", "data/sites/hub_cells.csv")
SITES = os.environ.get("HUB_SITES", "data/sites/hub_sites.csv")
CLIMATES = {"current": 0.0, "2040": 1.2}
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def screen(cell):
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    kmax = D.k_max(P, lim)
    W0 = E.load_weather(WX / f"{cell}.npz")
    out = {}
    for c, dT in CLIMATES.items():
        p = OUT / "items" / f"{cell}__{c}.json"
        if p.exists():
            out[c] = json.loads(p.read_text(encoding="utf-8"))
            continue
        W = E.shift_weather(W0, dT) if dT else W0
        m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
        # exact short-circuit: F_nc is decided as soon as one compressor-free design complies
        viol_dry = int(np.sum(E.eval_dry(P, W, kmax, False, 8.0, lim)["cch_req"] > 1e-9))
        viol_ad = viol_ct = None
        if viol_dry > m:
            viol_ad = int(np.sum(E.eval_dry(P, W, kmax, True, 8.0, lim)["cch_req"] > 1e-9))
            if viol_ad > m:
                viol_ct = int(np.sum(E.eval_tower(P, W, max(P["kct_grid"]), lim)["cch_req"] > 1e-9))
        f_nc = viol_dry <= m or (viol_ad is not None and viol_ad <= m) or (viol_ct is not None and viol_ct <= m)
        rec = {"cell": cell, "climate": c, "m": m, "viol_dry_kmax_tank8": viol_dry, "viol_ad_kmax_tank8": viol_ad,
               "viol_tower_kctmax": viol_ct, "F_dry": viol_dry <= m, "F_nc": bool(f_nc), "commit": COMMIT}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        os.replace(tmp, p)
        out[c] = rec
    return cell, out


def main():
    (OUT / "items").mkdir(parents=True, exist_ok=True)
    (OUT / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    cells = pd.read_csv(ROOT / CELLS).site_id.tolist()
    sites = pd.read_csv(ROOT / SITES)
    if "cell_id" not in sites.columns:
        sites["cell_id"] = sites["site_id"]
    for b in ("B_2027_MW", "B_now_MW"):
        if b not in sites.columns:
            sites[b] = sites["B_full_MW"]
    res = {}
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        for cell, out in pool.imap_unordered(screen, cells):
            res[cell] = out
            print(cell, {c: (o["F_dry"], o["F_nc"]) for c, o in out.items()}, flush=True)
    summary = {}
    for c in CLIMATES:
        for basis in ("B_full_MW", "B_2027_MW", "B_now_MW"):
            w = sites[basis].astype(float)
            fd = sites.cell_id.map(lambda x: res[x][c]["F_dry"]).astype(float)
            fn = sites.cell_id.map(lambda x: res[x][c]["F_nc"]).astype(float)
            summary[f"{c}|{basis}"] = {"denominator_MW": float(w.sum()), "F_dry": float((w * fd).sum() / w.sum()),
                                       "F_nc": float((w * fn).sum() / w.sum()),
                                       "F_nc_infeasible_MW": float((w * (1 - fn)).sum()),
                                       "F_dry_infeasible_MW": float((w * (1 - fd)).sum())}
    (OUT / "hub_screen.json").write_text(json.dumps({"summary": summary, "cells": res, "commit": COMMIT}, indent=1),
                                         encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
