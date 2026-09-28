"""PR-CMIP window validation (after internal review cmip-scenarios R1): every selected store must supply 20 values for each calendar month in
its window (historical 1995-2014; SSP 2031-2050) after the same sort / de-duplication / year selection as cmip_extract.clim, and the
extracted deltas must be finite. A model-scenario that fails is excluded explicitly (listed with the reason), not silently by NaN.
Rebuilds experiments/phase1/out/cmip/deltas_all.json with explicit per-scenario model lists and checks that the ensemble quantiles
equal the previous file (which excluded the same entries implicitly through nan-aware quantiles).
Usage: python experiments/phase1/cmip_validate.py
Output: experiments/phase1/out/cmip/window_validation.json; deltas_all.json (rewritten; previous copy kept as deltas_all_v1.json)
"""
import json
import os
import pathlib
import shutil
import sys
import time

import gcsfs
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from cmip_extract import BASE, FUT, OUT, open_tas  # noqa: E402

WINDOWS = {"historical": BASE, "ssp245": FUT, "ssp585": FUT}


def month_counts(tas, period):
    tas = tas.sortby("time")
    _, keep = np.unique(tas["time"].values, return_index=True)
    t = tas["time"].values[np.sort(keep)]
    y0, y1 = int(period[0][:4]), int(period[1][:4])
    sel = [x for x in t if y0 <= x.year <= y1]
    return [sum(1 for x in sel if x.month == m) for m in range(1, 13)]


def main():
    fs = gcsfs.GCSFileSystem(token="anon")
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("deltas_*.json"))
            if f.name not in ("deltas_all.json", "deltas_all_v1.json")]
    val = {}
    for r in recs:
        val[r["model"]] = {}
        for e, period in WINDOWS.items():
            tas, _, _ = open_tas(fs, r["stores"][e])
            cnt = month_counts(tas, period)
            ok = all(c == int(period[1][:4]) - int(period[0][:4]) + 1 for c in cnt)
            val[r["model"]][e] = {"store": r["stores"][e], "months_per_calendar_month": cnt, "window_complete": ok}
        for e in ("ssp245", "ssp585"):
            finite = bool(np.isfinite(np.array(r[e], dtype=float)).all())
            val[r["model"]][e]["deltas_finite"] = finite
            val[r["model"]][e]["valid"] = bool(val[r["model"]][e]["window_complete"] and val[r["model"]]["historical"]["window_complete"] and finite)
        print(r["model"], {e: val[r["model"]][e]["window_complete"] for e in WINDOWS}, flush=True)
    (OUT / "window_validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
    old = json.loads((OUT / "deltas_all.json").read_text(encoding="utf-8"))
    if not (OUT / "deltas_all_v1.json").exists():
        shutil.copy(OUT / "deltas_all.json", OUT / "deltas_all_v1.json")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv")
    summ = {"models": [r["model"] for r in recs], "failed": old["failed"], "cells": old["cells"], "window_rule":
            "each window supplies 20 values per calendar month and finite deltas; otherwise the model-scenario is excluded"}
    for e in ("ssp245", "ssp585"):
        use = [r for r in recs if val[r["model"]][e]["valid"]]
        x = np.array([r[e] for r in use], dtype=float)
        summ[e] = {q: np.quantile(x, p, axis=0).tolist() for q, p in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))}
        summ[e]["models_used"] = [r["model"] for r in use]
        summ[e]["models_excluded"] = {r["model"]: val[r["model"]][e] for r in recs if not val[r["model"]][e]["valid"]}
        summ[e]["n_models"] = len(use)
        summ[e]["JJA_median_capacity_weighted"] = float(np.average(np.median(x, axis=0)[5:8].mean(axis=0), weights=cells.B_full_MW))
        for q in ("p10", "median", "p90"):
            same = np.allclose(np.array(summ[e][q]), np.array(old[e][q]), atol=1e-12, rtol=0)
            summ[e][f"{q}_identical_to_v1"] = bool(same)
    tmp = OUT / "deltas_all.tmp"
    tmp.write_text(json.dumps(summ), encoding="utf-8")
    for _ in range(30):  # atomic replace; other runs read this file (retry while a reader holds it open on Windows)
        try:
            os.replace(tmp, OUT / "deltas_all.json")
            break
        except PermissionError:
            time.sleep(2)
    print({e: (summ[e]["n_models"], list(summ[e]["models_excluded"]), [summ[e][f"{q}_identical_to_v1"] for q in ("p10", "median", "p90")])
           for e in ("ssp245", "ssp585")})


if __name__ == "__main__":
    main()
