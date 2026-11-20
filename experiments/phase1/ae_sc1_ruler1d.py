"""SC-1 (MANIFEST.md, Applied Energy analyses): one-statistic baseline (design-point excess x only) for the two-statistic site screen.
Same fit / predict / band / metrics / leave-one-cell-out as experiments/phase1/ruler.py, on the frozen out/ruler/points.csv.
Usage: python experiments/phase1/ae_sc1_ruler1d.py   Output: experiments/phase1/out/ae/sc1_ruler1d.json
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import ruler as R  # noqa: E402

fr = pd.read_csv(ROOT / "experiments/phase1/out/ruler/points.csv")
two = json.loads((ROOT / "experiments/phase1/out/ruler/ruler_summary.json").read_text(encoding="utf-8"))
w = fr.w_MW.to_numpy()
res = {"lambda": R.LAM, "n_points": len(fr)}
for lab, thr in (("h8", 8.0), ("h0", 0.0)):
    truth = (fr.h_star <= thr + R.TOL).to_numpy()
    yb = truth.astype(float)
    out = {}
    for name, cols in (("one_stat_x", ["x"]), ("two_stat_xy_recomputed", ["x", "y"])):
        X = fr[cols].to_numpy()
        p = np.zeros(len(fr))
        for c in fr.cell.unique():
            te = (fr.cell == c).to_numpy()
            p[te] = R.predict(R.fit(X[~te], yb[~te]), X[te])
        m = R.metrics(p, truth, w)
        m["by_climate"] = {cl: R.metrics(p[(fr.climate == cl).to_numpy()], truth[(fr.climate == cl).to_numpy()],
                                         w[(fr.climate == cl).to_numpy()]) for cl in fr.climate.unique()}
        out[name] = m
    out["two_stat_frozen"] = two[lab]["loco"]
    res[lab] = out
o = ROOT / "experiments/phase1/out/ae/sc1_ruler1d.json"
o.write_text(json.dumps(res, indent=1), encoding="utf-8")
for lab in ("h8", "h0"):
    for k in ("one_stat_x", "two_stat_xy_recomputed", "two_stat_frozen"):
        m = res[lab][k]
        print(lab, k, round(m["error_among_confident_weighted"], 4), m["n_errors"], round(m["marginal_share_weighted"], 4), m["n_marginal"])
