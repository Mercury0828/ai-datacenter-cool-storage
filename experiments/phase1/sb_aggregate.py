"""PR-SB SB-2 / SB-3 aggregation (ledger PR-SB, declared before any run).
SB-2: F_dry(8 h) in GW for each of the 33 per-model 2040 fields (base design); minimum, family-balanced 10/50/90 % quantiles (12 PR-R2F
      families, p_m = 1 / (12 n_f)), maximum; per-cell family-balanced share feasible; classes over the 33 fields: consistently
      feasible / consistently infeasible / sensitive (GW).
SB-3: F_dry(8 h) at total approach 5 / 6 / 7 K - historical and +1.2 K from the existing AS2-2 grid (floor 4 K; HX 1 / 2 / 3 K),
      2040 median from the CDS rebuild (6 K) and the SB screens (5 K, 7 K); classes per climate.
Joint 2040 envelope (one factor at a time): robust core = feasible under all 33 fields at 6 K and under the median at 7 K; firmly
out = infeasible under all 33 at 6 K and under the median at 5 K; assumption-sensitive = the rest.
Check: hist_6K screens reproduce the CDS rebuild historical h* bit for bit at every cell.
Writes experiments/phase1/out/sb/sb23_robustness.json. Usage: python experiments/phase1/sb_aggregate.py
"""
import ast
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _grid(x):
    """AS2-2 'grid' is stored as a JSON object (older files may hold its repr as a string)."""
    return ast.literal_eval(x) if isinstance(x, str) else x

sys.path.insert(0, str(ROOT / "experiments/phase1"))
from r2f_cmip_2060 import FAMILIES, MODELS  # noqa: E402

OUT = ROOT / "experiments/phase1/out/sb"
H = 8.0 + 1e-9


def wq(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(np.searchsorted(cw, q - 1e-12), len(x) - 1)])


def main():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    cap = sites.groupby("cell_id").B_full_MW.sum() / 1e3  # GW per cell
    cells = list(cap.index)
    scr = {}
    for f in (OUT / "screen").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        scr[(r["cell"], r["variant"])] = r["h_star"]
    need = ["hist_6K", "median_5K", "median_7K"] + [f"m:{m}" for m in MODELS] + ["hist_5K", "hist_6.5K", "hist_7K", "s12_5K", "s12_7K",
                                                                                "p10_245_6K", "p90_585_6K"]
    missing = [(c, v) for c in cells for v in need if (c, v) not in scr]
    assert not missing, f"{len(missing)} screen results missing, e.g. {missing[:3]}"
    reb = {c: json.loads((ROOT / f"experiments/phase1/out/m2/cds_rebuild/{c}.json").read_text(encoding="utf-8")) for c in cells}
    check = max(abs(scr[(c, "hist_6K")] - reb[c]["historical"]["h_star"]) for c in cells)
    res = {"check_hist_6K_max_abs_diff_h": check}
    fam_of = {m: f for f, ms in FAMILIES.items() for m in ms}
    p = np.array([1.0 / (len(FAMILIES) * len(FAMILIES[fam_of[m]])) for m in MODELS])
    # SB-2
    feas = pd.DataFrame({m: [scr[(c, f"m:{m}")] <= H for c in cells] for m in MODELS}, index=cells)
    gw = np.array([cap[feas[m]].sum() for m in MODELS])
    share = feas.to_numpy(float) @ p
    cls = np.where(feas.all(axis=1), "consistently feasible", np.where(~feas.any(axis=1), "consistently infeasible", "sensitive"))
    res["SB2"] = {"F_dry_8h_GW_per_model": dict(zip(MODELS, gw.tolist())), "min": float(gw.min()), "max": float(gw.max()),
                  "family_balanced_q10_q50_q90": [wq(gw, p, q) for q in (0.1, 0.5, 0.9)],
                  "median_field_GW": float(cap[[reb[c]["ssp245_median"]["h_star"] <= H for c in cells]].sum()),
                  "classes_GW": {k: float(cap[cls == k].sum()) for k in ("consistently feasible", "sensitive", "consistently infeasible")},
                  "per_cell_share_feasible": dict(zip(cells, share.tolist()))}
    # SB-3
    sb3, mirror = {}, {}
    for cl, tag in (("historical", "hist"), ("stress_plus1.2K", "s12")):
        g = {c: _grid(json.loads((ROOT / f"experiments/phase1/out/as2/ruler/{c}__{cl}.json").read_text(encoding="utf-8"))["grid"])
             for c in cells}
        mirror[cl] = {a: float(cap[np.array([g[c][f"floor4_hx{hx}"]["h_star"] for c in cells]) <= H].sum())
                      for a, hx in (("5K", 1), ("6K", 2), ("6.5K", 2.5), ("7K", 3))}
        # PR-SB clarification 3: CDS data of record (headline path); AS2-2 mirror values kept beside
        sb3[cl] = {"5K": np.array([scr[(c, f"{tag}_5K")] for c in cells]),
                   "6K": np.array([reb[c][cl]["h_star"] for c in cells]),
                   "7K": np.array([scr[(c, f"{tag}_7K")] for c in cells])}
        if cl == "historical":
            sb3[cl]["6.5K"] = np.array([scr[(c, "hist_6.5K")] for c in cells])
    sb3["ssp245_median"] = {"5K": np.array([scr[(c, "median_5K")] for c in cells]),
                            "6K": np.array([reb[c]["ssp245_median"]["h_star"] for c in cells]),
                            "7K": np.array([scr[(c, "median_7K")] for c in cells])}
    res["SB3_mirror_AS2_2_GW"] = mirror
    res["C2_on_CDS_GW"] = {v: float(cap[np.array([scr[(c, v)] for c in cells]) <= H].sum()) for v in ("p10_245_6K", "p90_585_6K")}
    res["SB3"] = {}
    for cl, h in sb3.items():
        f5, f6, f7 = h["5K"] <= H, h["6K"] <= H, h["7K"] <= H
        res["SB3"][cl] = {"F_dry_8h_GW": {a: float(cap[h[a] <= H].sum()) for a in h},
                          "classes_GW": {"feasible at 7 K": float(cap[f7].sum()), "at 6 K not 7 K": float(cap[f6 & ~f7].sum()),
                                         "only at 5 K (lever-dependent)": float(cap[f5 & ~f6].sum()),
                                         "not even at 5 K": float(cap[~f5].sum())},
                          "monotone_check_cells_violating": int(np.sum((f6 & ~f5) | (f7 & ~f6)))}
    # joint 2040 envelope
    allm, nonem = feas.all(axis=1).to_numpy(), (~feas.any(axis=1)).to_numpy()
    med = sb3["ssp245_median"]
    core = allm & (med["7K"] <= H)
    out_ = nonem & (med["5K"] > H)
    res["joint_2040"] = {"robust core GW": float(cap[core].sum()), "firmly out GW": float(cap[out_].sum()),
                         "assumption-sensitive GW": float(cap[~core & ~out_].sum()), "frame GW": float(cap.sum()),
                         "robust_core_cells": [c for c, k in zip(cells, core) if k], "firmly_out_cells": [c for c, k in zip(cells, out_) if k]}
    (OUT / "sb23_robustness.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if k != "SB2" else {kk: vv for kk, vv in v.items() if kk not in ("per_cell_share_feasible", "F_dry_8h_GW_per_model")})
                      for k, v in res.items() if k != "joint_2040"}, indent=1))
    print(json.dumps({k: v for k, v in res["joint_2040"].items() if not k.endswith("cells")}, indent=1))


if __name__ == "__main__":
    main()
