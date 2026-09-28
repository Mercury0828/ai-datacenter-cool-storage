"""PR-DE DE-3 aggregation: the water-ban least-cost selection after the joint tank–chiller refinement vs PR-FF (B-full; both climates,
both weather tags): banned zone shares, installed / physically necessary / economic compressor capacity, the ban's added cost and
$/m³ (water-allowed least cost unchanged). Writes experiments/phase1/out/de/de3_summary.json.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from de_analysis import ban_zone_costs  # noqa: E402

OUT = ROOT / "experiments/phase1/out"
CL = ("historical", "stress_plus1.2K")


def z1_costs(tag):
    """Least-cost compressor-free water-ban design cost per (cell, climate); NaN if none is compliant (PR-FF menu)."""
    out = {}
    for f in (OUT / "ff" / tag).glob("cell_*.json"):
        it = json.loads(f.read_text(encoding="utf-8"))
        out[(it["cell"], it["climate"])] = ban_zone_costs(it)[0]
    return out


def near_tie_categories(g, w, econ, z1, cl):
    c1 = g.cell_id.map(lambda c: z1.get((c, cl), np.nan)).astype(float)
    z3 = g.new_c > 1e-9
    rel = (c1 - g.new_cost) / g.new_cost
    far = z3 & c1.notna() & (rel > 0.05)
    near = z3 & c1.notna() & (rel <= 0.05)
    inf = z3 & c1.isna()
    return {"feasible_alternative_gap_gt5pct": float((w * econ)[far].sum() / 1e3),
            "feasible_alternative_gap_le5pct": float((w * econ)[near].sum() / 1e3),
            "compressor_free_alternative_infeasible": float((w * econ)[inf].sum() / 1e3)}


def main():
    res = {}
    for tag in ("uncorrected", "corrected"):
        fs = pd.read_csv(OUT / f"ff_sites_{tag}.csv")
        d3 = {}
        for f in (OUT / "de" / "de3" / tag).glob("cell_*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            d3[(r["cell"], r["climate"])] = r
        res[tag] = {"items": len(d3)}
        z1 = z1_costs(tag)
        for cl in CL:
            g = fs[fs.climate == cl].copy()
            w = g.B_full_MW.astype(float)
            new_c, new_cost = [], []
            for c, c0, k0 in zip(g.cell_id, g.ban_c_ch, g.ban_cost):
                r = d3.get((c, cl))
                win = r["refined_ban_winner"] if r else None
                new_c.append(win["c_ch"] if win else c0)
                new_cost.append(win["cost"] if win else k0)
            g["new_c"], g["new_cost"] = new_c, new_cost
            phys = g.cphys_ban_h8.fillna(0)
            old_econ = (g.ban_c_ch - phys).clip(lower=0)
            new_econ = (g.new_c - phys).clip(lower=0)
            dw = (g.lc_water_m3 - g.ban_water_m3)
            res[tag][cl] = {
                "missing_items": int(sum((c, cl) not in d3 for c in g.cell_id.unique())),
                "banned_Z3_share": {"PR-FF": float(w[g.ban_c_ch > 1e-9].sum() / w.sum()), "refined": float(w[g.new_c > 1e-9].sum() / w.sum())},
                "installed_GW_th": {"PR-FF": float((w * g.ban_c_ch).sum() / 1e3), "refined": float((w * g.new_c).sum() / 1e3)},
                "physically_necessary_GW_th": float((w * phys).sum() / 1e3),
                "economic_GW_th": {"PR-FF": float((w * old_econ).sum() / 1e3), "refined": float((w * new_econ).sum() / 1e3)},
                "ban_added_cost_MUSD_yr": {"PR-FF": float((w * (g.ban_cost - g.lc_cost)).sum() / 1e6),
                                           "refined": float((w * (g.new_cost - g.lc_cost)).sum() / 1e6)},
                "USD_per_m3": {"PR-FF": float((w * (g.ban_cost - g.lc_cost)).sum() / (w * dw).sum()),
                               "refined": float((w * (g.new_cost - g.lc_cost)).sum() / (w * dw).sum())},
                # added 2026-09-26 after internal review decision-evidence-and-stress R1: near-tie categories of the refined winners' economic
                # capacity (a compressor-free alternative must be compliant to count as a near-tie)
                "economic_GW_th_refined_by_category": near_tie_categories(g, w, new_econ, z1, cl),
                "sites_switching_Z1_to_Z3": g.site_id[(g.ban_c_ch <= 1e-9) & (g.new_c > 1e-9)].tolist(),
                "MW_switching_Z1_to_Z3": float(w[(g.ban_c_ch <= 1e-9) & (g.new_c > 1e-9)].sum())}
    (OUT / "de" / "de3_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
