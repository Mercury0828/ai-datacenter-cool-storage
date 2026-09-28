"""PR-DE DE-1 (water-ban near-ties and regret) and DE-2 (capacity-estimate intervals, DD-003) — post-processing of the package-2
item files and site tables; no re-optimization. Writes experiments/phase1/out/de/de12_summary.json and prints a digest.
Usage: python experiments/phase1/de_analysis.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import ff_aggregate as A  # noqa: E402

OUT = ROOT / "experiments/phase1/out"
CL = ("historical", "stress_plus1.2K")
SIGMA = np.log(1.4) / 1.2816
NDRAW = 10000


def ban_zone_costs(it):
    pr = A.base_prices(it)
    rk = A.ratio_key(it, pr)
    ds = it["designs"]
    z1, c1 = A.best(ds, pr, rk, lambda d: A.is_ban(d) and d["c_ch"] <= 1e-9)
    z3, c3 = A.best(ds, pr, rk, lambda d: A.is_ban(d) and d["c_ch"] > 1e-9)
    return c1, c3


def de1(tag, sites):
    rows = []
    for f in (OUT / "ff" / tag).glob("*.json"):
        it = json.loads(f.read_text(encoding="utf-8"))
        c1, c3 = ban_zone_costs(it)
        rows.append({"cell_id": it["cell"], "climate": it["climate"], "z1_cost": c1, "z3_cost": c3})
    cz = pd.DataFrame(rows)
    fs = pd.read_csv(OUT / f"ff_sites_{tag}.csv")
    df = fs.merge(cz, on=["cell_id", "climate"], how="left")
    out = {}
    for cl in CL:
        g = df[df.climate == cl].copy()
        w = g.B_full_MW.astype(float)
        both = g.z1_cost.notna() & g.z3_cost.notna()
        lo = np.minimum(g.z1_cost, g.z3_cost)
        gap = (np.abs(g.z1_cost - g.z3_cost) / lo).where(both)
        econ = (g.ban_c_ch - g.cphys_ban_h8.fillna(0)).clip(lower=0)
        z3_lc = g.ban_zone == "Z3"
        # Corrected 2026-09-26 after internal review decision-evidence-and-stress R1: a missing Z1 cost compared as False, so infeasible
        # compressor-free alternatives were counted as near-ties. Feasibility is now masked explicitly (three categories).
        rel = (g.z1_cost - g.z3_cost) / g.z3_cost
        z1_infeasible = z3_lc & g.z1_cost.isna()
        far = z3_lc & g.z1_cost.notna() & (rel > 0.05)
        near = z3_lc & g.z1_cost.notna() & (rel <= 0.05)
        decisive = far | z1_infeasible  # registered convention: an infeasible alternative counts as decisive
        out[cl] = {"MW_gap_le_2pct": float(w[gap <= 0.02].sum()), "MW_gap_le_5pct": float(w[gap <= 0.05].sum()),
                   "economic_GW_th": float((w * econ).sum() / 1e3),
                   "economic_GW_th_decisive_gt5pct": float((w * econ)[decisive].sum() / 1e3),
                   "economic_GW_th_near_tie_le5pct": float((w * econ)[near].sum() / 1e3),
                   "economic_GW_th_by_category": {"feasible_alternative_gap_gt5pct": float((w * econ)[far].sum() / 1e3),
                                                  "feasible_alternative_gap_le5pct": float((w * econ)[near].sum() / 1e3),
                                                  "compressor_free_alternative_infeasible": float((w * econ)[z1_infeasible].sum() / 1e3)},
                   "regret_force_Z1_where_Z3_MUSD_yr": float((w * (g.z1_cost - g.z3_cost))[z3_lc & both].sum() / 1e6),
                   "regret_force_Z3_where_Z1_MUSD_yr": float((w * (g.z3_cost - g.z1_cost))[(g.ban_zone == "Z1") & both].sum() / 1e6),
                   "near_tie_sites_le5pct": g.site_id[(gap <= 0.05)].tolist()}
    return out, df


def de2(tag, df, flips):
    rng = np.random.default_rng(20260925)
    out = {}
    for cl in CL:
        g = df[df.climate == cl].reset_index(drop=True)
        w0 = g.B_full_MW.astype(float).to_numpy()
        fdry = g.cfdry_cost.notna().to_numpy()
        inst = g.ban_c_ch.fillna(0).to_numpy()
        phys = g.cphys_ban_h8.fillna(0).to_numpy()
        econ = np.clip(inst - phys, 0, None)
        dcost = (g.ban_cost - g.lc_cost).to_numpy()
        dwater = (g.lc_water_m3 - g.ban_water_m3).to_numpy()
        per = flips["tank_kUSD_per_MWh"]
        chg = np.array([per["17"]["_per_site"][f"{c}|{cl}"][1] != per["70"]["_per_site"][f"{c}|{cl}"][1] for c in g.cell_id])
        z = rng.standard_normal((NDRAW, len(w0)))
        W = w0 * np.exp(SIGMA * z)
        q = {"F_dry_8h_share": (W * fdry).sum(1) / W.sum(1), "F_dry_8h_GW": (W * fdry).sum(1) / 1e3,
             "ban_installed_GW_th": (W * inst).sum(1) / 1e3, "ban_physical_GW_th": (W * phys).sum(1) / 1e3,
             "ban_economic_GW_th": (W * econ).sum(1) / 1e3, "ban_added_cost_MUSD_yr": (W * dcost).sum(1) / 1e6,
             "ban_water_saved_GL_yr": (W * dwater).sum(1) / 1e6, "storage_cost_flip_GW_ban": (W * chg).sum(1) / 1e3}
        q["ban_USD_per_m3"] = q["ban_added_cost_MUSD_yr"] / q["ban_water_saved_GL_yr"]
        pt = {"F_dry_8h_share": (w0 * fdry).sum() / w0.sum(), "F_dry_8h_GW": (w0 * fdry).sum() / 1e3,
              "ban_installed_GW_th": (w0 * inst).sum() / 1e3, "ban_physical_GW_th": (w0 * phys).sum() / 1e3,
              "ban_economic_GW_th": (w0 * econ).sum() / 1e3, "ban_added_cost_MUSD_yr": (w0 * dcost).sum() / 1e6,
              "ban_water_saved_GL_yr": (w0 * dwater).sum() / 1e6, "storage_cost_flip_GW_ban": (w0 * chg).sum() / 1e3}
        pt["ban_USD_per_m3"] = pt["ban_added_cost_MUSD_yr"] / pt["ban_water_saved_GL_yr"]
        out[cl] = {k: {"point": float(pt[k]), "p10": float(np.quantile(v, 0.1)), "p50": float(np.quantile(v, 0.5)),
                       "p90": float(np.quantile(v, 0.9))} for k, v in q.items()}
        out[cl]["P_economic_ge_2GW_th"] = float(np.mean(q["ban_economic_GW_th"] >= 2.0))
        out[cl]["fully_correlated_bound_economic_GW_th"] = [pt["ban_economic_GW_th"] / 1.4, pt["ban_economic_GW_th"] * 1.4]
    return out


def main():
    (OUT / "de").mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    res = {"sigma": SIGMA, "ndraw": NDRAW}
    for tag in ("uncorrected", "corrected"):
        d1, df = de1(tag, sites)
        flips = json.loads((OUT / f"ff_summary_{tag}.json").read_text(encoding="utf-8"))["flips|B_full_MW"]
        res[tag] = {"DE1": d1, "DE2": de2(tag, df, flips)}
    (OUT / "de" / "de12_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for tag in ("uncorrected", "corrected"):
        for cl in CL:
            a = res[tag]["DE1"][cl]
            b = res[tag]["DE2"][cl]
            print(f"{tag} {cl}: DE-1 econ {a['economic_GW_th']:.2f} GW_th = decisive {a['economic_GW_th_decisive_gt5pct']:.2f} + near-tie "
                  f"{a['economic_GW_th_near_tie_le5pct']:.2f}; MW gap<=2% {a['MW_gap_le_2pct']:.0f}, <=5% {a['MW_gap_le_5pct']:.0f}; regret "
                  f"force-Z1 {a['regret_force_Z1_where_Z3_MUSD_yr']:.1f} M$/yr, force-Z3 {a['regret_force_Z3_where_Z1_MUSD_yr']:.1f} M$/yr")
            print("   DE-2 10-90%:", {k: (round(v["p10"], 3), round(v["point"], 3), round(v["p90"], 3)) for k, v in b.items()
                                     if isinstance(v, dict)}, "P(econ>=2)", round(b["P_economic_ge_2GW_th"], 3))


if __name__ == "__main__":
    main()
