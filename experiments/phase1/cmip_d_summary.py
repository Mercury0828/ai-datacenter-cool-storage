"""PR-CMIP / DD-016: package-2 D quantities under the SSP2-4.5 2040s median scenario beside the uncorrected historical and +1.2 K
values (B-full): water-allowed zone shares; water-ban split (installed / physically necessary / economic compressor capacity); the
ban's added cost, water saved and $ per m3; economic-capacity near-tie categories with the feasibility mask of the corrected DE-1.
Run after `ff_aggregate.py --tag cmip_ssp245_median --climates ssp245_median`. Writes experiments/phase1/out/cmip/cmip_d_summary.json.
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


def block(fs, tag, cl):
    g = fs[fs.climate == cl].copy()
    w = g.B_full_MW.astype(float)
    z1 = {}
    for f in (OUT / "ff" / tag).glob(f"cell_*__{cl}.json"):
        it = json.loads(f.read_text(encoding="utf-8"))
        z1[it["cell"]] = ban_zone_costs(it)[0]
    c1 = g.cell_id.map(lambda c: z1.get(c, np.nan)).astype(float)
    phys = g.cphys_ban_h8.fillna(0.0)
    econ = (g.ban_c_ch - phys).clip(lower=0)
    z3 = g.ban_c_ch > 1e-9
    rel = (c1 - g.ban_cost) / g.ban_cost
    dw = g.lc_water_m3 - g.ban_water_m3
    dc = g.ban_cost - g.lc_cost
    return {"water_allowed_zone_share": {Z: float(w[g.lc_zone == Z].sum() / w.sum()) for Z in ("Z1", "Z2", "Z3", "Zempty")},
            "ban_zone_share": {Z: float(w[g.ban_zone == Z].sum() / w.sum()) for Z in ("Z1", "Z3", "Zempty")},
            "ban_installed_GW_th": float((w * g.ban_c_ch.fillna(0)).sum() / 1e3),
            "ban_physically_necessary_GW_th": float((w * phys).sum() / 1e3),
            "ban_economic_GW_th": float((w * econ).sum() / 1e3),
            "economic_by_category_GW_th": {
                "feasible_alternative_gap_gt5pct": float((w * econ)[z3 & c1.notna() & (rel > 0.05)].sum() / 1e3),
                "feasible_alternative_gap_le5pct": float((w * econ)[z3 & c1.notna() & (rel <= 0.05)].sum() / 1e3),
                "compressor_free_alternative_infeasible": float((w * econ)[z3 & c1.isna()].sum() / 1e3)},
            "ban_added_cost_MUSD_yr": float((w * dc).sum() / 1e6), "ban_water_saved_GL_yr": float((w * dw).sum() / 1e6),
            "ban_USD_per_m3": float((w * dc).sum() / (w * dw).sum()),
            "compressor_free_water_free_feasible_GW": float(w[g.cfdry_cost.notna()].sum() / 1e3)}


def main():
    res = {}
    base = pd.read_csv(OUT / "ff_sites_uncorrected.csv")
    for cl in ("historical", "stress_plus1.2K"):
        res[cl] = block(base, "uncorrected", cl)
    scen = pd.read_csv(OUT / "ff_sites_cmip_ssp245_median.csv")
    res["ssp245_median"] = block(scen, "cmip_ssp245_median", "ssp245_median")
    (OUT / "cmip" / "cmip_d_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
