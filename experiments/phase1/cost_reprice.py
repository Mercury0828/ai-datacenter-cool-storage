"""PR-COST (ledger 2026-09-26, declared before any run): reprice the saved PR-FF menus over the sourced cost basis
(`docs/cost_basis.md`) and re-select per site: one-at-a-time sourced variants, joint sourced-low / sourced-high cases, a dry-cooler
break-even sweep (assumed input) and an omitted-O&M check. Menus: uncorrected historical and +1.2 K; SSP2-4.5 circa 2040 (DD-016).
Writes experiments/phase1/out/cost/cost_summary.json.
Usage: python experiments/phase1/cost_reprice.py
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
MENUS = (("uncorrected", "historical"), ("uncorrected", "stress_plus1.2K"), ("cmip_ssp245_median", "ssp245_median"))
CEN = {"chiller": 300e3, "dc": 80e3, "ad_frac": 0.25, "tank": 35e3, "tower": 60e3, "crf": 0.1168, "elec_mult": 1.0, "om": False}
VARIANTS = {
    "central": {},
    "chiller_250k": {"chiller": 250e3}, "chiller_381k": {"chiller": 381e3},
    "store_15k": {"tank": 15e3}, "store_44k": {"tank": 44e3},
    "tower_44k": {"tower": 44e3}, "tower_100k": {"tower": 100e3},
    "elec_0.625x": {"elec_mult": 0.625}, "elec_1.5x": {"elec_mult": 1.5},
    "crf_6pct_20yr": {"crf": 0.0872}, "crf_10pct_15yr": {"crf": 0.1315},
    "pads_0.10": {"ad_frac": 0.10}, "pads_0.50": {"ad_frac": 0.50},
    "joint_sourced_low": {"chiller": 250e3, "tank": 15e3, "tower": 44e3, "crf": 0.0872, "elec_mult": 0.625},
    "joint_sourced_high": {"chiller": 381e3, "tank": 44e3, "tower": 100e3, "crf": 0.1315, "elec_mult": 1.5},
    "omitted_om": {"om": True},
}
DC_MULT = [0.5, 1.5, 2, 3, 4, 5, 6, 8, 10]


def annual_cost(d, it, v):
    """Annualized cost ($/yr per MW of IT) of one menu design under variant v."""
    q = dict(CEN, **v)
    pr = A.base_prices(it)
    pr["elec"] = pr["elec"] * q["elec_mult"]
    rk = A.ratio_key(it, pr)
    c = d["capex"]
    s_dc = q["dc"] / CEN["dc"]
    dc = c["dc"] * s_dc
    ad = c["ad"] * s_dc * q["ad_frac"] / CEN["ad_frac"]
    tank = c["tank_MWh"] * q["tank"]
    ch = c["chiller"] * q["chiller"] / CEN["chiller"]
    tw = c["tower"] * q["tower"] / CEN["tower"]
    ox = A.ox_of(d, rk)
    cost = q["crf"] * (dc + ad + tank + ch + tw) + ox["elec_MWh"] * pr["elec"] + ox["water_m3"] * pr["water"]
    if q["om"]:
        cost += 0.024 * ch + 0.02 * (dc + ad + tank + tw)
    return cost, ox["water_m3"]


def evaluate(items, sites, phys, v):
    rows = []
    for (cell, _), it in items.items():
        comp = [d for d in it["designs"] if d["compliant"]]
        costs = [annual_cost(d, it, v) for d in comp]
        i_lc = int(np.argmin([x[0] for x in costs]))
        ban_idx = [i for i, d in enumerate(comp) if A.is_ban(d)]
        i_ban = min(ban_idx, key=lambda i: costs[i][0])
        lc, ban = comp[i_lc], comp[i_ban]
        rows.append({"cell_id": cell, "lc_cost": costs[i_lc][0], "lc_water": costs[i_lc][1], "lc_zone": A.zone(lc),
                     "ban_cost": costs[i_ban][0], "ban_c_ch": ban["c_ch"], "ban_tank_h": ban["tank_h"]})
    df = sites.merge(pd.DataFrame(rows), on="cell_id")
    w = df.B_full_MW.astype(float).to_numpy()
    ph = df.site_id.map(phys).fillna(0.0).to_numpy()
    dcost = float((w * (df.ban_cost - df.lc_cost)).sum())
    dwat = float((w * df.lc_water).sum())
    return {"water_allowed_zone_share": {Z: float(w[(df.lc_zone == Z).to_numpy()].sum() / w.sum()) for Z in ("Z1", "Z2", "Z3")},
            "banned_Z3_share": float(w[(df.ban_c_ch > 1e-9).to_numpy()].sum() / w.sum()),
            "installed_GW_th": float((w * df.ban_c_ch).sum() / 1e3), "physical_GW_th": float((w * ph).sum() / 1e3),
            "economic_GW_th": float((w * np.clip(df.ban_c_ch.to_numpy() - ph, 0, None)).sum() / 1e3),
            "ban_store_GWh_th": float((w * df.ban_tank_h).sum() / 1e3),
            "ban_added_cost_MUSD_yr": dcost / 1e6, "water_saved_GL_yr": dwat / 1e6, "USD_per_m3": dcost / dwat}


def main():
    (OUT / "cost").mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    res = {"central_inputs": CEN, "variants": VARIANTS, "dc_multipliers": DC_MULT, "menus": {}}
    for tag, cl in MENUS:
        items = {}
        for f in (OUT / "ff" / tag).glob(f"cell_*__{cl}.json"):
            it = json.loads(f.read_text(encoding="utf-8"))
            items[(it["cell"], it["climate"])] = it
        fs = pd.read_csv(OUT / f"ff_sites_{tag}.csv")
        phys = fs[fs.climate == cl].set_index("site_id").cphys_ban_h8.to_dict()
        key = f"{tag}|{cl}"
        res["menus"][key] = {"items": len(items), "variants": {}, "dc_breakeven": {}}
        for name, v in VARIANTS.items():
            res["menus"][key]["variants"][name] = evaluate(items, sites, phys, v)
        for m in DC_MULT:
            res["menus"][key]["dc_breakeven"][f"{m:g}"] = evaluate(items, sites, phys, {"dc": CEN["dc"] * m})
        first = [m for m in DC_MULT if m > 1 and res["menus"][key]["dc_breakeven"][f"{m:g}"]["water_allowed_zone_share"]["Z3"] > 0.05]
        res["menus"][key]["dc_multiplier_water_allowed_Z3_gt5pct"] = first[0] if first else None
        sourced = [k for k in VARIANTS if k not in ("central",)]
        vals = [res["menus"][key]["variants"][k]["USD_per_m3"] for k in sourced]
        res["menus"][key]["USD_per_m3_range_sourced_and_om"] = [min(vals), max(vals)]
        res["menus"][key]["max_water_allowed_Z3_sourced_and_om"] = max(res["menus"][key]["variants"][k]["water_allowed_zone_share"]["Z3"]
                                                                        for k in sourced)
        print(key, "central $/m3", round(res["menus"][key]["variants"]["central"]["USD_per_m3"], 2), "range",
              [round(x, 2) for x in res["menus"][key]["USD_per_m3_range_sourced_and_om"]], "max WA Z3",
              round(res["menus"][key]["max_water_allowed_Z3_sourced_and_om"], 4), "dc break-even", res["menus"][key]["dc_multiplier_water_allowed_Z3_gt5pct"],
              flush=True)
    (OUT / "cost" / "cost_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
