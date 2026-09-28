"""PR-COST-c (ledger 2026-09-26, declared after internal review R2, before any run): differential pumping bound.
A pumping power of +/-5 kW per MW of design heat, at full load in every hour, is charged to tower designs relative to dry-family
designs at the variant's electricity price. Cases: dry-cooler multipliers {1, 2, 2.25, 2.5, 2.75, 3, 4} x pads scaled / decoupled x
pumping {-5, 0, +5} kW/MW, three menus. Writes experiments/phase1/out/cost/cost_c_summary.json.
Usage: python experiments/phase1/cost_reprice_c.py
"""
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import cost_reprice as C  # noqa: E402
import cost_reprice_b as B  # noqa: E402

MULT = [1, 2, 2.25, 2.5, 2.75, 3, 4]
PUMP_KW_PER_MW = [-5.0, 0.0, 5.0]


def make_cost(pump_kw):
    def annual_cost(d, it, v):
        cost, water = B.annual_cost(d, it, v)
        if d["family"] == "tower" and pump_kw:
            q = dict(C.CEN, **{k: x for k, x in v.items() if k != "pads_abs"})
            elec = it["prices"]["elec"] * q["elec_mult"]
            cost += pump_kw / 1e3 * 8760.0 * elec  # $/yr per MW of design heat
        return cost, water
    return annual_cost


def main():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    res = {"multipliers": MULT, "pump_kw_per_MW": PUMP_KW_PER_MW, "menus": {}}
    for tag, cl in C.MENUS:
        items = {}
        for f in (C.OUT / "ff" / tag).glob(f"cell_*__{cl}.json"):
            it = json.loads(f.read_text(encoding="utf-8"))
            items[(it["cell"], it["climate"])] = it
        fs = pd.read_csv(C.OUT / f"ff_sites_{tag}.csv")
        phys = fs[fs.climate == cl].set_index("site_id").cphys_ban_h8.to_dict()
        key = f"{tag}|{cl}"
        res["menus"][key] = {}
        orig = C.annual_cost
        try:
            for pk in PUMP_KW_PER_MW:
                C.annual_cost = make_cost(pk)
                for m in MULT:
                    for pads in ("scaled", "abs"):
                        v = {"dc": C.CEN["dc"] * m, "pads_abs": pads == "abs"}
                        r = C.evaluate(items, sites, phys, v)
                        res["menus"][key][f"pump{pk:+g}|dc{m:g}|pads_{pads}"] = r
        finally:
            C.annual_cost = orig
        print(key, flush=True)
    (C.OUT / "cost" / "cost_c_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
