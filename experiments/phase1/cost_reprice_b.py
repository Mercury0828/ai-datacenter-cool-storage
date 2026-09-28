"""PR-COST-b (ledger 2026-09-26, declared after the PR-COST result, before any run): the dry-cooler flip point.
1. Pads decoupled: dry-cooler multipliers {0.5, 1.5, 2, 2.5, 3, 4, 6, 10} with the pad cost held at its central absolute value.
2. Fine grid {2.0, 2.25, 2.5, 2.75, 3.0} in both pad treatments (scaled with the dry cooler, as PR-COST; decoupled).
3. NETL-anchored cases: module cost 2.0x and 3.0x combined with the joint sourced-low and joint sourced-high inputs.
Writes experiments/phase1/out/cost/cost_b_summary.json.
Usage: python experiments/phase1/cost_reprice_b.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import cost_reprice as C  # noqa: E402
import ff_aggregate as A  # noqa: E402

DECOUPLED = [0.5, 1.5, 2, 2.5, 3, 4, 6, 10]
ORIG = C.annual_cost  # PR-COST's function, kept before evaluate() is pointed at the PR-COST-b version
FINE = [2.0, 2.25, 2.5, 2.75, 3.0]


def annual_cost(d, it, v):
    """As cost_reprice.annual_cost, with an optional pad cost held at its central absolute value (v['pads_abs'])."""
    if not v.get("pads_abs"):
        return ORIG(d, it, {k: x for k, x in v.items() if k != "pads_abs"})
    q = dict(C.CEN, **{k: x for k, x in v.items() if k != "pads_abs"})
    pr = A.base_prices(it)
    pr["elec"] = pr["elec"] * q["elec_mult"]
    rk = A.ratio_key(it, pr)
    c = d["capex"]
    dc = c["dc"] * q["dc"] / C.CEN["dc"]
    ad = c["ad"] * q["ad_frac"] / C.CEN["ad_frac"]  # not scaled with the dry-cooler price
    tank = c["tank_MWh"] * q["tank"]
    ch = c["chiller"] * q["chiller"] / C.CEN["chiller"]
    tw = c["tower"] * q["tower"] / C.CEN["tower"]
    ox = A.ox_of(d, rk)
    cost = q["crf"] * (dc + ad + tank + ch + tw) + ox["elec_MWh"] * pr["elec"] + ox["water_m3"] * pr["water"]
    if q["om"]:
        cost += 0.024 * ch + 0.02 * (dc + ad + tank + tw)
    return cost, ox["water_m3"]


def main():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    low = C.VARIANTS["joint_sourced_low"]
    high = C.VARIANTS["joint_sourced_high"]
    cases = {}
    for m in DECOUPLED:
        cases[f"dc{m:g}_pads_abs"] = {"dc": C.CEN["dc"] * m, "pads_abs": True}
    for m in FINE:
        cases[f"dc{m:g}_pads_scaled"] = {"dc": C.CEN["dc"] * m}
        cases[f"dc{m:g}_pads_abs"] = {"dc": C.CEN["dc"] * m, "pads_abs": True}
    for m in (2.0, 3.0):
        for tag, j in (("low", low), ("high", high)):
            cases[f"netl_dc{m:g}_joint_{tag}_pads_scaled"] = dict(j, dc=C.CEN["dc"] * m)
            cases[f"netl_dc{m:g}_joint_{tag}_pads_abs"] = dict(j, dc=C.CEN["dc"] * m, pads_abs=True)
    res = {"cases": cases, "menus": {}}
    for tag, cl in C.MENUS:
        items = {}
        for f in (C.OUT / "ff" / tag).glob(f"cell_*__{cl}.json"):
            it = json.loads(f.read_text(encoding="utf-8"))
            items[(it["cell"], it["climate"])] = it
        fs = pd.read_csv(C.OUT / f"ff_sites_{tag}.csv")
        phys = fs[fs.climate == cl].set_index("site_id").cphys_ban_h8.to_dict()
        key = f"{tag}|{cl}"
        res["menus"][key] = {}
        C.annual_cost = annual_cost  # evaluate() calls C.annual_cost
        try:
            for name, v in cases.items():
                res["menus"][key][name] = C.evaluate(items, sites, phys, v)
        finally:
            C.annual_cost = ORIG
        r = res["menus"][key]
        print(key, {n: round(r[n]["USD_per_m3"], 2) for n in cases}, flush=True)
    (C.OUT / "cost" / "cost_b_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
