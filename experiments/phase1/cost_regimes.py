"""Regime table for C4 (after internal review R3, wording only — no new experiment): for every PR-COST-b and PR-COST-c case,
the capacity share whose least-cost water-allowed design is a cooling tower, and the ban's $/m³, grouped into three regimes —
adiabatic (tower share < 5 %), mixed transition (5–95 %), tower (≥ 95 %). Re-selects exactly as cost_reprice.evaluate (same cost
functions), adding the selected family. Writes experiments/phase1/out/cost/cost_regimes.json.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import cost_reprice as C  # noqa: E402
import cost_reprice_b as B  # noqa: E402
import cost_reprice_c as CC  # noqa: E402


def tower_share(items, sites, cost_fn, v):
    fam = {}
    for (cell, _), it in items.items():
        comp = [d for d in it["designs"] if d["compliant"]]
        costs = [cost_fn(d, it, v)[0] for d in comp]
        fam[cell] = comp[int(np.argmin(costs))]["family"]
    w = sites.B_full_MW.astype(float)
    return float(w[sites.cell_id.map(fam) == "tower"].sum() / w.sum())


def main():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    b = json.loads((C.OUT / "cost" / "cost_b_summary.json").read_text(encoding="utf-8"))
    c = json.loads((C.OUT / "cost" / "cost_c_summary.json").read_text(encoding="utf-8"))
    rows = []
    for tag, cl in C.MENUS:
        key = f"{tag}|{cl}"
        items = {}
        for f in (C.OUT / "ff" / tag).glob(f"cell_*__{cl}.json"):
            it = json.loads(f.read_text(encoding="utf-8"))
            items[(it["cell"], it["climate"])] = it
        for name, v in b["cases"].items():
            ts = tower_share(items, sites, B.annual_cost, v)
            rows.append({"menu": key, "source": "PR-COST-b", "case": name, "tower_share": ts, "USD_per_m3": b["menus"][key][name]["USD_per_m3"]})
        for name, r in c["menus"][key].items():
            pk = float(name.split("|")[0].replace("pump", ""))
            m = float(name.split("|")[1].replace("dc", ""))
            v = {"dc": C.CEN["dc"] * m, "pads_abs": name.endswith("abs")}
            ts = tower_share(items, sites, CC.make_cost(pk), v)
            rows.append({"menu": key, "source": "PR-COST-c", "case": name, "tower_share": ts, "USD_per_m3": r["USD_per_m3"]})
        print(key, flush=True)
    df = pd.DataFrame(rows)
    df["regime"] = np.where(df.tower_share < 0.05, "adiabatic", np.where(df.tower_share >= 0.95, "tower", "mixed"))
    out = {"n_cases": len(df), "regimes": {}}
    for reg, g in df.groupby("regime"):
        out["regimes"][reg] = {"n": int(len(g)), "USD_per_m3_min": float(g.USD_per_m3.min()), "USD_per_m3_max": float(g.USD_per_m3.max()),
                               "tower_share_min": float(g.tower_share.min()), "tower_share_max": float(g.tower_share.max())}
    out["cases"] = df.to_dict(orient="records")
    (C.OUT / "cost" / "cost_regimes.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["regimes"], indent=1))


if __name__ == "__main__":
    main()
