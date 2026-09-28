"""Package 3 aggregation (PR-HW): capacity-weighted F_dry(8 h) and F_nc for liquid T_TCS ∈ {40, 45, 50} °C × {f_air = 0; f_air > 0
with air-path supply-air limit T_sa ∈ {27, 32, 35} °C} × climate, on the full frame (one denominator, three bases); the built-in
45 °C check against the Phase-0 hub screen; F_dry(h) per path from h*; and the pre-registered promotion criterion for B.
Writes experiments/phase1/out/hw_summary.json.
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/phase1/out"
BASES = ("B_full_MW", "B_2027_MW", "B_now_MW")
CLIMATES = {"historical": "current", "stress_plus1.2K": "2040"}
LIQ = ("40", "45", "50")
AIR = ("27", "32", "35")
H = [0, 1, 2, 4, 8, 12, 24, 48]


def main():
    items = {}
    for f in (OUT / "hw").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        items[(r["cell"], r["climate"])] = r
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    missing = [(c, cl) for c in cells for cl in CLIMATES if (c, cl) not in items]
    res = {"n_items": len(items), "complete": not missing}
    if missing:
        keep = {c for c in cells if all((c, cl) in items for cl in CLIMATES)}
        sites = sites[sites.cell_id.isin(keep)]
        res["partial_cells"] = len(keep)
    # built-in check: liquid 45 °C = Phase-0 hub screen
    mism = []
    for (c, cl), r in items.items():
        hub = json.loads((ROOT / f"experiments/phase0/out/hub/items/{c}__{CLIMATES[cl]}.json").read_text(encoding="utf-8"))
        l45 = r["liquid"]["45"]
        if (l45["F_dry"], l45["F_nc"], l45["viol_dry"]) != (hub["F_dry"], hub["F_nc"], hub["viol_dry_kmax_tank8"]):
            mism.append(f"{c}|{cl}")
    res["check_liquid45_equals_hub_screen"] = {"PASS": not mism, "mismatches": mism}
    table = {}
    for b in BASES:
        w = sites[b].astype(float)
        W = float(w.sum())
        tb = {}
        for cl in CLIMATES:
            get = lambda path, key, T: sites.cell_id.map(lambda c: items[(c, cl)][path][T][key])  # noqa: E731
            for T in LIQ:
                ld, ln = get("liquid", "F_dry", T).astype(bool), get("liquid", "F_nc", T).astype(bool)
                tb[f"{cl}|T{T}|f_air=0"] = {"F_dry_8h": float(w[ld].sum() / W), "F_nc": float(w[ln].sum() / W)}
                for Tsa in AIR:
                    ad, an = get("air", "F_dry", Tsa).astype(bool), get("air", "F_nc", Tsa).astype(bool)
                    tb[f"{cl}|T{T}|f_air>0|Tsa{Tsa}"] = {"F_dry_8h": float(w[ld & ad].sum() / W),
                                                         "F_nc": float(w[ln & an].sum() / W)}
            for path, levels in (("liquid", LIQ), ("air", AIR)):
                for T in levels:
                    hs = get(path, "h_star", T).astype(float)
                    tb[f"{cl}|{path}{T}|F_dry_curve"] = {str(h): float(w[hs <= h + 1e-9].sum() / W) for h in H}
        table[b] = tb
    res["table"] = table
    # promotion criterion for B (PR-HW, declared before the run)
    t = table["B_full_MW"]
    W = float(sites.B_full_MW.sum())
    crit = {}
    for cl in CLIMATES:
        base = t[f"{cl}|T45|f_air=0"]["F_nc"]
        main_c = base - t[f"{cl}|T45|f_air>0|Tsa32"]["F_nc"]
        other = {"Tsa35@T45": base - t[f"{cl}|T45|f_air>0|Tsa35"]["F_nc"]}
        for T in ("40", "50"):
            other[f"Tsa32@T{T}"] = t[f"{cl}|T{T}|f_air=0"]["F_nc"] - t[f"{cl}|T{T}|f_air>0|Tsa32"]["F_nc"]
        crit[cl] = {"contrast_T45_Tsa32_points": 100 * main_c, "contrast_T45_Tsa32_GW": main_c * W / 1e3,
                    "other_contrasts_points": {k: 100 * v for k, v in other.items()},
                    "main_met": main_c >= 0.20 and main_c * W >= 7500, "others_met": all(v >= 0.10 for v in other.values())}
    res["promotion_criterion_B"] = {"by_climate": crit,
                                    "PROMOTE": all(c["main_met"] and c["others_met"] for c in crit.values())}
    (OUT / "hw_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"check": res["check_liquid45_equals_hub_screen"], "promotion": res["promotion_criterion_B"]}, indent=1))
    for k, v in t.items():
        if "curve" not in k:
            print(f"{k:40s} F_dry(8h) {v['F_dry_8h']:.3f}  F_nc {v['F_nc']:.3f}")


if __name__ == "__main__":
    main()
