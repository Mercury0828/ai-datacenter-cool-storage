"""PR-SB SB-5 (ledger PR-SB, declared before any run): the mechanism behind the ban's $/m3. For every repriced case (PR-COST sourced
variants at the central dry-cooler price; PR-COST-b and -c cases with their cost_regimes labels) and climate: numerator = annualized
incremental cost of the water ban (M$/yr), denominator = evaporative water avoided (GL/yr, the water-allowed optimum's use), ratio
= $/m3. Summarized per regime (adiabatic / mixed / tower) and climate: ranges of numerator, denominator and ratio.
Writes experiments/phase1/out/sb/sb5_mechanism.json. Usage: python experiments/phase1/sb_mechanism.py
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
COST = ROOT / "experiments/phase1/out/cost"
OUT = ROOT / "experiments/phase1/out/sb"


def main():
    reg = json.loads((COST / "cost_regimes.json").read_text(encoding="utf-8"))
    label = {(c["menu"], c["source"], c["case"]): (c["regime"], c["tower_share"]) for c in reg["cases"]}
    rows = []
    a = json.loads((COST / "cost_summary.json").read_text(encoding="utf-8"))
    for menu, m in a["menus"].items():
        for case, v in m["variants"].items():  # sourced variants at the central dry-cooler price: adiabatic regime (C4)
            rows.append((menu, "PR-COST", case, "adiabatic", None, v["ban_added_cost_MUSD_yr"], v["water_saved_GL_yr"], v["USD_per_m3"]))
    for src, fn in (("PR-COST-b", "cost_b_summary.json"), ("PR-COST-c", "cost_c_summary.json")):
        b = json.loads((COST / fn).read_text(encoding="utf-8"))
        for menu, m in b["menus"].items():
            for case, v in m.items():
                if not isinstance(v, dict) or "USD_per_m3" not in v:
                    continue
                r, ts = label[(menu, src, case)]
                rows.append((menu, src, case, r, ts, v["ban_added_cost_MUSD_yr"], v["water_saved_GL_yr"], v["USD_per_m3"]))
    # consistency: ratio = numerator / denominator ($ M / GL = $ / m3 x 1e6 / 1e6)
    worst = max(abs(num / den - usd) / usd for *_, num, den, usd in rows)
    res = {"n_cases": len(rows), "max_rel_check_ratio": worst, "by_regime_climate": {}}
    for regime in ("adiabatic", "mixed", "tower"):
        for menu in sorted({r[0] for r in rows}):
            sel = [r for r in rows if r[3] == regime and r[0] == menu]
            if not sel:
                continue
            num = np.array([r[5] for r in sel])
            den = np.array([r[6] for r in sel])
            usd = np.array([r[7] for r in sel])
            res["by_regime_climate"][f"{regime}|{menu}"] = {
                "n": len(sel), "ban_added_cost_MUSD_yr": [float(num.min()), float(np.median(num)), float(num.max())],
                "water_saved_GL_yr": [float(den.min()), float(np.median(den)), float(den.max())],
                "USD_per_m3": [float(usd.min()), float(np.median(usd)), float(usd.max())]}
    res["cases"] = [dict(zip(("menu", "source", "case", "regime", "tower_share", "ban_added_cost_MUSD_yr", "water_saved_GL_yr",
                              "USD_per_m3"), r)) for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sb5_mechanism.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("cases", len(rows), "ratio check", worst)
    for k, v in res["by_regime_climate"].items():
        print(k, "n", v["n"], "cost M$/yr", [round(x) for x in v["ban_added_cost_MUSD_yr"]], "water GL/yr",
              [round(x, 1) for x in v["water_saved_GL_yr"]], "$/m3", [round(x, 1) for x in v["USD_per_m3"]])


if __name__ == "__main__":
    main()
