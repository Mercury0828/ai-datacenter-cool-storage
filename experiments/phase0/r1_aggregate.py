"""R1 aggregation — Test A (A-phys, A-econ) and Test B exactly as pre-registered (ledger PR-R1, commit e69a14f).

Reads experiments/phase0/out/r1/items/*.json; writes experiments/phase0/out/r1/r1_results.json and r1_summary.md.
Shares are capacity-weighted over the 15 weighted sites (Luleå is a check site with weight 0), headline basis B-full,
with B-2027 and B-now alongside, and a Monte Carlo capacity interval (DD-003: lognormal, sigma = ln 1.4 / 1.2816).
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
ITEMS = ROOT / "experiments/phase0/out/r1/items"
OUT = ROOT / "experiments/phase0/out/r1"
CLIMATES = ["current", "2040"]
VARIANTS = ["headline", "hybrid", "t40", "water_constrained"]
ZONES = ["Z1", "Z2", "Z3", "Zempty"]
SIGMA = math.log(1.4) / 1.2816
RNG = np.random.default_rng(20260924)


def load():
    sites = pd.read_csv(ROOT / "data/sites/phase0_sites.csv")
    items = {}
    for p in ITEMS.glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        items[(r["site"], r["climate"], r["variant"])] = r
    return sites, items


def shares(sites, items, climate, variant, basis="B_full_MW", mult=None):
    w = sites[sites.weighted].set_index("site_id")[basis].astype(float)
    if mult is not None:
        w = w * mult
    tot = w.sum()
    out = {z: 0.0 for z in ZONES} | {"F_nc": 0.0, "F_dry": 0.0, "n_sites": 0, "denominator_MW": float(tot)}
    if tot <= 0:
        return out
    for s, wi in w.items():
        r = items[(s, climate, variant)]["site_result"]
        out[r["zone"]] += wi / tot
        out["F_nc"] += (wi / tot) if r["F_nc"] else 0.0
        out["F_dry"] += (wi / tot) if r["F_dry"] else 0.0
        out["n_sites"] += 1
    return out


def fdry_with_tank_bound(item, max_tank):
    """F_dry recomputed from stored designs with a tighter storage bound (flip-point axis of the declared 8 h bound)."""
    def ok(designs):
        return any(d["compliant"] and d["c_ch"] <= 1e-9 and not d["evap"] and d["tank_h"] <= max_tank + 1e-9
                   for d in designs)
    liq = ok(item["liquid"]["designs"])
    if "air" in item:
        return liq and ok(item["air"]["designs"])
    return liq


def main():
    sites, items = load()
    ws = sites[sites.weighted]
    expected = len(sites) * len(CLIMATES) * len(VARIANTS)
    res = {"n_items": len(items), "expected_items": expected, "shares": {}, "mc": {}, "tests": {}, "checks": {}}
    for c in CLIMATES:
        for v in VARIANTS:
            key = f"{c}|{v}"
            res["shares"][key] = {b: shares(sites, items, c, v, b) for b in ("B_full_MW", "B_2027_MW", "B_now_MW")}
            draws = []
            for _ in range(2000):
                m = np.exp(RNG.normal(0.0, SIGMA, len(ws)))
                draws.append(shares(sites, items, c, v, "B_full_MW", pd.Series(m, index=ws.site_id.values)))
            res["mc"][key] = {k: [float(np.percentile([d[k] for d in draws], q)) for q in (10, 50, 90)]
                              for k in ZONES + ["F_nc", "F_dry"]}
    # ---- Test A (headline only) and Test B (all variants), headline basis B-full
    a_phys, a_econ = {}, {}
    for c in CLIMATES:
        s = res["shares"][f"{c}|headline"]["B_full_MW"]
        a_phys[c] = {"F_nc_feasible": s["F_nc"], "F_nc_infeasible": 1 - s["F_nc"], "F_dry": s["F_dry"],
                     "pass": bool(s["F_nc"] >= 0.10 and (1 - s["F_nc"]) >= 0.10)}
        a_econ[c] = {"Z1+Z2": s["Z1"] + s["Z2"], "Z3+Zempty": s["Z3"] + s["Zempty"],
                     "pass": bool((s["Z1"] + s["Z2"]) >= 0.10 and (s["Z3"] + s["Zempty"]) >= 0.10)}
    test_b = {}
    for c in CLIMATES:
        for v in VARIANTS:
            s = res["shares"][f"{c}|{v}"]["B_full_MW"]
            m = max(s[z] for z in ZONES)
            test_b[f"{c}|{v}"] = {"1-max_z": 1 - m, "argmax": max(ZONES, key=lambda z: s[z]), "pass": bool((1 - m) >= 0.10)}
    A_phys = any(x["pass"] for x in a_phys.values())
    A_econ = any(x["pass"] for x in a_econ.values())
    B = any(x["pass"] for x in test_b.values())
    outcome = ("(i) A-phys passes -> proceed" if A_phys else
               "(ii-a) A-phys fails, A-econ or Test B passes -> alternative contribution; ladder; owner at gate (a)"
               if (A_econ or B) else "(ii-b) all fail -> negative; ladder")
    res["tests"] = {"A_phys": a_phys, "A_econ": a_econ, "B": test_b, "A_phys_pass": A_phys, "A_econ_pass": A_econ,
                    "B_pass": B, "outcome": outcome}
    # ---- expectation checks
    lu = items.get(("lulea", "current", "headline"))
    gy = items.get(("microsoft_goodyear", "current", "water_constrained"))
    res["checks"] = {
        "lulea_current_headline": lu["site_result"] if lu else None,
        "goodyear_current_water_constrained": gy["site_result"] if gy else None,
        "goodyear_2040_water_constrained": items[("microsoft_goodyear", "2040", "water_constrained")]["site_result"],
    }
    # ---- F_dry sensitivity to the storage bound (headline, current & 2040)
    sens = {}
    for c in CLIMATES:
        for tb in (0.0, 2.0, 4.0, 8.0):
            w = ws.set_index("site_id")["B_full_MW"]
            sens[f"{c}|tank<={tb}h"] = float(sum(w[s] for s in w.index if fdry_with_tank_bound(items[(s, c, "headline")], tb)) / w.sum())
    res["F_dry_vs_tank_bound"] = sens
    # ---- per-site table and a first 'who pays' scale (headline least-cost designs, B-full MW)
    rows = []
    for _, r in sites.iterrows():
        for c in CLIMATES:
            for v in VARIANTS:
                it = items[(r.site_id, c, v)]
                b = it["liquid"]["best"]
                rows.append({"site": r.site_id, "w_full": r.B_full_MW, "climate": c, "variant": v,
                             "zone": it["site_result"]["zone"], "F_nc": it["site_result"]["F_nc"],
                             "F_dry": it["site_result"]["F_dry"], "cost_per_MW_yr": it["site_result"]["cost"],
                             "liq_family": b["family"] if b else None, "liq_k": b["k"] if b else None,
                             "liq_ad": b["ad"] if b else None, "liq_tank_h": b["tank_h"] if b else None,
                             "liq_c_ch": b["c_ch"] if b else None, "liq_kct": b["kct"] if b else None,
                             "water_m3_per_MW_yr": b.get("ox_water_m3") if b else None,
                             "elec_MWh_per_MW_yr": b.get("ox_elec_MWh") if b else None,
                             "peak_MW_per_MW": b.get("ox_peak_MW") if b else None,
                             "air_zone": it["air"]["zone"] if "air" in it else None})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r1_sites.csv", index=False)
    scale = {}
    for c in CLIMATES:
        for v in ("headline", "water_constrained"):
            d = df[(df.climate == c) & (df.variant == v) & (df.w_full > 0)]
            scale[f"{c}|{v}"] = {"sample_GW": float(d.w_full.sum() / 1000),
                                 "water_GL_per_yr": float((d.w_full * d.water_m3_per_MW_yr.fillna(0)).sum() / 1e6),
                                 "cooling_TWh_per_yr": float((d.w_full * d.elec_MWh_per_MW_yr.fillna(0)).sum() / 1e6),
                                 "peak_cooling_MW": float((d.w_full * d.peak_MW_per_MW.fillna(0)).sum()),
                                 "trim_chiller_MW": float((d.w_full * d.liq_c_ch.fillna(0)).sum())}
    res["scale"] = scale
    (OUT / "r1_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    # ---- markdown summary
    L = [f"# R1 results (items {len(items)}/{expected})", "", "Capacity shares, basis B-full (MC 10-90 % in brackets):", "",
         "| climate | variant | Z1 | Z2 | Z3 | Z∅ | F_nc | F_dry |", "|---|---|---|---|---|---|---|---|"]
    for c in CLIMATES:
        for v in VARIANTS:
            s = res["shares"][f"{c}|{v}"]["B_full_MW"]
            mc = res["mc"][f"{c}|{v}"]
            cell = lambda k: f"{s[k]:.2f} [{mc[k][0]:.2f}–{mc[k][2]:.2f}]"
            L.append(f"| {c} | {v} | {cell('Z1')} | {cell('Z2')} | {cell('Z3')} | {cell('Zempty')} | {cell('F_nc')} | {cell('F_dry')} |")
    L += ["", f"**Test A-phys:** {a_phys}", "", f"**Test A-econ:** {a_econ}", "",
          f"**Test B:** {json.dumps(test_b, default=float)}", "", f"**Outcome:** {outcome}", "",
          f"**F_dry vs storage bound (headline):** {sens}", "", f"**Checks:** {res['checks']}", "",
          f"**Scale (15-site sample, B-full):** {json.dumps(scale, indent=1, default=float)}"]
    (OUT / "r1_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
