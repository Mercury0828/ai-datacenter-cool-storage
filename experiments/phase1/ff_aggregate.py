"""Package 2 aggregation (PR-FF): least-cost routes with / without water, physical minimum compressor capacity under storage and
water budgets, physical-vs-economic split, resource cost of eliminating compressors, scale (GW_th / GW_e / TWh / GL kept
separate), mixed-mode comparator, flip points, near-ties, violation severity — capacity-weighted on the full frame.
Reads experiments/phase1/out/ff/<tag>/*.json (one per cell × climate). Writes experiments/phase1/out/ff_summary_<tag>.json and
ff_sites_<tag>.csv.
Usage: python experiments/phase1/ff_aggregate.py [--tag uncorrected]
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/phase1/out"
BASES = ("B_full_MW", "B_2027_MW", "B_now_MW")
CLIMATES = ("historical", "stress_plus1.2K")
WATER_MIN = f"{1e9:.4g}"
SWEEPS = {"tank_kUSD_per_MWh": [10, 17, 25, 35, 50, 70, 100], "h_max": [0, 2, 4, 8],
          "water_mult": [0, 0.25, 0.5, 1, 2, 3, 5, 10, 20], "elec_mult": [0.5, 0.625, 1, 1.5, 2],
          "demand_USD_per_kW_month": [0, 10, 20]}
PLAUSIBLE = {"tank_kUSD_per_MWh": (17, 70), "water_mult": (0.25, 3), "elec_mult": (0.625, 1.5)}  # params sensitivity ranges


def zone(d):
    return "Z3" if d["c_ch"] > 1e-9 else ("Z2" if (d["ad"] or d["family"] == "tower") else "Z1")


def ratio_key(item, pr):
    mult = (pr["water"] / pr["elec"]) / (item["prices"]["water"] / item["prices"]["elec"])
    keys = {float(k): k for k in (f"{x:.4g}" for x in item["ratio_mult"])}
    best = min(keys, key=lambda v: abs(v - mult))
    assert abs(best - mult) <= 1e-3 * max(1.0, mult), (mult, best)
    return keys[best]


def ox_of(d, rk):
    return d["ox_ratio"][rk] if (d["ad"] and rk is not None) else d["ox"]


def cost(d, pr, rk):
    c = d["capex"]
    capex = c["dc"] + c["ad"] + c["tank_MWh"] * pr["tank"] + c["chiller"] + c["tower"]
    ox = ox_of(d, rk)
    return (pr["crf"] * capex + ox["elec_MWh"] * pr["elec"] + ox["water_m3"] * pr["water"]
            + pr.get("demand", 0.0) * 12.0 * 1000.0 * ox["peak_MW_mean"])


def best(designs, pr, rk, ok=lambda d: True):
    cand = [d for d in designs if d["compliant"] and ok(d)]
    if not cand:
        return None, None
    d = min(cand, key=lambda x: cost(x, pr, rk))
    return d, cost(d, pr, rk)


def base_prices(item):
    p = item["prices"]
    return {"elec": p["elec"], "water": p["water"], "tank": p["tank_per_MWh"], "crf": p["crf"], "demand": 0.0}


def is_ban(d):
    return d["family"] == "dry" and not d["ad"]


def c_phys(designs, h, water_cap, peak_cap, sf_rows=None):
    """Smallest risk-sized chiller over compliant designs with tank <= h and water within caps. Adiabatic designs are admitted
    under the pads-first schedule (registered) and, after internal review full-frame-consequences R1, also under the storage-first causal
    witness policy (`sf_rows`, ff_water_frontier.py). Witness-based: an upper bound on the physical minimum; exact for zero water."""
    vals = []
    for r in sf_rows or []:
        if r["compliant"] and r["tank_h"] <= h + 1e-9 and r["water_m3_mean"] <= water_cap + 1e-9 and r["water_peak_m3h"] <= peak_cap + 1e-12:
            vals.append(r["c_ch"])
    for d in designs:
        if not d["compliant"] or d["tank_h"] > h + 1e-9:
            continue
        if d["family"] == "tower":
            w, wp = d["ox"]["water_m3"], d.get("tower_peak_water_m3h", np.inf)
        elif d["ad"]:
            o = d["ox_ratio"][WATER_MIN]
            w, wp = o["water_m3"], o.get("water_peak_m3h_max", np.inf)
        else:
            w, wp = 0.0, 0.0
        if w <= water_cap + 1e-9 and wp <= peak_cap + 1e-12:
            vals.append(d["c_ch"])
    return min(vals) if vals else np.nan


def site_rows(items, sites):
    rows = []
    for (cell, cl), it in items.items():
        ds = it["designs"]
        pr = base_prices(it)
        rk = ratio_key(it, pr)
        lc, lc_c = best(ds, pr, rk)
        ban, ban_c = best(ds, pr, rk, is_ban)
        cf, cf_c = best(ds, pr, rk, lambda d: d["c_ch"] <= 1e-9)
        cfd, cfd_c = best(ds, pr, rk, lambda d: d["c_ch"] <= 1e-9 and is_ban(d))
        # E5 "compressor-free and water-free where feasible plus C_phys elsewhere" (registered; completion added after internal review
        # full-frame R1 note F5): where no water-free compressor-free design exists, the least-cost water-free design whose
        # chiller equals the zero-water physical minimum (store <= 8 h)
        if cfd is None:
            cp0 = c_phys(ds, 8.0, 0.0, 0.0)
            cfc, cfc_c = best(ds, pr, rk, lambda d, c=cp0: is_ban(d) and d["tank_h"] <= 8.0 + 1e-9 and d["c_ch"] <= c + 1e-9)
        else:
            cfc, cfc_c = cfd, cfd_c
        tw, tw_c = best(ds, pr, rk, lambda d: d["family"] == "tower")
        mm1, mm1_c = best(ds, pr, rk, lambda d: is_ban(d) and d["c_ch"] <= 0.1 + 1e-9)
        mm2, mm2_c = best(ds, pr, rk, lambda d: is_ban(d) and d["c_ch"] <= 0.2 + 1e-9)
        w_tower = tw["ox"]["water_m3"] if tw else np.inf
        wp_tower = tw.get("tower_peak_water_m3h", np.inf) if tw else np.inf
        r = {"cell": cell, "climate": cl, "lc_cost": lc_c, "lc_zone": zone(lc) if lc else "Zempty",
             "lc_c_ch": lc["c_ch"] if lc else np.nan, "lc_design": design_str(lc),
             "ban_cost": ban_c, "ban_zone": zone(ban) if ban else "Zempty", "ban_c_ch": ban["c_ch"] if ban else np.nan,
             "ban_design": design_str(ban), "cf_cost": cf_c, "cf_design": design_str(cf), "cfdry_cost": cfd_c,
             "cfdry_design": design_str(cfd), "cfdrycomplete_cost": cfc_c, "cfdrycomplete_design": design_str(cfc), "tower_cost": tw_c, "tower_design": design_str(tw), "mm01_cost": mm1_c,
             "mm02_cost": mm2_c, "tower_water_m3": w_tower}
        for tag, d in (("lc", lc), ("ban", ban), ("cf", cf), ("cfdry", cfd), ("cfdrycomplete", cfc), ("tower", tw)):
            o = ox_of(d, rk) if d else None
            for q in ("elec_MWh", "water_m3", "peak_MW_mean", "ch_MWh"):
                r[f"{tag}_{q}"] = o[q] if o else np.nan
            r[f"{tag}_tank_h"] = d["tank_h"] if d else np.nan
            r[f"{tag}_c_ch"] = d["c_ch"] if d else np.nan
            r[f"{tag}_family"] = d["family"] if d else None
            r[f"{tag}_k"] = d["k"] if d else np.nan
            r[f"{tag}_ad"] = d["ad"] if d else None
            r[f"{tag}_kct"] = d["kct"] if d else np.nan
            r[f"{tag}_viol_unmet_MWh"] = sum(s["unmet_MWh"] for s in d.get("severity", [])) if d else np.nan
            r[f"{tag}_viol_hours"] = sum(s["hours"] for s in d.get("severity", [])) if d else np.nan
        wf = OUT / "ff_water" / it.get("_tag", "") / f"{cell}__{cl}.json"
        sf = json.loads(wf.read_text(encoding="utf-8"))["designs"] if wf.exists() else None
        r["storage_first_witnesses"] = sf is not None
        for h in (0, 2, 4, 8):
            r[f"cphys_ban_h{h}"] = c_phys(ds, h, 0.0, 0.0)
            for frac in (0.01, 0.1, 1.0):
                r[f"cphys_w{frac:g}_h{h}"] = c_phys(ds, h, frac * w_tower, np.inf, sf)
            r[f"cphys_peak50_h{h}"] = c_phys(ds, h, np.inf, 0.5 * wp_tower, sf)
            r[f"cphys_free_h{h}"] = c_phys(ds, h, np.inf, np.inf, sf)
        # near-ties: best design in another zone
        if lc:
            alt = [cost(d, pr, rk) for d in ds if d["compliant"] and zone(d) != zone(lc)]
            r["near_tie_gap"] = (min(alt) - lc_c) / lc_c if alt else np.inf
        rows.append(r)
    df = pd.DataFrame(rows)
    return sites.merge(df, left_on="cell_id", right_on="cell", how="left")


def design_str(d):
    if d is None:
        return None
    if d["family"] == "tower":
        return f"tower kct={d['kct']:g} C={d['c_ch']:.3f}"
    return f"dry k={d['k']:.4f}{' +ad' if d['ad'] else ''} tank={d['tank_h']:g}h C={d['c_ch']:.3f}"


def fleet(df, basis):
    out = {}
    for cl in CLIMATES:
        g = df[df.climate == cl]
        w = g[basis].astype(float)
        W = float(w.sum())
        r = {"denominator_MW": W}
        for pre in ("lc", "ban"):
            z = g[f"{pre}_zone"]
            r[f"{pre}_zone_share"] = {Z: float(w[z == Z].sum() / W) for Z in ("Z1", "Z2", "Z3", "Zempty")}
        for pre in ("lc", "ban", "cf", "cfdry", "cfdrycomplete", "tower"):
            ok = g[f"{pre}_elec_MWh"].notna()
            r[f"{pre}_scale"] = {"covered_MW": float(w[ok].sum()),
                                 "chiller_GW_th": float((w * g[f"{pre}_c_ch"]).sum() / 1e3),
                                 "peak_GW_e": float((w * g[f"{pre}_peak_MW_mean"]).sum() / 1e3),
                                 "elec_TWh_yr": float((w * g[f"{pre}_elec_MWh"]).sum() / 1e6),
                                 "water_GL_yr": float((w * g[f"{pre}_water_m3"]).sum() / 1e6),
                                 "tank_GWh_th": float((w * g[f"{pre}_tank_h"]).sum() / 1e3),
                                 "viol_unmet_GWh_in_violating_years": float((w * g[f"{pre}_viol_unmet_MWh"]).sum() / 1e3),
                                 "cost_BUSD_yr": float((w * g[f"{pre}_cost"]).sum() / 1e9) if f"{pre}_cost" in g else None}
        # physical minimum compressor capacity (GW_th) under storage and water budgets
        r["cphys_GW_th"] = {c: float((w * g[c]).sum() / 1e3) for c in g.columns if c.startswith("cphys_")}
        r["cphys_MW_with_C_gt_0"] = {c: float(w[g[c] > 1e-9].sum()) for c in g.columns if c.startswith("cphys_")}
        # physical necessity vs economic selection under a water ban (8 h storage)
        phys = g["cphys_ban_h8"].fillna(0.0)
        r["ban_split_GW_th"] = {"installed_least_cost": float((w * g.ban_c_ch).sum() / 1e3),
                                "physically_necessary": float((w * phys).sum() / 1e3),
                                "economic": float((w * (g.ban_c_ch - phys).clip(lower=0)).sum() / 1e3),
                                "MW_economic_only": float(w[(g.ban_c_ch > 1e-9) & (phys <= 1e-9)].sum())}
        r["water_allowed_split_GW_th"] = {"installed_least_cost": float((w * g.lc_c_ch).sum() / 1e3),
                                          "physically_necessary": float((w * g["cphys_free_h8"].fillna(0)).sum() / 1e3)}
        # resource cost of eliminating compressors (sites whose least-cost design has C > 0)
        z3 = g.lc_c_ch > 1e-9
        r["eliminate_compressors_with_water"] = {
            "MW": float(w[z3].sum()), "d_cost_MUSD_yr": float((w * (g.cf_cost - g.lc_cost))[z3].sum() / 1e6),
            "d_water_GL_yr": float((w * (g.cf_water_m3 - g.lc_water_m3))[z3].sum() / 1e6),
            "d_elec_TWh_yr": float((w * (g.cf_elec_MWh - g.lc_elec_MWh))[z3].sum() / 1e6),
            "d_peak_GW_e": float((w * (g.cf_peak_MW_mean - g.lc_peak_MW_mean))[z3].sum() / 1e3)}
        zb = (g.ban_c_ch > 1e-9) & g.cfdry_cost.notna()
        r["eliminate_compressors_water_ban"] = {
            "MW_feasible": float(w[zb].sum()), "MW_infeasible": float(w[(g.ban_c_ch > 1e-9) & g.cfdry_cost.isna()].sum()),
            "d_cost_MUSD_yr": float((w * (g.cfdry_cost - g.ban_cost))[zb].sum() / 1e6),
            "d_tank_GWh_th": float((w * (g.cfdry_tank_h - g.ban_tank_h))[zb].sum() / 1e3),
            "d_elec_TWh_yr": float((w * (g.cfdry_elec_MWh - g.ban_elec_MWh))[zb].sum() / 1e6)}
        r["mixed_mode"] = {"MW_feasible_C<=0.1": float(w[g.mm01_cost.notna()].sum()),
                           "MW_feasible_C<=0.2": float(w[g.mm02_cost.notna()].sum()),
                           "MW_mm01_cheaper_than_tower": float(w[(g.mm01_cost < g.tower_cost)].sum())}
        r["near_ties_MW"] = {"2%": float(w[g.near_tie_gap <= 0.02].sum()), "5%": float(w[g.near_tie_gap <= 0.05].sum())}
        out[cl] = r
    return out


def flips(items, sites, basis="B_full_MW"):
    """Zone shares (water allowed and banned) along one-at-a-time price / allowance sweeps."""
    res = {}
    for axis, levels in SWEEPS.items():
        res[axis] = {}
        for lv in levels:
            per = {}
            for (cell, cl), it in items.items():
                pr = base_prices(it)
                hmax = 8.0
                if axis == "tank_kUSD_per_MWh":
                    pr["tank"] = lv * 1e3
                elif axis == "h_max":
                    hmax = float(lv)
                elif axis == "water_mult":
                    pr["water"] *= lv
                elif axis == "elec_mult":
                    pr["elec"] *= lv
                else:
                    pr["demand"] = float(lv)
                rk = ratio_key(it, pr)
                ok_h = (lambda d, hm=hmax: d["tank_h"] <= hm + 1e-9)
                lc, _ = best(it["designs"], pr, rk, ok_h)
                bn, _ = best(it["designs"], pr, rk, lambda d, hm=hmax: is_ban(d) and d["tank_h"] <= hm + 1e-9)
                per[(cell, cl)] = (zone(lc) if lc else "Zempty", zone(bn) if bn else "Zempty")
            lvres = {}
            for cl in CLIMATES:
                s = sites.copy()
                s["lc"] = s.cell_id.map(lambda c: per[(c, cl)][0])
                s["bn"] = s.cell_id.map(lambda c: per[(c, cl)][1])
                w = s[basis].astype(float)
                lvres[cl] = {"lc": {Z: float(w[s.lc == Z].sum() / w.sum()) for Z in ("Z1", "Z2", "Z3", "Zempty")},
                             "ban": {Z: float(w[s.bn == Z].sum() / w.sum()) for Z in ("Z1", "Z3", "Zempty")}}
            lvres["_per_site"] = {f"{c}|{cl}": v for (c, cl), v in per.items()}
            res[axis][str(lv)] = lvres
        if axis in PLAUSIBLE:
            lo, hi = (str(x) for x in PLAUSIBLE[axis])
            moved = {}
            for cl in CLIMATES:
                a, b = res[axis][lo]["_per_site"], res[axis][hi]["_per_site"]
                w = sites.set_index("site_id")[basis].astype(float)
                cellw = sites.groupby("cell_id")[basis].sum()
                for j, lab in ((0, "water_allowed"), (1, "water_banned")):
                    ch = [c for c in cellw.index if a[f"{c}|{cl}"][j] != b[f"{c}|{cl}"][j]]
                    moved[f"{cl}|{lab}"] = {"MW": float(cellw[ch].sum()), "cells": ch}
            res[axis]["moved_between_plausible_bounds"] = moved
    return res


def coincident(tag, sites, basis="B_full_MW"):
    """E6: fleet cooling-electric peak, hour by hour (UTC-aligned), vs the sum of site peaks, for the base-price selections."""
    d = pathlib.Path("D:/_caches/hotloop/ff_hourly") / tag
    if not d.exists():
        return None
    cw = sites.groupby("cell_id")[basis].sum()
    res = {}
    for cl in CLIMATES:
        for key in ("lc", "ban", "cf", "cfdry", "tower"):
            fleet_sum, peak_sum, n = None, None, 0
            for cell, w in cw.items():
                f = d / f"{cell}__{cl}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                if f"{key}_elec" not in z.files:
                    continue
                e = z[f"{key}_elec"].astype(np.float64) * w
                fleet_sum = e if fleet_sum is None else fleet_sum + e
                pk = e.max(axis=1)
                peak_sum = pk if peak_sum is None else peak_sum + pk
                n += 1
            if fleet_sum is None:
                continue
            coin = fleet_sum.max(axis=1)
            res[f"{cl}|{key}"] = {"cells": n, "coincident_peak_GW_e_mean": float(coin.mean() / 1e3),
                                  "coincident_peak_GW_e_max": float(coin.max() / 1e3),
                                  "sum_of_site_peaks_GW_e_mean": float(peak_sum.mean() / 1e3),
                                  "ratio_coincident_to_sum_mean": float((coin / peak_sum).mean()),
                                  "annual_TWh_mean": float(fleet_sum.sum(axis=1).mean() / 1e6)}
    return res


def main():
    global CLIMATES
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="uncorrected")
    ap.add_argument("--climates", default=",".join(CLIMATES), help="comma-separated climate labels (DD-016: a PR-CMIP scenario)")
    a = ap.parse_args()
    CLIMATES = tuple(a.climates.split(","))
    items = {}
    for f in (OUT / "ff" / a.tag).glob("*.json"):
        it = json.loads(f.read_text(encoding="utf-8"))
        it["_tag"] = a.tag
        items[(it["cell"], it["climate"])] = it
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    missing = [(c, cl) for c in cells for cl in CLIMATES if (c, cl) not in items]
    if missing:
        print(f"{len(missing)} items missing; aggregating the complete cells only")
        keep = {c for c in cells if all((c, cl) in items for cl in CLIMATES)}
        sites = sites[sites.cell_id.isin(keep)]
        items = {k: v for k, v in items.items() if k[0] in keep}
    df = site_rows(items, sites)
    res = {"tag": a.tag, "n_items": len(items), "complete": not missing,
           "fleet": {b: fleet(df, b) for b in BASES}, "flips|B_full_MW": flips(items, sites),
           "coincident|B_full_MW": coincident(a.tag, sites)}
    (OUT / f"ff_summary_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    df.to_csv(OUT / f"ff_sites_{a.tag}.csv", index=False)
    print(json.dumps(res["fleet"]["B_full_MW"], indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
