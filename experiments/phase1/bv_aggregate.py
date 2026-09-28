"""Aggregate package 1 (boundary validity, ledger PR-BV; DD-012/DD-013) on the full Epoch frame (93 sites, one denominator).

Per weather tag (uncorrected, corrected): capacity-weighted F_dry(h) from BV-1 h*(k_max) for h in [0, 48] h; the built-in
validation at h = 8 h against the hub screen of the same weather (per cell, pass/fail); the h*(k) storage–dry-cooler trade-off;
tank volume / footprint / cost; critical-episode diagnostics; BV-2 (charge swing, utilization 0.9, continuous-state check).
Across tags (BV-3): F_dry(8 h), h* and F_nc under (i) uncorrected, (ii) corrected, (iii) corrected + 1.2 K, and GW changing
category. Writes experiments/phase1/out/bv_summary.json and bv_sites.csv.
Usage: python experiments/phase1/bv_aggregate.py
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/phase1/out"
BASES = ("B_full_MW", "B_2027_MW", "B_now_MW")
CLIM = {"historical": "current", "stress_plus1.2K": "2040"}  # BV-1 label -> Phase-0 hub-screen label
HUB = {"uncorrected": ROOT / "experiments/phase0/out/hub/items", "corrected": OUT / "bv3/hub_corrected/items"}
H_GRID = [float(h) for h in np.arange(0.0, 48.001, 0.5)]
TOL = 1e-9
M3_PER_MWH = 3.6e9 / (4.186e6 * 10.0)  # 86.0 m3 per MWh_th at a 10 K swing (PR-BV)
TANK_HEIGHT_M = 20.0
COST_PER_MWH = 35e3


def load_items(tag):
    d = OUT / "bv1" / tag
    items = {}
    for f in d.glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        items[(r["cell"], r["climate"])] = r
    return items


def load_hub(tag):
    d = HUB[tag]
    res = {}
    if d.exists():
        for f in d.glob("*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            res[(r["cell"], r["climate"])] = r
    return res


def wfrac(mask, w):
    return float((w * mask).sum() / w.sum())


def summarize_tag(tag, sites, cells):
    items = load_items(tag)
    hub = load_hub(tag)
    out = {"n_items": len(items), "complete": len(items) == 2 * len(cells)}
    per_site = {}
    for cl, hub_cl in CLIM.items():
        have = [c for c in cells if (c, cl) in items]
        if len(have) < len(cells):
            out[cl] = {"missing_cells": len(cells) - len(have)}
            continue
        ks = sorted(items[(have[0], cl)]["k"], key=float)
        kmax = ks[-1]
        hs = {c: items[(c, cl)]["k"][kmax]["h_star"] for c in cells}
        s = sites.copy()
        s["h_star"] = s.cell_id.map(hs)
        per_site[cl] = s["h_star"]
        rec = {"k_max": float(kmax), "F_dry_curve": {}}
        for b in BASES:
            w = s[b].astype(float)
            rec["F_dry_curve"][b] = [wfrac((s.h_star <= h + TOL).astype(float), w) for h in H_GRID]
            rec[f"F_dry_8h|{b}"] = wfrac((s.h_star <= 8 + TOL).astype(float), w)
            rec[f"F_dry_8h_MW|{b}"] = float((w * (s.h_star <= 8 + TOL)).sum())
            rec[f"F_dry_0h|{b}"] = wfrac((s.h_star <= TOL).astype(float), w)
            rec[f"F_dry_24h|{b}"] = wfrac((s.h_star <= 24 + TOL).astype(float), w)
            rec[f"F_dry_48h|{b}"] = wfrac((s.h_star <= 48 + TOL).astype(float), w)
            rec[f"denominator_MW|{b}"] = float(w.sum())
        # built-in validation vs the hub screen of the same weather (per cell)
        if all((c, hub_cl) in hub for c in cells):
            mism = [{"cell": c, "h_star": hs[c], "hub_F_dry": hub[(c, hub_cl)]["F_dry"],
                     "hub_viol_dry": hub[(c, hub_cl)]["viol_dry_kmax_tank8"]}
                    for c in cells if (hs[c] <= 8 + TOL) != bool(hub[(c, hub_cl)]["F_dry"])]
            rec["validation_vs_hub_screen_8h"] = {"PASS": not mism, "mismatches": mism}
            rec["F_nc_hub"] = {b: wfrac(s.cell_id.map(lambda c: hub[(c, hub_cl)]["F_nc"]).astype(float), s[b].astype(float))
                               for b in BASES}
            s["F_nc"] = s.cell_id.map(lambda c: bool(hub[(c, hub_cl)]["F_nc"]))
            per_site[cl + "|F_nc"] = s["F_nc"]
        else:
            rec["validation_vs_hub_screen_8h"] = "hub screen not available for this weather"
        # BV-4 sanity: h*(k) must be non-increasing in k (need falls and spare rises with k)
        nonmono = [c for c in cells if any(items[(c, cl)]["k"][ks[i + 1]]["h_star"] > items[(c, cl)]["k"][ks[i]]["h_star"] + 1e-9
                                           for i in range(len(ks) - 1))]
        rec["h_star_monotone_in_k"] = {"PASS": not nonmono, "violations": nonmono}
        # BV-4 witness: smallest grid k with h*(k) <= 8 h (a cheaper dry cooler than k_max) and its h*
        kmin = {c: next((k for k in ks if items[(c, cl)]["k"][k]["h_star"] <= 8 + TOL), None) for c in cells}
        per_site[cl + "|k_min_8h"] = s.cell_id.map(lambda c: float(kmin[c]) if kmin[c] else np.nan)
        per_site[cl + "|h_star_at_k_min_8h"] = s.cell_id.map(
            lambda c: items[(c, cl)]["k"][kmin[c]]["h_star"] if kmin[c] else np.nan)
        per_site[cl + "|critical_episode_h"] = s.cell_id.map(lambda c: items[(c, cl)]["k"][kmax]["critical_episode_h"])
        # storage–dry-cooler trade-off: F_dry(8 h) by k and the smallest grid k with h*(k) <= 8 h
        w = s.B_full_MW.astype(float)
        rec["tradeoff_F_dry_8h_by_k|B_full_MW"] = {
            k: wfrac(s.cell_id.map(lambda c: items[(c, cl)]["k"][k]["h_star"] <= 8 + TOL).astype(float), w) for k in ks}
        rec["tradeoff_F_dry_4h_by_k|B_full_MW"] = {
            k: wfrac(s.cell_id.map(lambda c: items[(c, cl)]["k"][k]["h_star"] <= 4 + TOL).astype(float), w) for k in ks}
        feas = s.h_star <= 8 + TOL
        # tank volume, footprint, cost (feasible sites; sized at h* and at the full 8 h)
        mwh_hstar = float((s.h_star[feas] * w[feas]).sum())
        mwh_8 = float(8.0 * w[feas].sum())
        rec["tank|B_full_MW"] = {
            "feasible_MW": float(w[feas].sum()),
            "at_h_star": {"MWh_th": mwh_hstar, "volume_m3": mwh_hstar * M3_PER_MWH,
                          "footprint_m2": mwh_hstar * M3_PER_MWH / TANK_HEIGHT_M, "cost_USD": mwh_hstar * COST_PER_MWH},
            "at_8h": {"MWh_th": mwh_8, "volume_m3": mwh_8 * M3_PER_MWH, "footprint_m2": mwh_8 * M3_PER_MWH / TANK_HEIGHT_M,
                      "cost_USD": mwh_8 * COST_PER_MWH},
            "per_MW_at_8h": {"volume_m3": 8 * M3_PER_MWH, "footprint_m2": 8 * M3_PER_MWH / TANK_HEIGHT_M,
                             "cost_USD": 8 * COST_PER_MWH},
            "capacity_weighted_h_star_feasible": {"median": wq(s.h_star[feas], w[feas], 0.5),
                                                  "p90": wq(s.h_star[feas], w[feas], 0.9)}}
        ep = s.cell_id.map(lambda c: items[(c, cl)]["k"][kmax]["critical_episode_h"])
        nh = s.cell_id.map(lambda c: items[(c, cl)]["k"][kmax]["need_hours_critical_year"])
        rec["critical_episode|feasible|B_full_MW"] = {"episode_h_median": wq(ep[feas], w[feas], 0.5),
                                                      "episode_h_p90": wq(ep[feas], w[feas], 0.9),
                                                      "need_hours_median": wq(nh[feas], w[feas], 0.5)}
        rec["critical_episode|infeasible|B_full_MW"] = {"episode_h_median": wq(ep[~feas], w[~feas], 0.5),
                                                        "need_hours_median": wq(nh[~feas], w[~feas], 0.5)}
        # BV-2: charge swing and utilization 0.9
        sw = s.cell_id.map(lambda c: items[(c, cl)].get("bv2", {}).get("mean_charge_swing_K", np.nan)).astype(float)
        ok = feas & np.isfinite(sw) & (s.h_star > 0)
        rec["bv2|B_full_MW"] = {
            "F_dry_8h_util0.9": wfrac((s.h_star / 0.9 <= 8 + TOL).astype(float), w),
            "F_dry_8h_util0.9_MW": float((w * (s.h_star / 0.9 <= 8 + TOL)).sum()),
            # energy-weighted mean swing per item, diagnostic only; it does NOT give a stored volume (internal review boundary-validity R1
            # finding 1: volume needs sum(q_t / dT_t) over the parcels in inventory at its peak). The volume statement that holds:
            # every charged parcel's swing is dT_fw * (1 + charge) >= dT_fw = 10 K by construction, so the 10 K-swing volume
            # upper-bounds any stored inventory under ideal (lossless, perfectly stratified) storage.
            "energy_weighted_mean_charge_swing_K_diagnostic": {
                "capacity_weighted_mean": float((sw[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None,
                "min_over_items": float(sw[ok].min()) if ok.any() else None,
                "max_over_items": float(sw[ok].max()) if ok.any() else None},
            "min_hourly_charge_swing_K_by_construction": 10.0}
        out[cl] = rec
    # crossings between climates at 8 h
    if "historical" in per_site and "stress_plus1.2K" in per_site:
        a, b = per_site["historical"] <= 8 + TOL, per_site["stress_plus1.2K"] <= 8 + TOL
        out["crossing_8h_MW|B_full_MW"] = {"feasible_to_infeasible": float(sites.B_full_MW[a & ~b].sum()),
                                           "infeasible_to_feasible": float(sites.B_full_MW[~a & b].sum())}
    return out, per_site


def continuous_check(tag, sites, cells):
    """DD-013: reset (1 January) vs continuous-state sequent peak at k_max."""
    d = OUT / "bv2_continuous" / tag
    items = {}
    for f in d.glob("*.json") if d.exists() else []:
        r = json.loads(f.read_text(encoding="utf-8"))
        items[(r["cell"], r["climate"])] = r
    if len(items) < 2 * len(cells):
        return {"complete": False, "n_items": len(items)}, {}
    bv1 = load_items(tag)
    out, cols = {"complete": True}, {}
    for cl in CLIM:
        hr = {c: items[(c, cl)]["h_star_reset"] for c in cells}
        hc = {c: items[(c, cl)]["h_star_continuous"] for c in cells}
        same = True
        if all((c, cl) in bv1 for c in cells):
            kmax = sorted(bv1[(cells[0], cl)]["k"], key=float)[-1]
            same = all(abs(hr[c] - bv1[(c, cl)]["k"][kmax]["h_star"]) < 1e-9 for c in cells)
        s = sites.copy()
        s["hr"], s["hc"] = s.cell_id.map(hr), s.cell_id.map(hc)
        w = s.B_full_MW.astype(float)
        chg = (s.hr <= 8 + TOL) & ~(s.hc <= 8 + TOL)
        carry = [c for c in cells if items[(c, cl)]["years_with_carry"] > 0]
        out[cl] = {"reset_equals_BV1": same, "cells_with_carry_into_jan1": carry,
                   "max_carry_h": max(items[(c, cl)]["max_carry"] for c in cells),
                   "cells_h_star_changed": [c for c in cells if hc[c] > hr[c] + 1e-9],
                   "class_change_8h_MW|B_full_MW": float(w[chg].sum()), "class_change_sites": s.site_id[chg].tolist(),
                   "F_dry_8h_continuous|B_full_MW": wfrac((s.hc <= 8 + TOL).astype(float), w),
                   "F_dry_8h_reset|B_full_MW": wfrac((s.hr <= 8 + TOL).astype(float), w)}
        cols[f"{cl}|h_star_continuous"] = s["hc"]
    return out, cols


def lapse_summary(sites):
    """DD-014 sensitivity (iv): lapse-rate-adjusted weather at sites with |dz| > 100 m vs the unadjusted classification."""
    d = OUT / "bv3" / "lapse"
    rows = []
    if not d.exists():
        return None
    bv1 = {t: load_items(t) for t in ("uncorrected", "corrected")}
    hub = {t: load_hub(t) for t in ("uncorrected", "corrected")}
    w = sites.set_index("site_id").B_full_MW
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        wx, cl, cell = r["weather"], r["climate"], r["cell"]
        b = bv1[wx].get((cell, cl))
        hb = hub[wx].get((cell, CLIM[cl]))
        h0 = b["k"][sorted(b["k"], key=float)[-1]]["h_star"] if b else None
        rows.append({"site": r["site"], "weather": wx, "climate": cl, "dz_m": r["dz_m"], "shift_K": r["lapse_shift_K"],
                     "MW": float(w[r["site"]]), "h_star_unadjusted": h0, "h_star_adjusted": r["h_star_reset"],
                     "F_dry_unadjusted": (h0 <= 8 + TOL) if h0 is not None else None, "F_dry_adjusted": r["F_dry"],
                     "F_nc_unadjusted": hb["F_nc"] if hb else None, "F_nc_adjusted": r["F_nc"]})
    df = pd.DataFrame(rows)
    res = {"n_items": len(df), "rows": rows}
    for (wx, cl), g in df.groupby(["weather", "climate"]):
        ok = g.F_dry_unadjusted.notna()
        res[f"{wx}|{cl}"] = {
            "F_dry_8h_lost_MW": float(g.MW[ok & (g.F_dry_unadjusted == True) & (g.F_dry_adjusted == False)].sum()),  # noqa: E712
            "F_dry_8h_gained_MW": float(g.MW[ok & (g.F_dry_unadjusted == False) & (g.F_dry_adjusted == True)].sum()),  # noqa: E712
            "F_nc_lost_MW": float(g.MW[(g.F_nc_unadjusted == True) & (g.F_nc_adjusted == False)].sum())}  # noqa: E712
    return res


def wq(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    if len(x) == 0:
        return None
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(cw, q)])


def main():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    res, cols = {"h_grid": H_GRID}, {}
    for tag in ("uncorrected", "corrected"):
        if (OUT / "bv1" / tag).exists():
            res[tag], ps = summarize_tag(tag, sites, cells)
            for k, v in ps.items():
                cols[f"{tag}|{k}"] = v
        res[f"bv2_continuous|{tag}"], pc = continuous_check(tag, sites, cells)
        for k, v in pc.items():
            cols[f"{tag}|{k}"] = v
    res["bv3_lapse"] = lapse_summary(sites)
    # BV-3 scenarios: (i) uncorrected historical, (ii) corrected historical, (iii) corrected + 1.2 K
    sc = {"(i) uncorrected": "uncorrected|historical", "(ii) corrected": "corrected|historical",
          "(iii) corrected+1.2K": "corrected|stress_plus1.2K", "(ref) uncorrected+1.2K": "uncorrected|stress_plus1.2K"}
    w = sites.B_full_MW.astype(float)
    if all(v in cols for v in sc.values()):
        res["bv3_scenarios|B_full_MW"] = {}
        for lab, key in sc.items():
            h = cols[key]
            r = {"F_dry_8h_MW": float((w * (h <= 8 + TOL)).sum()), "h_star_median_weighted": wq(h, w, 0.5)}
            if key + "|F_nc" in cols:
                r["F_nc_MW"] = float((w * cols[key + "|F_nc"]).sum())
            res["bv3_scenarios|B_full_MW"][lab] = r
        base = cols["uncorrected|historical"] <= 8 + TOL
        for lab in ("(ii) corrected", "(iii) corrected+1.2K"):
            x = cols[sc[lab]] <= 8 + TOL
            res["bv3_scenarios|B_full_MW"][lab]["F_dry_8h_change_vs_(i)"] = {
                "feasible_to_infeasible_MW": float(w[base & ~x].sum()), "infeasible_to_feasible_MW": float(w[~base & x].sum()),
                "sites_changing": sites.site_id[base != x].tolist()}
            if sc[lab] + "|F_nc" in cols and "uncorrected|historical|F_nc" in cols:
                a, b = cols["uncorrected|historical|F_nc"], cols[sc[lab] + "|F_nc"]
                res["bv3_scenarios|B_full_MW"][lab]["F_nc_change_vs_(i)"] = {
                    "feasible_to_infeasible_MW": float(w[a & ~b].sum()), "infeasible_to_feasible_MW": float(w[~a & b].sum())}
    # compositional adverse case (diagnostic, NOT pre-registered; internal review boundary-validity R1 non-blocking note: sensitivity
    # invariance is not compositional): weather tag × climate, plus the DD-014 lapse shift at its ten sites, plus utilization 0.9
    lap = {}
    for f in (OUT / "bv3" / "lapse").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        lap[(r["site"], r["weather"], r["climate"])] = r["h_star_reset"]
    comp = {}
    for tag in ("uncorrected", "corrected"):
        for cl in CLIM:
            key = f"{tag}|{cl}"
            if key not in cols:
                continue
            h0 = cols[key].astype(float).values
            h = np.array([lap.get((sid, tag, cl), h0[i]) for i, sid in enumerate(sites.site_id)])
            base = h0 <= 8 + TOL
            ok = h / 0.9 <= 8 + TOL
            comp[f"{key}+lapse+util0.9"] = {"F_dry_8h_MW": float((w * ok).sum()), "F_dry_8h_MW_without": float((w * base).sum()),
                                           "sites_lost": sites.site_id[base & ~ok].tolist(),
                                           "MW_lost": float(w[base & ~ok].sum())}
    res["compositional_adverse_diagnostic|B_full_MW"] = comp
    (OUT / "bv_summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    tab = sites[["site_id", "site", "country", "cell_id", *BASES]].copy()
    for k, v in cols.items():
        tab[k] = v.values
    tab.to_csv(OUT / "bv_sites.csv", index=False)
    for tag in ("uncorrected", "corrected"):
        for cl in CLIM:
            r = res.get(tag, {}).get(cl, {})
            if "F_dry_8h|B_full_MW" in r:
                v = r["validation_vs_hub_screen_8h"]
                print(f"{tag:11s} {cl:16s} F_dry(0/8/24/48 h) = {r['F_dry_0h|B_full_MW']:.3f} / {r['F_dry_8h|B_full_MW']:.3f} / "
                      f"{r['F_dry_24h|B_full_MW']:.3f} / {r['F_dry_48h|B_full_MW']:.3f}; 8 h MW {r['F_dry_8h_MW|B_full_MW']:.0f}; "
                      f"validation {v if isinstance(v, str) else ('PASS' if v['PASS'] else 'FAIL ' + str(len(v['mismatches'])))}")
            elif r:
                print(tag, cl, r)
    for tag in ("uncorrected", "corrected"):
        c = res.get(f"bv2_continuous|{tag}", {})
        print(f"continuous-state check {tag}:", {cl: {k: v for k, v in c[cl].items() if k != "cells_h_star_changed"}
                                                 for cl in CLIM if cl in c} if c.get("complete") else c)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("bv3_scen")}, indent=1))
    if res["bv3_lapse"]:
        print("lapse (iv):", json.dumps({k: v for k, v in res["bv3_lapse"].items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
