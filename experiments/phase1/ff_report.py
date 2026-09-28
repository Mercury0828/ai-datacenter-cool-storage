"""Package 2 report (PR-FF): condense ff_summary_<tag>.json (uncorrected and, if present, corrected) into the ledger tables and
evaluate the pre-registered decision-rule criteria (A1 §4 end-of-Phase-1 rule, as declared in PR-FF):
1 Material: under a water ban with ≤ 8 h storage, Σ w·C_phys or the economically selected compressor capacity ≥ 2 GW_th (B-full).
2 Robust: the same quantity ≥ 1 GW_th in both climates, under corrected weather, on B-full and B-2027.
3 Decision-changing: a flip inside the declared plausible ranges moves ≥ 2 GW between zones (water allowed or banned).
Writes experiments/phase1/out/ff_report.md and prints it.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/phase1/out"
CL = ("historical", "stress_plus1.2K")


def load(tag):
    p = OUT / f"ff_summary_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    S = {t: load(t) for t in ("uncorrected", "corrected")}
    lines = []
    for tag, s in S.items():
        if s is None:
            continue
        lines.append(f"### {tag} (items {s['n_items']}, complete {s['complete']})")
        for b in ("B_full_MW", "B_2027_MW", "B_now_MW"):
            f = s["fleet"][b]
            for cl in CL:
                r = f[cl]
                z, zb = r["lc_zone_share"], r["ban_zone_share"]
                lines.append(f"- {b} {cl}: water allowed Z1/Z2/Z3/Z∅ = {z['Z1']:.3f}/{z['Z2']:.3f}/{z['Z3']:.3f}/{z['Zempty']:.3f}; "
                             f"banned Z1/Z3/Z∅ = {zb['Z1']:.3f}/{zb['Z3']:.3f}/{zb['Zempty']:.3f}; ban split GW_th installed "
                             f"{r['ban_split_GW_th']['installed_least_cost']:.2f} = physical {r['ban_split_GW_th']['physically_necessary']:.2f} "
                             f"+ economic {r['ban_split_GW_th']['economic']:.2f}; water-allowed installed "
                             f"{r['water_allowed_split_GW_th']['installed_least_cost']:.2f} (physical "
                             f"{r['water_allowed_split_GW_th']['physically_necessary']:.2f})")
        f = s["fleet"]["B_full_MW"]
        for cl in CL:
            r = f[cl]
            lines.append(f"- B-full {cl} scale (GW_th chiller / GW_e peak / TWh/yr / GL/yr / GWh_th store / B$/yr):")
            for pre, lab in (("lc", "least cost, water"), ("ban", "least cost, water ban"), ("cf", "compressor-free, water"),
                             ("cfdry", "compressor-free, water-free (feasible sites)"),
                             ("cfdrycomplete", "water-free: compressor-free where feasible + C_phys elsewhere"),
                             ("tower", "tower counterfactual")):
                x = r[f"{pre}_scale"]
                lines.append(f"  - {lab}: {x['chiller_GW_th']:.2f} / {x['peak_GW_e']:.2f} / {x['elec_TWh_yr']:.2f} / "
                             f"{x['water_GL_yr']:.2f} / {x['tank_GWh_th']:.1f} / {x['cost_BUSD_yr']:.3f} (covered {x['covered_MW']:.0f} MW; "
                             f"unmet in violating years {x['viol_unmet_GWh_in_violating_years']:.2f} GWh)")
            e = r["eliminate_compressors_with_water"]
            lines.append(f"  - eliminate compressors with water: {e['MW']:.0f} MW, Δcost {e['d_cost_MUSD_yr']:.1f} M$/yr, Δwater "
                         f"{e['d_water_GL_yr']:.2f} GL/yr, Δelec {e['d_elec_TWh_yr']:.3f} TWh/yr, Δpeak {e['d_peak_GW_e']:.3f} GW_e")
            e = r["eliminate_compressors_water_ban"]
            lines.append(f"  - eliminate compressors under a ban: feasible {e['MW_feasible']:.0f} MW (infeasible {e['MW_infeasible']:.0f}), "
                         f"Δcost {e['d_cost_MUSD_yr']:.1f} M$/yr, Δstore {e['d_tank_GWh_th']:.1f} GWh_th, Δelec {e['d_elec_TWh_yr']:.3f} TWh/yr")
            cp = r["cphys_GW_th"]
            lines.append("  - C_phys GW_th (ban / ≤1% / ≤10% / ≤100% tower water / peak ≤50% / free) at h = 0, 2, 4, 8: " + "; ".join(
                f"h{h}: {cp[f'cphys_ban_h{h}']:.2f}/{cp[f'cphys_w0.01_h{h}']:.2f}/{cp[f'cphys_w0.1_h{h}']:.2f}/{cp[f'cphys_w1_h{h}']:.2f}/"
                f"{cp[f'cphys_peak50_h{h}']:.2f}/{cp[f'cphys_free_h{h}']:.2f}" for h in (0, 2, 4, 8)))
            lines.append(f"  - mixed mode: {r['mixed_mode']}; near-ties MW: {r['near_ties_MW']}")
        co = s.get("coincident|B_full_MW") or {}
        for k, v in co.items():
            lines.append(f"- coincident {k}: {v['coincident_peak_GW_e_mean']:.3f} GW_e (max {v['coincident_peak_GW_e_max']:.3f}) vs sum of "
                         f"site peaks {v['sum_of_site_peaks_GW_e_mean']:.3f}; ratio {v['ratio_coincident_to_sum_mean']:.3f}")
        fl = s["flips|B_full_MW"]
        for axis in ("tank_kUSD_per_MWh", "water_mult", "elec_mult"):
            mv = fl[axis].get("moved_between_plausible_bounds", {})
            lines.append(f"- flips {axis} (plausible bounds): " + "; ".join(f"{k}: {v['MW']:.0f} MW" for k, v in mv.items()))
    # decision-rule criteria
    crit = {}
    u, c = S["uncorrected"], S["corrected"]
    if u is not None:
        # criterion 1 is met by the economic quantity; criterion 2 must use the SAME quantity (internal review full-frame R1 note)
        def qty(s, b, cl):
            return s["fleet"][b][cl]["ban_split_GW_th"]["economic"]
        crit["1_material_B_full_hist_GW_th"] = qty(u, "B_full_MW", "historical")
        crit["1_met"] = crit["1_material_B_full_hist_GW_th"] >= 2.0
        rob = {f"uncorrected|{b}|{cl}": qty(u, b, cl) for b in ("B_full_MW", "B_2027_MW") for cl in CL}
        if c is not None:
            rob.update({f"corrected|{b}|{cl}": qty(c, b, cl) for b in ("B_full_MW", "B_2027_MW") for cl in CL})
        crit["2_robust_values_GW_th"] = rob
        crit["2_met"] = (c is not None) and all(v >= 1.0 for v in rob.values())
        moved = {}
        for tag, s in S.items():
            if s is None:
                continue
            for axis in ("tank_kUSD_per_MWh", "water_mult", "elec_mult"):
                for k, v in s["flips|B_full_MW"][axis].get("moved_between_plausible_bounds", {}).items():
                    moved[f"{tag}|{axis}|{k}"] = v["MW"]
        crit["3_moves_MW"] = moved
        crit["3_met"] = any(v >= 2000 for v in moved.values())
        crit["ALL_MET"] = crit["1_met"] and crit["2_met"] and crit["3_met"]
    lines.append("### Decision-rule criteria (PR-FF)")
    lines.append("```\n" + json.dumps(crit, indent=1) + "\n```")
    text = "\n".join(lines)
    (OUT / "ff_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
