"""Tables and figures of the Supplemental Information, generated from frozen artifacts only.

Writes paper/si/tables/*.tex and paper/si/figs/*.pdf|png. Usage: python paper/si/make_si.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figs" / "src"))
import common as K  # noqa: E402

TAB = HERE / "tables"
FIG = HERE / "figs"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
OUT = K.OUT

FAMILIES = {"CAM": ["CESM2-WACCM", "CMCC-CM2-SR5", "CMCC-ESM2", "FIO-ESM-2-0", "NorESM2-LM", "NorESM2-MM", "TaiESM1", "CIESM",
                    "BCC-CSM2-MR"],
            "ECHAM": ["AWI-CM-1-1-MR", "CAMS-CSM1-0", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "NESM3"],
            "IFS": ["EC-Earth3", "EC-Earth3-CC", "EC-Earth3-Veg", "EC-Earth3-Veg-LR"],
            "UM": ["ACCESS-CM2", "KACE-1-0-G"], "GFDL-AM": ["GFDL-CM4", "GFDL-ESM4", "KIOST-ESM"],
            "CAS/IAP": ["CAS-ESM2-0", "FGOALS-f3-L", "FGOALS-g3"], "INM": ["INM-CM4-8", "INM-CM5-0"],
            "LMDZ": ["IPSL-CM6A-LR"], "MIROC": ["MIROC6"], "MRI": ["MRI-ESM2-0"], "CanAM": ["CanESM5"], "GFS": ["IITM-ESM"]}


def tex(s):
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


def cds():
    rec = {}
    for f in (OUT / "m2" / "cds_rebuild").glob("cell_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        rec[d["cell"]] = {cl: d[cl]["h_star"] for cl in ("historical", "stress_plus1.2K", "ssp245_median")}
    return rec


def fmt_h(h):
    if h <= 1e-9:
        return "0"
    if h > 99.5:
        return "%.0f" % h
    return "%.1f" % h


# ------------------------------------------------------------------------------------------------ Table S1
def table_s1():
    s = K.sites()
    rec = cds()
    sb = K.jload("sb/sb23_robustness.json")
    core, out = set(sb["joint_2040"]["robust_core_cells"]), set(sb["joint_2040"]["firmly_out_cells"])
    s = s.sort_values("B_full_MW", ascending=False)
    rows = []
    for _, r in s.iterrows():
        h = rec[r.cell_id]
        cls = "core" if r.cell_id in core else ("out" if r.cell_id in out else "sens.")
        rows.append("%s & %s & %.0f & %.0f & %.0f & %.2f, %.2f & %s & %s & %s & %s \\\\" % (
            tex(r.site), tex(r.country.replace("United States", "USA").replace("United Arab Emirates", "UAE")
                             .replace("United Kingdom", "UK")),
            r.B_full_MW, r.B_2027_MW, r.B_now_MW, r.era5_lat, r.era5_lon,
            fmt_h(h["historical"]), fmt_h(h["stress_plus1.2K"]), fmt_h(h["ssp245_median"]), cls))
    body = "\n".join(rows)
    (TAB / "table_s1.tex").write_text(
        "\\begin{footnotesize}\n\\setlength{\\tabcolsep}{3.2pt}\n"
        "\\begin{longtable}{@{}p{46mm}lrrrlrrrl@{}}\n"
        "\\caption{\\textbf{The 93 tracked sites.} IT capacity in MW for the three bases (B-full, the full announced build-out;"
        " B-2027, capacity dated no later than December 2027; B-now, capacity operating on 24 September 2026), the ERA5 grid"
        " point used (latitude, longitude), the minimum store \\hstar{} in hours of full IT heat under 1990--2025 weather, the"
        " +1.2~K stress and the SSP2-4.5 scenario around 2040 (Climate Data Store series), and the robustness class around"
        " 2040 (Note~S8). Rows are rounded; totals and classes use unrounded values.}\\label{tab:s1}\\\\\n\\toprule\n"
        "Site & Country & B-full & B-2027 & B-now & ERA5 point & \\hstar{} hist. & \\hstar{} +1.2~K & \\hstar{} 2040 & Class\\\\\n"
        "\\midrule\n\\endfirsthead\n\\toprule\nSite & Country & B-full & B-2027 & B-now & ERA5 point & \\hstar{} hist. &"
        " \\hstar{} +1.2~K & \\hstar{} 2040 & Class\\\\\n\\midrule\n\\endhead\n" + body +
        "\n\\bottomrule\n\\end{longtable}\n\\end{footnotesize}\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ Table S5
def table_s6():
    s = K.sites()
    w = s.groupby("cell_id").B_full_MW.sum()
    allm = K.jload("cmip/deltas_all.json")
    cells = allm["cells"]
    wc = np.array([w.get(c, 0.0) for c in cells])
    per = K.jload("sb/sb23_robustness.json")["SB2"]["F_dry_8h_GW_per_model"]
    rows = []
    for fam, models in FAMILIES.items():
        for m in models:
            d = json.loads((OUT / "cmip" / f"deltas_{m}.json").read_text(encoding="utf-8"))
            a = np.array(d["ssp245"], dtype=float)
            if a.shape[0] == 12 and a.shape[1] == len(cells):
                a = a.T
            jja = np.nanmean(a[:, 5:8], axis=1)
            val = np.nansum(jja * wc) / wc[np.isfinite(jja)].sum()
            rows.append("%s & %s & %.2f & %.1f \\\\" % (fam if m == models[0] else "", tex(m), val, per[m]))
    (TAB / "table_s6.tex").write_text(
        "\\begin{table}[p]\n\\centering\n\\small\n\\caption{\\textbf{The 33 CMIP6 models with complete SSP2-4.5 output"
        " and their families.} Families follow a declared grouping by shared atmospheric-model lineage and institution"
        " (used for the held-out test of Note~S12). June to August change of near-surface temperature, 2031--2050 minus 1995--2014, averaged over the 80 grid"
        " cells with capacity weights, and the capacity that passes within 8~h of storage when the model's own monthly field"
        " is applied (GW, 6~K total approach).}\\label{tab:s6}\n"
        "\\begin{tabular}{@{}llrr@{}}\n\\toprule\nFamily & Model & JJA change [K] & Capacity within 8~h [GW]\\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ Table S6
def table_s8():
    s = K.sites()
    pr = K.jload("sb/sb4_price1k.json")
    rows = []
    for cl, lab in (("historical", "1990--2025"), ("ssp245_median", "around 2040")):
        for c in sorted(pr[cl]["cells"], key=lambda c: -c["MW"]):
            names = "; ".join(s[s.cell_id == c["cell"]].site.tolist())
            rows.append("%s & %s & %.0f & %.1f & %.1f & %.0f & %.0f & %.0f \\\\" % (
                lab, tex(names), c["MW"], c["h6_record"], c["h5_record"], c["B_storage_capex_per_kW"],
                c["B_ban_capex_per_kW"], c["B_lc_capex_per_kW"]))
    (TAB / "table_s8.tex").write_text(
        "\\begin{table}[p]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{\\textbf{The priced one-kelvin lever at the sites it keeps within 8~h.} Stores at the 6~K and 5~K total"
        " approach (hours of full IT heat), and the break-even capital premium of the 5~K design (\\$ per kW of IT, 2024) against"
        " three alternatives on the same weather: the water-free, compressor-free design at 6~K with its longer store, the"
        " least-cost design under a water ban (trim chiller allowed) and the least-cost design with water allowed. Negative"
        " values mean the alternative is cheaper. Annualized differences are converted with the capital recovery factor"
        " of 0.1168.}\\label{tab:s8}\n\\begin{tabular}{@{}lp{58mm}rrrrrr@{}}\n\\toprule\nClimate & Sites & MW & "
        "\\hstar{} 6~K & \\hstar{} 5~K & vs longer store & vs ban design & vs water allowed\\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ Table S8
NAMES = {"B-SAA": "Ensemble average (B-SAA)", "B-ROB": "Worst training field (B-ROB)", "B-QNT": "Ensemble quantile (B-QNT)",
         "B-SCAL": "Summer-warming quantile (B-SCAL)", "BL2-shift": "Uniform margin (BL2-shift)",
         "BL3": "Margin, no violating year (BL3)", "BL1-shift": "Design day + margin (BL1-shift)",
         "DR-F": "Robust over model fields (DR-F)", "B-MON-q0.5": "Monthly quantile field, 0.5 (B-MON)", "B-MON-q0.75": "Monthly quantile field, 0.75 (B-MON)",
         "B-MON-q0.9": "Monthly quantile field, 0.9 (B-MON)", "B-MON-q1": "Monthly quantile field, 1 (B-MON)"}


def table_s12():
    r = K.jload("r2f/stage2_summary.json")
    order = ["DR-F", "B-SAA", "B-ROB", "B-QNT", "B-SCAL", "BL2-shift", "BL3", "BL1-shift", "B-MON-q0.5", "B-MON-q0.75",
             "B-MON-q0.9", "B-MON-q1"]
    rows = []
    for n in order:
        cells = []
        for eta in ("0.025", "0.05", "0.1"):
            p = r[eta]["procedures"][n]
            mark = "" if p["V"] <= float(eta) + 1e-12 else "$^{\\dagger}$"
            cells.append("%.2f%s & %s" % (100 * p["V"], mark, "{:,.0f}".format(p["K"])))
        ri = r["0.05"]["bootstrap"]["risk_interval_90"][n]
        rows.append("%s & %s & %.1f--%.1f \\\\" % (NAMES[n], " & ".join(cells), 100 * ri[0], 100 * ri[1]))
    (TAB / "table_s12.tex").write_text(
        "\\begin{table}[p]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\caption{\\textbf{Held-out validation across climate-model families.} Pooled violation (\\%, capacity-weighted,"
        " averaged over the 12 held-out families) and registered annualized cost (\\$ per MW-year, historical weather)"
        " of every procedure at the three risk budgets, the eight tunable ones calibrated at each budget and the four monthly quantile fields fixed, and the 90\\,\\% cluster-bootstrap"
        " risk interval at 5\\,\\%. $^{\\dagger}$ not compliant at that budget.}\\label{tab:s12}\n"
        "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n& \\multicolumn{2}{c}{2.5\\,\\% budget} & \\multicolumn{2}{c}{5\\,\\% budget}"
        " & \\multicolumn{2}{c}{10\\,\\% budget} & 90\\,\\% interval\\\\\nProcedure & Viol. & Cost & Viol. & Cost & Viol. & Cost & at 5\\,\\% \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ Tables S10, S11 (Note S12)
TESTS = {"ssp245_median": "SSP2-4.5 median, around 2040", "ssp585_p90": "SSP5-8.5 90th percentile, around 2040",
         "F0": "Observed 2015--2025"}


def setting(knob):
    """'BL1-shift|a1.75|50y' -> readable setting of an executable benchmark point."""
    parts = knob.split("|")
    val = parts[1]
    amt = ("+%s~K" % val[1:]) if val[0] == "a" else ("$%s\\times$ drift" % val[1:])
    if parts[0] == "DR":
        return ("$\\omega$ = %s~K" % val[1:]) if val[0] == "a" else ("$\\omega$ = %s$\\times$ drift" % val[1:])
    if parts[0] == "BL1-shift":
        return "design day, %s-yr, %s" % (parts[2].rstrip("y"), amt)
    return {"BL2-shift": "uniform margin, ", "BL3": "margin, no violating year, "}[parts[0]] + amt


def table_s11():
    r = K.jload("r2s_summary.json")["oracle"]
    c = K.jload("r2s_certificates.json")
    rows = []
    for t, lab in TESTS.items():
        for j, lv in enumerate(("0.025", "0.05", "0.1")):
            o, cc = r[t][lv], c[t][lv]
            b, d = o["best_executable_baseline"], o["executable"]["DR"]
            rows.append("%s & %.1f & %s & %s (%s) & %s (%s) & %.2f & %.2f \\\\" % (
                lab if j == 0 else "", 100 * float(lv), "{:,.0f}".format(o["pooled_oracle_cost"]),
                "{:,.0f}".format(b["cost"]), setting(b["knob"]), "{:,.0f}".format(d["cost"]), setting(d["knob"]),
                o["headroom_pooled_oracle_pct"], cc["headroom_pooled_vs_pareto_polyline_pct"]))
        rows.append("\\addlinespace")
    (TAB / "table_s11.tex").write_text(
        "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{\\textbf{Oracle headroom on the development benchmark.} Registered annualized cost (\\$ per MW-year) of the"
        " pooled perfect-foresight oracle, of the cheapest executable baseline point and of the cheapest executable point of the"
        " per-year robust formulation, each at a pooled capacity-weighted test violation within the budget, on the 15 sites."
        " An executable point uses one setting for all sites and no interpolation. Headroom is the oracle's saving against the"
        " cheapest executable baseline point and against the interpolated baseline frontier (\\%).}\\label{tab:s11}\n"
        "\\begin{tabular}{@{}p{27mm}rrp{40mm}p{33mm}rr@{}}\n\\toprule\n"
        "Test & Budget (\\%) & Oracle & Cheapest baseline point & Per-year robust point & \\multicolumn{2}{c}{Headroom (\\%)}\\\\\n"
        "\\cmidrule(l){6-7}\n& & & & & executable & frontier\\\\\n\\midrule\n" + "\n".join(rows[:-1])
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


def table_s13():
    f = K.jload("r2f/frontier_summary_0.05.json")
    fmt = {"DR-F": "normalized radius %.4f", "B-SAA": "violation budget %.4f", "B-QNT": "quantile %.3f", "B-SCAL": "quantile %.3f",
           "BL2-shift": "shift %.1f~K", "BL3": "shift %.1f~K"}
    rows = []
    for n in sorted((k for k in f if not k.startswith("DR-F_")), key=lambda k: f[k]["best_compliant_point"]["K"]):
        b = f[n]["best_compliant_point"]
        s = ("design day, %s-yr, +%.1f~K" % (b["variant"].rstrip("y"), b["knob"])) if n == "BL1-shift" else fmt[n] % b["knob"]
        rows.append("%s & %s & %.2f & %s & %s \\\\" % (NAMES[n], s, 100 * b["V"], "{:,.0f}".format(b["K"]),
                                                     "{:,}".format(f[n]["points"])))
    (TAB / "table_s13.tex").write_text(
        "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
        "\\caption{\\textbf{Retrospective test frontiers of the held-out validation at 5\\,\\%.} For each calibrated procedure,"
        " every distinct setting on its path was applied to all 12 folds and evaluated on the test; the table gives the cheapest"
        " setting whose pooled violation is at most 5\\,\\%, its violation (\\%) and cost (\\$ per MW-year), and the number of"
        " settings evaluated. The settings are chosen with hindsight and do not change the verdict.}\\label{tab:s13}\n"
        "\\begin{tabular}{@{}llrrr@{}}\n\\toprule\nProcedure & Cheapest compliant setting & Violation & Cost & Settings\\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ per-site tables (Notes S6, S11)
def design_parts(s):
    """'dry k=0.8557 +ad tank=0.179778h C=0.000' -> (k, pads, store h)."""
    k = float(s.split("k=")[1].split()[0])
    tank = float(s.split("tank=")[1].split("h")[0])
    return k, "+ad" in s, tank


def longtable(path, spec, caption, label, head, rows):
    (TAB / path).write_text(
        "{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{" + spec + "}\n\\caption{" + caption + "}\\label{"
        + label + "}\\\\\n\\toprule\n" + head + "\\midrule\n\\endfirsthead\n\\toprule\n" + head + "\\midrule\n\\endhead\n"
        "\\bottomrule\n\\endfoot\n" + "\n".join(rows) + "\n\\end{longtable}}\n", encoding="utf-8")


def table_s5():
    d = pd.read_csv(OUT / "ff_sites_uncorrected.csv")
    h = d[d.climate == "historical"].set_index("site_id")
    st = d[d.climate == "stress_plus1.2K"].set_index("site_id")
    h = h.sort_values("B_full_MW", ascending=False)
    rows = []
    for sid, r in h.iterrows():
        k, ad, tank = design_parts(r.lc_design)
        _, _, tb = design_parts(r.ban_design)
        rows.append("%s & %s & %.2f & %s & %s & %s / %s & %s & %.0f / %.0f & %.0f / %.0f \\\\" % (
            tex(r.site), "{:,.0f}".format(r.B_full_MW), k, "yes" if ad else "no", fmt_h(tank), "{:,.0f}".format(r.lc_water_m3),
            "{:,.0f}".format(st.loc[sid, "lc_water_m3"]), fmt_h(tb), 1e3 * r.ban_c_ch, 1e3 * st.loc[sid, "ban_c_ch"],
            1e3 * r.cphys_ban_h8,
            1e3 * st.loc[sid, "cphys_ban_h8"]))
    tot = {cl: g for cl, g in d.groupby("climate")}
    w = {cl: (g.lc_water_m3 * g.B_full_MW).sum() / 1e6 for cl, g in tot.items()}
    c = {cl: (g.ban_c_ch * g.B_full_MW).sum() / 1e3 for cl, g in tot.items()}
    p = {cl: (g.cphys_ban_h8 * g.B_full_MW).sum() / 1e3 for cl, g in tot.items()}
    head = ("& & \\multicolumn{4}{c}{Water allowed} & \\multicolumn{3}{c}{Water ban}\\\\\n\\cmidrule(lr){3-6}\\cmidrule(l){7-9}\n"
            "Site & MW & $k$ & Pads & Store & Water & Store & Chiller & Minimum\\\\\n")
    longtable("table_s5.tex", "@{}lrrlrrrrr@{}",
              "\\textbf{Least-cost designs by site at central prices.} IT capacity (B-full). With water allowed: dry-cooler size $k$"
              " (modules per MW), adiabatic pads and store (h) with 1990--2025 weather, and annual water use (m\\textsuperscript{3}"
              " per MW of IT) with 1990--2025 weather / under the +1.2~K stress. Under a water ban: store (h) with 1990--2025"
              " weather, installed trim-chiller capacity and the minimum chiller capacity within the design envelope, store at most"
              " 8~h (kW of heat per MW of IT, 1990--2025 / +1.2~K). Fleet totals: water %.1f / %.1f~GL per year, installed"
              " chillers %.2f / %.2f~\\GWth{}, envelope minimum %.2f / %.2f~\\GWth{}. Earlier ERA5 copy (Note~S2)."
              % (w["historical"], w["stress_plus1.2K"], c["historical"], c["stress_plus1.2K"], p["historical"],
                 p["stress_plus1.2K"]), "tab:s5", head, rows)


def table_s10():
    s = pd.read_csv(K.ROOT / "data" / "sites" / "hub_sites.csv").sort_values("B_full_MW", ascending=False)
    rows, pool = [], {"st": [0.0, 0.0], "dr": [0.0, 0.0]}
    for _, r in s.iterrows():
        b = json.loads((OUT / "oos_b" / f"{r.cell_id}.json").read_text(encoding="utf-8"))["a"]
        c = json.loads((OUT / "oos_c" / f"{r.cell_id}.json").read_text(encoding="utf-8"))
        h_tr = b["h_train"]
        e_test = np.asarray(b["E_hist"])[25:]
        v_st = int((e_test > h_tr + 1e-9).sum()) if h_tr <= 8 + 1e-9 else None
        ao, asc = c["a"]["obs"], c["a"]["scen"]
        v_o = sum(ao["viol"]) if ao["h_drift"] <= 8 + 1e-9 else None
        v_s = sum(asc["viol"]) if asc["h_drift"] <= 8 + 1e-9 else None
        if v_st is not None:
            pool["st"][0] += r.B_full_MW * v_st / 11
            pool["st"][1] += r.B_full_MW
        if v_o is not None:
            pool["dr"][0] += r.B_full_MW * v_o / 11
            pool["dr"][1] += r.B_full_MW
        n = lambda v: "--" if v is None else "%d" % v  # noqa: E731
        rows.append("%s & %s & %.2f & %.2f & %s & %s & %s & %s & %s & %s \\\\" % (
            tex(r.site), "{:,.0f}".format(r.B_full_MW), c["delta_K"]["obs"], c["delta_K"]["scen"], fmt_h(h_tr), n(v_st),
            fmt_h(ao["h_drift"]), n(v_o), fmt_h(asc["h_drift"]), n(v_s)))
    print("table_s10 pooled stationary %.4f, drift-aware %.4f" % (pool["st"][0] / pool["st"][1], pool["dr"][0] / pool["dr"][1]))
    head = ("& & \\multicolumn{2}{c}{Drift (K)} & \\multicolumn{2}{c}{Stationary} & \\multicolumn{2}{c}{Drift-aware, 2015--2025}"
            " & \\multicolumn{2}{c}{Drift-aware, 2040}\\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}"
            "\\cmidrule(l){9-10}\nSite & MW & 2020 & 2040 & Store & Viol. & Store & Viol. & Store & Viol.\\\\\n")
    longtable("table_s10.tex", "@{}lrrrrrrrrr@{}",
              "\\textbf{Water-free stores sized on 1990--2014, by site.} IT capacity (B-full); drift added for the observed decade"
              " (use-period middle 2020) and around 2040 (K); the stationary store sized on the training years and its violating"
              " years of the 11 observed test years; the drift-aware store and its violating years of 11; the drift-aware store"
              " around 2040 and its violating years of 36 in the central SSP2-4.5 scenario. Stores in hours of full IT heat;"
              " dashes mark stores beyond 8~h, which are not retained. Earlier ERA5 copy (Note~S2).", "tab:s10", head, rows)


# ------------------------------------------------------------------------------------------------ Tables S13, S14 (Note S13)
EXT_NAMES = {"humain_dammam": "Dammam, Saudi Arabia", "humain_riyadh": "Riyadh, Saudi Arabia",
             "reliance_jamnagar": "Jamnagar, India", "adani_google_vizag": "Visakhapatnam, India"}
CLIM_SHORT = {"historical": "1990--2025", "stress_plus1.2K": "+1.2~K"}


def call(p):
    return "within 8~h" if p >= 0.9 else ("beyond 8~h" if p <= 0.1 else "marginal")


def table_s15():
    e = pd.read_csv(OUT / "ruler" / "external.csv")
    md = pd.read_csv(K.ROOT / "data" / "sites" / "md_hubs.csv").set_index("site_id")["site"]
    rows = []
    for sid, g in e.groupby("site", sort=False):
        name = EXT_NAMES.get(sid) or md[sid]
        for j, (_, r) in enumerate(g.iterrows()):
            truth = "within 8~h" if r.h_star <= 8 + 1e-9 else "beyond 8~h"
            c = call(r.p_h8)
            flag = "$^{\\ast}$" if c != "marginal" and c != truth else ""
            rows.append("%s & %s & %.1f & %s & %.1f & %.2f & %s%s & %s \\\\" % (
                tex(name) if j == 0 else "", CLIM_SHORT[r.climate], r.T_04, sgn(r.x), r.y, r.p_h8, c, flag, fmt_h(r.h_star)))
    (TAB / "table_s15.tex").write_text(
        "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{\\textbf{The climate ruler on 15 grid cells outside the fitted set.} The 11 western US hubs of Mokkapati"
        " and Das (Cheyenne, which shares a grid cell with a tracked site, is excluded) and four announced hot sites outside the"
        " tracked frame. $T_{0.4}$ (\\textdegree{}C), the coordinates $x$ and $y$ (K), the fitted probability of a store of at"
        " most 8~h, the ruler's call and the store \\hstar{} (h) from the full model. $^{\\ast}$ wrong confident call.}"
        "\\label{tab:s15}\n\\begin{tabular}{@{}llrrrrlr@{}}\n\\toprule\nSite & Climate & $T_{0.4}$ & $x$ & $y$ & $p$ & Call"
        " & \\hstar{}\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


def sgn(v, fmt="%.1f"):
    s = fmt % abs(v)
    return ("$-$" + s) if v < 0 and s != fmt % 0 else s


def table_s17():
    p = pd.read_csv(OUT / "ruler" / "points.csv")
    s = pd.read_csv(K.ROOT / "data" / "sites" / "hub_sites.csv")
    big = s.sort_values("B_full_MW", ascending=False).groupby("cell_id")
    name = {c: tex(g.site.iloc[0]) + ("" if len(g) == 1 else " +%d" % (len(g) - 1)) for c, g in big}
    mw = s.groupby("cell_id").B_full_MW.sum()

    def cl(r):
        c = "W" if r.p_loco_h8 >= 0.9 else ("B" if r.p_loco_h8 <= 0.1 else "M")
        wrong = (c == "W" and r.h_star > 8 + 1e-9) or (c == "B" and r.h_star <= 8 + 1e-9)
        return c + ("$^{\\ast}$" if wrong else "")

    rows = []
    for cell in mw.sort_values(ascending=False).index:
        g = p[p.cell == cell].set_index("climate")
        h, st, sc = g.loc["historical"], g.loc["stress_plus1.2K"], g.loc["ssp245_median"]
        rows.append("%s & %s & %.2f & %s & %.1f & %s & %s & %s & %s & %s & %s & %.1f & %s & %s \\\\" % (
            name[cell], "{:,.0f}".format(mw[cell]), h.T_lim, sgn(h.x), h.y, fmt_h(h.h_star), cl(h), sgn(st.x), fmt_h(st.h_star),
            cl(st), sgn(sc.x), sc.y, fmt_h(sc.h_star), cl(sc)))
    head = ("& & & \\multicolumn{4}{c}{1990--2025} & \\multicolumn{3}{c}{+1.2~K stress} & \\multicolumn{4}{c}{SSP2-4.5, around"
            " 2040}\\\\\n\\cmidrule(lr){4-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-14}\nGrid cell & MW & $T_{\\lim}$ & $x$ & $y$"
            " & \\hstar{} & Call & $x$ & \\hstar{} & Call & $x$ & $y$ & \\hstar{} & Call\\\\\n")
    longtable("table_s17.tex", "@{}lrrrrrlrrlrrrl@{}",
              "\\textbf{Coordinates and held-out calls of the climate ruler.} Each of the 80 grid cells is named by its largest"
              " site (+$n$ other sites in the cell), with its IT capacity (MW) and limit temperature $T_{\\lim}$"
              " (\\textdegree{}C). For each climate: the coordinates $x$ and $y$ (K), the store \\hstar{} (h) and the ruler's"
              " call when the cell is held out, W within 8~h ($p \\ge 0.9$), B beyond 8~h ($p \\le 0.1$) and M marginal."
              " $^{\\ast}$ wrong confident call. The uniform +1.2~K stress leaves $y$ unchanged. Earlier ERA5 copy (Note~S2).",
              "tab:s17", head, rows)


def station_name(s):
    if s[:2].isalpha() and s[2] == "_":
        return "%s, %s" % (" ".join(s[3:].split("_")), s[:2])
    return s


def table_s16():
    s = pd.read_csv(OUT / "ruler_b" / "stations.csv")
    rows = []
    for _, r in s.iterrows():
        truth = "within" if r.station_h_star <= 8 + 1e-9 else "beyond"
        c = call(r.station_p8).replace("~h", "").replace(" 8", "")
        flag = "$^{\\ast}$" if c != "marginal" and c != truth else ""
        rows.append("%s & %s & %d & %.1f / %.1f & %s / %s & %.1f / %.1f & %s / %s & %s%s \\\\" % (
            tex(station_name(r.station)), "USCRN" if r.kind == "uscrn" else "ISD", r.years_kept, r.station_T04, r.era5_T04,
            sgn(r.station_x), sgn(r.era5_x), r.station_y, r.era5_y, fmt_h(r.station_h_star), fmt_h(r.era5_h_star), c, flag))
    (TAB / "table_s16.tex").write_text(
        "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\caption{\\textbf{Station records as inputs to the climate ruler.} For each station, the years with complete records"
        " (gaps of up to 6~h filled), and station / ERA5 values over the same years of $T_{0.4}$ (\\textdegree{}C), the"
        " coordinates $x$ and $y$ (K) and the store \\hstar{} (h) computed from each series, and the ruler's call with station"
        " inputs. $^{\\ast}$ wrong confident call against the station series' own store.}\\label{tab:s16}\n"
        "\\begin{tabular}{@{}llrrrrrl@{}}\n\\toprule\nStation & Network & Years & $T_{0.4}$ & $x$ & $y$ & \\hstar{} & Call\\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ figures
def fig_s1():
    s = K.sites()
    rec = cds()
    grid = np.linspace(0, 24, 961)
    fig, axs = K.plt.subplots(1, 2, figsize=(K.FULL, 62 * K.MM), sharey=True)
    for ax, cl, title in ((axs[0], "historical", "1990\u20132025 weather"), (axs[1], "ssp245_median", "SSP2-4.5, circa 2040")):
        h = s.cell_id.map(lambda c: rec[c][cl]).to_numpy(float)
        for basis, lab, col, ls in (("B_full_MW", "B-full (37.3 GW)", K.C["hist"], "-"),
                                    ("B_2027_MW", "B-2027 (24.1 GW)", K.C["green"], (0, (4, 1.5))),
                                    ("B_now_MW", "B-now (13.5 GW)", K.C["purple"], (0, (1, 1)))):
            w = s[basis].to_numpy(float)
            share = np.array([w[h <= g + 1e-9].sum() for g in grid]) / w.sum()
            ax.step(grid, 100 * share, where="post", color=col, ls=ls, lw=1.1, label=lab)
        ax.plot([8, 8], [0, 100], color=K.C["grey"], lw=0.5, ls=":")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.set_xlabel("Store allowed [hours of full IT heat]")
        ax.set_title(title, fontsize=7, pad=2)
    axs[0].set_ylabel("Share of capacity without\ncompressors or water [%]")
    axs[1].legend(loc="lower right", borderaxespad=0.2)
    K.panel_label(axs[0], "A", dx=-0.2)
    K.panel_label(axs[1], "B", dx=-0.08)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.9, wspace=0.08)
    save(fig, "figs1")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    K.plt.close(fig)


def fig_s4():
    b = K.jload("bv_summary.json")["uncorrected"]["historical"]
    t8 = b["tradeoff_F_dry_8h_by_k|B_full_MW"]
    t4 = b["tradeoff_F_dry_4h_by_k|B_full_MW"]
    ks = sorted(t8, key=float)
    fig, ax = K.plt.subplots(figsize=(K.SINGLE, 58 * K.MM))
    ax.plot([float(k) for k in ks], [100 * t8[k] for k in ks], marker="o", ms=3, color=K.C["hist"], label="Store up to 8 h")
    ax.plot([float(k) for k in ks], [100 * t4[k] for k in ks], marker="s", ms=3, color=K.C["stress"], ls=(0, (3, 1.5)),
            label="Store up to 4 h")
    ax.set_xlabel("Dry-cooler size [reference modules per MW]")
    ax.set_ylabel("Share of capacity [%]")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", borderaxespad=0.2)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.2, top=0.96)
    save(fig, "figs4")


def fig_s6():
    ff = K.jload("ff_summary_uncorrected.json")
    fl = ff["flips|B_full_MW"]
    fig, axs = K.plt.subplots(1, 2, figsize=(K.FULL, 60 * K.MM))
    ax = axs[0]
    st = fl["tank_kUSD_per_MWh"]
    xs = sorted([x for x in st if x.replace(".", "", 1).isdigit()], key=float)
    for cl, col, ls in (("historical", K.C["hist"], "-"), ("stress_plus1.2K", K.C["stress"], (0, (3, 1.5)))):
        ys = [100 * st[x][cl]["ban"]["Z3"] if isinstance(st[x].get(cl), dict) else np.nan for x in xs]
        ax.plot([float(x) for x in xs], ys, marker="o", ms=3, color=col, ls=ls,
                label="1990\u20132025" if cl == "historical" else "+1.2 K stress")
    ax.set_xlabel("Store price [$ per kWh of heat]")
    ax.set_ylabel("Banned capacity choosing a chiller [%]")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", borderaxespad=0.2)
    K.panel_label(ax, "A", dx=-0.2)
    ax = axs[1]
    wp = fl["water_mult"]
    xs = sorted([x for x in wp if x.replace(".", "", 1).isdigit()], key=float)
    for z, lab, col, mk, ls in (("Z1", "Dry only", K.C["hist"], "o", "-"), ("Z2", "Pads or towers", K.C["green"], "s", (0, (3, 1.5))),
                                ("Z3", "With chiller", K.C["c2040"], "^", (0, (1, 1)))):
        ys = [100 * wp[x]["historical"]["lc"][z] for x in xs]
        ax.plot([2 * float(x) for x in xs], ys, marker=mk, ms=3, color=col, ls=ls, label=lab)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(["0", "1", "2", "5", "10", "20", "40"])
    ax.xaxis.set_minor_formatter(K.plt.NullFormatter())
    ax.set_xlabel("Water price [$ per m\u00b3]")
    ax.set_ylabel("Route share, water allowed [%]")
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", borderaxespad=0.2)
    K.panel_label(ax, "B", dx=-0.2)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.9, wspace=0.3)
    save(fig, "figs6")


def fig_s7():
    ff = K.jload("ff_summary_uncorrected.json")["fleet"]["B_full_MW"]
    fig, axs = K.plt.subplots(1, 2, figsize=(K.FULL, 60 * K.MM), sharey=True)
    for ax, cl, title in ((axs[0], "historical", "1990\u20132025 weather"), (axs[1], "stress_plus1.2K", "+1.2 K stress")):
        cp = ff[cl]["cphys_GW_th"]
        hs = [0, 2, 4, 8]
        for key, lab, col, mk in (("ban", "No water", "black", "o"), ("w0.01", "\u2264 1 % of tower water", K.C["c2040"], "s"),
                                  ("w0.1", "\u2264 10 % of tower water", K.C["stress"], "D"),
                                  ("w1", "\u2264 tower water", K.C["hist"], "^")):
            ax.plot(hs, [cp[f"cphys_{key}_h{h}"] for h in hs], marker=mk, ms=3.5, color=col, label=lab)
        ax.set_xticks(hs)
        ax.set_xlabel("Store allowed [hours of full IT heat]")
        ax.set_title(title, fontsize=7, pad=2)
    axs[0].set_ylabel("Compressor capacity needed [GW, thermal]")
    axs[1].legend(loc="upper right", borderaxespad=0.2)
    K.panel_label(axs[0], "A", dx=-0.15)
    K.panel_label(axs[1], "B", dx=-0.08)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.9, wspace=0.08)
    save(fig, "figs7")


def fig_s8():
    s = K.sites()
    share = K.jload("sb/sb23_robustness.json")["SB2"]["per_cell_share_feasible"]
    fig = K.plt.figure(figsize=(K.FULL, 80 * K.MM))
    ax = fig.add_axes([0.0, 0.05, 0.84, 0.9])
    K.draw_conus(ax)
    us = s[s.country == "United States"].sort_values("B_full_MW", ascending=False)
    x, y = K.albers(us.lon.to_numpy(), us.lat.to_numpy())
    v = us.cell_id.map(share).to_numpy(float)
    sc = ax.scatter(x, y, s=K.msize(us.B_full_MW.to_numpy()), c=100 * v, cmap="viridis", vmin=0, vmax=100,
                    edgecolors="black", linewidths=0.3, zorder=3)
    cax = fig.add_axes([0.87, 0.2, 0.02, 0.6])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Share of the 33 model fields that\nkeep the site within 8 h [%]", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    save(fig, "figs8")


def fig_s9():
    t = pd.read_csv(OUT / "as2" / "ruler_table.csv")
    fig, axs = K.plt.subplots(1, 2, figsize=(K.FULL, 62 * K.MM))
    for ax, cl, title in ((axs[0], "historical", "1990\u20132025 weather"), (axs[1], "stress_plus1.2K", "+1.2 K stress")):
        g = t[t.climate == cl]
        m = g.iloc[:, 2:].to_numpy(float)
        im = ax.imshow(m, cmap="Blues", vmin=22, vmax=37, origin="upper", aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, "%.1f" % m[i, j], ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > 31 else "black")
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels(["1", "1.5", "2", "2.5", "3"])
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels(["3", "3.5", "4", "4.5", "5"])
        ax.set_xlabel("Plate-exchanger approach [K]")
        ax.set_ylabel("Dry-cooler minimum approach [K]")
        ax.set_title(title, fontsize=7, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
    K.panel_label(axs[0], "A", dx=-0.22)
    K.panel_label(axs[1], "B", dx=-0.22)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.2, top=0.9, wspace=0.4)
    save(fig, "figs9")


def fig_s10():
    oa = K.jload("oos_summary.json")["a"]["margin"]
    ob = K.jload("oos_b_summary.json")["a"]["inner_ladder"]
    al = sorted(oa, key=float)
    fig, ax = K.plt.subplots(figsize=(K.SINGLE, 60 * K.MM))
    ax.plot([float(a) for a in al], [100 * oa[a]["pooled_obs"][0] for a in al], marker="o", ms=3, color=K.C["hist"],
            label="Test 2015\u20132025")
    bl = sorted(ob, key=float)
    ax.plot([float(a) for a in bl], [100 * ob[a]["inner_validation_violation"] for a in bl], marker="s", ms=3,
            color=K.C["c2040"], ls=(0, (3, 1.5)), label="Validation 2010\u20132014")
    ax.axhline(5, color=K.C["grey"], lw=0.5, ls=":")
    ax.set_xlabel("Proportional store margin [multiple of the sized store]")
    ax.set_ylabel("Site-years violating the screen [%]")
    ax.set_ylim(0, 30)
    ax.legend(loc="upper right", borderaxespad=0.2)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.2, top=0.96)
    save(fig, "figs10")


def fig_s12():
    r = K.jload("r2f/stage2_summary.json")["0.05"]["procedures"]
    fams = list(FAMILIES)
    fam_key = {"CAS/IAP": "CAS-IAP"}
    fig, ax = K.plt.subplots(figsize=(K.FULL, 62 * K.MM))
    x = np.arange(len(fams))
    for j, (n, col, mk) in enumerate((("BL2-shift", "black", "o"), ("DR-F", K.C["c2040"], "D"), ("B-QNT", K.C["hist"], "s"),
                                      ("B-ROB", K.C["green"], "^"))):
        vals = [100 * r[n]["V_f"][fam_key.get(f, f)] for f in fams]
        ax.plot(x + (j - 1.5) * 0.12, vals, ls="", marker=mk, ms=4, mfc=col if j != 1 else "white", mec=col,
                label=NAMES[n])
    ax.axhline(5, color=K.C["grey"], lw=0.5, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(fams, rotation=0)
    ax.set_ylabel("Violation when the family is held out [%]")
    ax.set_xlabel("Held-out climate-model family")
    ax.legend(loc="upper left", borderaxespad=0.2, ncol=2)
    ax.set_ylim(0, 30)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.96)
    save(fig, "figs12")


def fig_s13():
    p = pd.read_csv(OUT / "ruler_b" / "points_A.csv")
    fig, ax = K.plt.subplots(figsize=(K.SINGLE, 70 * K.MM))
    for cl, mk in (("historical", "o"), ("stress_plus1.2K", "s"), ("ssp245_median", "D")):
        g = p[p.climate == cl]
        for lab, f, col in (("No store", lambda h: h <= 1e-9, "#0072B2"), ("Within 8 h", lambda h: (h > 1e-9) & (h <= 8 + 1e-9),
                                                                            "#E69F00"),
                            ("Beyond 8 h", lambda h: h > 8 + 1e-9, "#D55E00")):
            m = f(g.h_star.to_numpy(float))
            ax.scatter(g.x[m], g.y_A[m], s=10, marker=mk, c=col, edgecolors="black", linewidths=0.3)
    yy = np.linspace(2, 20, 40)
    ax.plot(3.80 + 0.54 * (yy - 10.05), yy, color="black", lw=0.9)
    ax.set_xlim(-22, 12)
    ax.set_xlabel("0.4 % dry-bulb minus limit temperature [K]")
    ax.set_ylabel("Hottest-month mean daily range [K]")
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.97)
    save(fig, "figs13")


def cell_years(cell):
    """Hourly need and spare of the dry chain at k_max on the Climate Data Store table of a hub cell (1990-2025), the
    yearly minimum stores E*_y (annual reset, store full on 1 January) and the chain's limit temperature at mean pressure."""
    for p in (K.ROOT, K.ROOT / "experiments" / "phase1"):
        sys.path.insert(0, str(p))
    import cds_extract as X
    from bv1_min_storage import need_spare
    from model import design as D
    from model import evaluate as E
    from model import params as PR

    pts = pd.read_csv(OUT / "m2" / "cds_points.csv")
    row = pts[pts.names.str.contains(f"hub_cells:{cell}(?:;|$)", regex=True)].iloc[0]
    W = E.load_weather(X.TAB / f"{row.point}.npz")
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    ch, need, spare, _ = need_spare(P, W, max(D.k_grid(P, lim)), lim)
    ey = np.array([E.sequent_peak(need[y], spare[y])[0] for y in range(need.shape[0])])
    m = D.allowed_violations(P["eta_risk"], need.shape[0])
    grid = np.arange(25.0, 45.0, 0.01)
    pm = float(np.mean(W["p"]))
    t_lim = float(grid[np.argmax(ch.trim_need(1.0, grid, np.full_like(grid, pm)) > 0)])
    return W, need, spare, ey, m, t_lim


def fig_s3():
    W, need, spare, ey, m, t_lim = cell_years("cell_490_321")
    years = np.asarray(W["years"])
    order = np.argsort(ey)[::-1]
    ys = order[m]  # the year that sets h*
    d = np.zeros(need.shape[1])
    k = 0.0
    for h in range(need.shape[1]):  # heat drawn by the energy balance (sequent-peak recursion)
        k = k + need[ys, h] if need[ys, h] > 0 else max(0.0, k - spare[ys, h])
        d[h] = k
    tr = K.jload(f"st/trace_cell_490_321_{int(years[ys])}.json")  # PR-ST series tank, explicit temperatures and flows
    a = {key: np.array([np.nan if x is None else x for x in v], float) for key, v in tr.items()}
    hm = int(np.argmax(d))
    lo, hi = max(0, hm - 8 * 24), min(len(d), hm + 4 * 24)
    t = (np.arange(lo, hi) - lo) / 24.0
    day0 = pd.Timestamp(int(years[ys]), 1, 1) + pd.Timedelta(hours=lo)
    w = slice(lo, hi)
    q = a["q_store"][w]
    dis, rec = np.where(q > 0, q, 0.0), np.where(q < 0, -q, 0.0)
    fig = K.plt.figure(figsize=(K.FULL, 150 * K.MM))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1.0], left=0.095, right=0.985, bottom=0.08, top=0.96, hspace=0.2,
                          wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    tdb = W["t_db"][ys, lo:hi]
    ax.plot(t, tdb, color="black", lw=0.7)
    ax.axhline(t_lim, color=K.C["c2040"], lw=0.8, ls=(0, (4, 2)))
    ax.fill_between(t, t_lim, tdb, where=need[ys, lo:hi] > 0, color=K.C["c2040"], alpha=0.35, lw=0)
    ax.text(0.01, 0.97, "Dashed: chain limit, %.1f °C" % t_lim, transform=ax.transAxes, fontsize=7,
            color=K.C["c2040"], va="top")
    ax.set_ylim(20, 47)
    ax.set_ylabel("Dry-bulb [°C]")
    ax.tick_params(labelbottom=False)
    K.panel_label(ax, "A", dx=-0.13)
    axt = fig.add_subplot(gs[1, 0], sharex=ax)
    axt.plot(t, a["t_chain"][w], color="black", lw=0.8, label="Chain outlet")
    axt.plot(t, a["t_supply"][w], color=K.C["grey"], lw=0.8, ls=(0, (4, 2)), label="Supply (mixing balance)")
    cd = q > 0
    cc = q < 0
    axt.scatter(t[cd], a["t_cold"][w][cd], s=3, color=K.C["hist"], lw=0, label="Cold water drawn", zorder=3)
    axt.scatter(t[cc], a["t_warm"][w][cc], s=3, color=K.C["c2040"], lw=0, label="Warm water displaced", zorder=3)
    axt.set_ylim(29, 58)
    axt.set_yticks([30, 35, 40, 45, 50])
    axt.set_ylabel("Water [°C]")
    axt.legend(loc="upper left", ncol=2, borderaxespad=0.2, handlelength=1.2, columnspacing=1.0, markerscale=2.5)
    axt.tick_params(labelbottom=False)
    K.panel_label(axt, "B", dx=-0.13)
    axp = fig.add_subplot(gs[2, 0], sharex=ax)
    axp.bar(t, dis, width=1 / 24, color=K.C["c2040"], lw=0, label="Discharge")
    axp.bar(t, -rec, width=1 / 24, color=K.C["hist"], lw=0, label="Recharge")
    axp.axhline(0, color="black", lw=0.5)
    axp.set_ylim(-1.1, 1.1)
    axp.set_ylabel("Store power\n[design heat]")
    axp.legend(loc="lower left", ncol=2, borderaxespad=0.2, handlelength=1.0)
    axp.tick_params(labelbottom=False)
    K.panel_label(axp, "C", dx=-0.13)
    axb = fig.add_subplot(gs[3, 0], sharex=ax)
    axb.plot(t, d[lo:hi], color=K.C["hist"], lw=1.0, label="Heat")
    axb.plot(t, a["v_drawn"][w], color="black", lw=0.8, ls=(0, (3, 1.5)), label="Cold water")
    axb.axhline(ey[ys], color=K.C["grey"], lw=0.6, ls=":")
    axb.set_ylabel("Store drawn [h]")
    axb.legend(loc="upper left", ncol=2, borderaxespad=0.2, handlelength=1.6)
    axb.set_xlabel("Days from %s %d (UTC)" % (day0.strftime("%d %B").lstrip("0"), day0.year))
    axb.set_xlim(0, t[-1])
    K.panel_label(axb, "D", dx=-0.13)
    gsr = gs[:, 1].subgridspec(2, 1, hspace=0.25)
    for j, (cell, lab) in enumerate((("cell_490_321", "Abilene, TX"), ("cell_531_368", "Mount Pleasant, WI"))):
        axc = fig.add_subplot(gsr[j, 0])
        e = ey if j == 0 else cell_years(cell)[3]
        o = np.argsort(e)[::-1]
        col = np.array([K.C["lightgrey"]] * len(e), dtype=object)
        col[o[:m]] = "black"
        col[o[m]] = K.C["hist"]
        axc.bar(years, e, color=list(col), width=0.8, lw=0)
        axc.axhline(e[o[m]], color=K.C["hist"], lw=0.6, ls=":")
        axc.set_ylabel("Yearly store [h]")
        axc.set_title(lab, fontsize=7, pad=2)
        axc.set_xlim(1989, 2026)
        if j == 0:
            axc.tick_params(labelbottom=False)
        else:
            axc.set_xlabel("Year")
        K.panel_label(axc, "EF"[j], dx=-0.2)
    save(fig, "figs3")
    dm = np.nanmax(a["v_drawn"][w])
    print("fig_s3: design year %d, h* %.3f h, largest %.3f h in %d, T_lim %.2f C, peak hour %s, cold water drawn at peak %.3f U,"
          " chain outlet max %.2f C, cold water %.2f-%.2f C" % (years[ys], ey[ys], ey[order[0]], years[order[0]], t_lim,
                                                                 str(day0 + pd.Timedelta(hours=hm - lo)), dm,
                                                                 np.nanmax(a["t_chain"][w]), np.nanmin(a["t_cold"][w][cd]),
                                                                 np.nanmax(a["t_cold"][w][cd])))


def fig_s2():
    rec = cds()
    fig, ax = K.plt.subplots(figsize=(K.SINGLE, 78 * K.MM))
    floor = 0.01
    moved = 0
    for cl, col, mk, lab in (("historical", K.C["hist"], "o", "1990–2025"), ("stress_plus1.2K", K.C["stress"], "s", "+1.2 K stress"),
                             ("ssp245_median", K.C["c2040"], "D", "SSP2-4.5, circa 2040")):
        hc, hm = [], []
        for cell in rec:
            if cl == "ssp245_median":
                m = json.loads((OUT / "cmip" / "scen" / f"ssp245_median__{cell}.json").read_text(encoding="utf-8"))["h_star"]
            else:
                m = json.loads((OUT / "bv1" / "uncorrected" / f"{cell}__{cl}.json").read_text(encoding="utf-8"))["k"]["1.4948"]["h_star"]
            hc.append(rec[cell][cl])
            hm.append(m)
        hc, hm = np.asarray(hc), np.asarray(hm)
        moved += int(((hc <= 8 + 1e-9) != (hm <= 8 + 1e-9)).sum())
        ax.scatter(np.maximum(hm, floor), np.maximum(hc, floor), s=9, marker=mk, facecolors="none", edgecolors=col,
                   linewidths=0.7, label=lab)
    ax.plot([floor, 600], [floor, 600], color=K.C["grey"], lw=0.5)
    ax.axvline(8, color=K.C["grey"], lw=0.5, ls=":")
    ax.axhline(8, color=K.C["grey"], lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.01, 0.1, 1, 10, 100]
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(K.matplotlib.ticker.FixedLocator(ticks))
        a.set_major_formatter(K.matplotlib.ticker.FixedFormatter(["0", "0.1", "1", "10", "100"]))
        a.set_minor_formatter(K.matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.007, 700)
    ax.set_ylim(0.007, 700)
    ax.set_xlabel("Store, earlier ERA5 copy [h]")
    ax.set_ylabel("Store, Climate Data Store series [h]")
    ax.legend(loc="upper left", borderaxespad=0.2, handletextpad=0.2)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.15, top=0.97)
    save(fig, "figs2")
    print("fig_s2: points changing side of 8 h:", moved)


def fig_s11():
    s = K.sites()
    drift = {}
    for f in (OUT / "oos_c").glob("cell_*.json"):
        c = json.loads(f.read_text(encoding="utf-8"))
        drift[c["cell"]] = c["delta_K"]["obs"]
    fig = K.plt.figure(figsize=(K.FULL, 80 * K.MM))
    ax = fig.add_axes([0.0, 0.05, 0.84, 0.9])
    K.draw_conus(ax)
    us = s[s.country == "United States"].sort_values("B_full_MW", ascending=False)
    x, y = K.albers(us.lon.to_numpy(), us.lat.to_numpy())
    v = us.cell_id.map(drift).to_numpy(float)
    sc = ax.scatter(x, y, s=K.msize(us.B_full_MW.to_numpy()), c=v, cmap="viridis", vmin=0, vmax=1.5, edgecolors="black",
                    linewidths=0.3, zorder=3)
    cax = fig.add_axes([0.87, 0.2, 0.02, 0.6])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Drift added for 2015–2025 [K]", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    save(fig, "figs11")


def fig_s5():
    d = pd.read_csv(OUT / "ff_sites_uncorrected.csv")
    h = d[d.climate == "historical"]
    fig, axs = K.plt.subplots(1, 3, figsize=(K.FULL, 62 * K.MM), sharey=True)
    for ax, (p, lab, col) in zip(axs, (("lc", "Water allowed", K.C["hist"]), ("ban", "Water ban", K.C["c2040"]),
                                       ("cfdry", "No compressor, no water", K.C["green"]))):
        g = h[h[p + "_viol_hours"] > 0]
        ax.scatter(g[p + "_viol_hours"], g[p + "_viol_unmet_MWh"], s=K.msize(g.B_full_MW.to_numpy()), facecolors="none",
                   edgecolors=col, linewidths=0.7)
        ax.set_yscale("log")
        ax.set_ylim(0.0006, 20)
        ax.set_xlim(-1.5, 50)
        ax.set_title("%s, %d sites" % (lab, len(g)), fontsize=7, pad=2)
        ax.set_xlabel("Hours with unmet duty")
        print("fig_s5 %s: sites %d, fleet unmet %.2f GWh" % (p, len(g), (g[p + "_viol_unmet_MWh"] * g.B_full_MW).sum() / 1e3))
    axs[0].yaxis.set_major_locator(K.matplotlib.ticker.FixedLocator([0.001, 0.01, 0.1, 1, 10]))
    axs[0].yaxis.set_major_formatter(K.matplotlib.ticker.FixedFormatter(["0.001", "0.01", "0.1", "1", "10"]))
    axs[0].yaxis.set_minor_formatter(K.matplotlib.ticker.NullFormatter())
    axs[0].set_ylabel("Unmet heat [h of full IT heat]")
    for j, ax in enumerate(axs):
        K.panel_label(ax, "ABC"[j], dx=-0.16 if j == 0 else -0.08)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.92, wspace=0.12)
    save(fig, "figs5")


def main():
    K.style()
    table_s1()
    table_s6()
    table_s8()
    table_s12()
    table_s11()
    table_s13()
    table_s15()
    table_s16()
    table_s5()
    table_s17()
    table_s10()
    for f in (fig_s1, fig_s4, fig_s6, fig_s7, fig_s8, fig_s9, fig_s10, fig_s12, fig_s13, fig_s3, fig_s2, fig_s11, fig_s5):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # report and continue; every failure is fixed before the SI compiles
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
