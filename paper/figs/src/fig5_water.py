"""Figure 5 — water, compressors and the price of a water ban (headline D, C4).

Artifacts: experiments/phase1/out/ff_summary_uncorrected.json and ff_summary_cmip_ssp245_median.json (PR-FF fleet totals:
least-cost designs with water allowed, water-ban designs, the cooling-tower counterfactual; envelope-minimum
compressors), cost/cost_regimes.json (PR-COST-b cases: dry-cooler price multiplier, tower share, $ per m3),
sb/sb5_mechanism.json (PR-SB SB-5: 243 cost cases, the ban's added cost and water saved). Water references for scale:
delivered industrial water $0.84-1.72/m3 (PNNL-34006, 2024 $) and seawater reverse osmosis $0.64-1.62/m3 (World Bank 2019).
"""
import re

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

import common as K

CLIM = [("uncorrected|historical", "1990–2025 weather", K.C["hist"], "-"),
        ("cmip_ssp245_median|ssp245_median", "SSP2-4.5, circa 2040", K.C["c2040"], "-"),
        ("uncorrected|stress_plus1.2K", "+1.2 K stress", K.C["stress"], (0, (3, 1.5)))]
REG = {"adiabatic": ("o", "Adiabatic regime"), "mixed": ("s", "Mixed transition"), "tower": ("^", "Tower regime")}
CENTRAL = {"uncorrected|historical": 17.70, "uncorrected|stress_plus1.2K": 18.03, "cmip_ssp245_median|ssp245_median": 18.04}


def fleet():
    u = K.jload("ff_summary_uncorrected.json")["fleet"]["B_full_MW"]
    c = K.jload("ff_summary_cmip_ssp245_median.json")["fleet"]["B_full_MW"]
    return {"historical": u["historical"], "stress_plus1.2K": u["stress_plus1.2K"], "ssp245_median": c["ssp245_median"]}


def main():
    K.style()
    fl = fleet()
    fig = K.plt.figure(figsize=(K.FULL, 122 * K.MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], left=0.075, right=0.985, bottom=0.085, top=0.955,
                          wspace=0.27, hspace=0.45)

    # (A) water and cost of the three route choices, 1990-2025 weather
    ax = fig.add_subplot(gs[0, 0])
    h = fl["historical"]
    routes = [("Least-cost,\nwater allowed", h["lc_scale"], "#009E73"),
              ("Least-cost,\nwater banned", h["ban_scale"], "#4D4D4D"),
              ("Cooling\ntowers", h["tower_scale"], K.C["sky"])]
    xs = np.arange(len(routes))
    for i, (lab, sc, col) in enumerate(routes):
        ax.bar(i, sc["water_GL_yr"], width=0.62, color=col, lw=0)
        ax.text(i, sc["water_GL_yr"] + 12, "%.0f GL\n$%.2f bn/yr" % (sc["water_GL_yr"], sc["cost_BUSD_yr"]),
                ha="center", va="bottom", fontsize=7)
    ax.plot([-0.45, 2.45], [117.6, 117.6], color="black", lw=0.6, ls=(0, (3, 2)))
    ax.text(-0.45, 124, "US fleet-average water intensity\napplied to the same load", fontsize=7, va="bottom", ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in routes])
    ax.set_ylim(0, 620)
    ax.set_ylabel("On-site water use [GL per year]")
    K.panel_label(ax, "A", dx=-0.2)

    # (B) compressors under a water ban
    axb = fig.add_subplot(gs[0, 1])
    lab = [("historical", "1990–\n2025"), ("stress_plus1.2K", "+1.2 K\nstress"), ("ssp245_median", "SSP2-4.5,\n2040")]
    phys = [fl[k]["cphys_GW_th"]["cphys_ban_h8"] for k, _ in lab]
    inst = [fl[k]["ban_scale"]["chiller_GW_th"] for k, _ in lab]
    x = np.arange(3)
    axb.bar(x, phys, width=0.55, color="black", lw=0, label="Minimum in envelope")
    axb.bar(x, np.array(inst) - np.array(phys), bottom=phys, width=0.55, color="white", edgecolor="black", lw=0.6,
            label="Chosen for cost")
    for i in range(3):
        axb.text(i + 0.31, phys[i] / 2, "%.2f" % phys[i], fontsize=7, va="center", ha="left")
        axb.text(i, inst[i] + 0.12, "%.1f" % inst[i], fontsize=7, va="bottom", ha="center")
    axb.set_xticks(x)
    axb.set_xticklabels([l for _, l in lab])
    axb.set_xlim(-0.6, 3.1)
    axb.set_ylim(0, 6)
    axb.set_ylabel("Compressor capacity under a ban [GW, thermal]")
    axb.legend(loc="upper left", borderaxespad=0.1)
    K.panel_label(axb, "B", dx=-0.13)

    # (C) the ban's price per m3 against the dry-cooler price
    axc = fig.add_subplot(gs[1, 0])
    cases = K.jload("cost/cost_regimes.json")["cases"]
    axc.axhspan(0.64, 1.72, color=K.C["lightgrey"], lw=0)
    axc.text(0.52, 0.7, "Delivered water or desalination, $0.6–1.7/m³", fontsize=7, va="bottom", ha="left")
    axc.plot([1, 1], [6.5, 24], color="black", lw=0.6)
    for menu, lab_, col, ls in CLIM:
        pts = {1.0: (CENTRAL[menu], "adiabatic")}
        for c in cases:
            m = re.fullmatch(r"dc([0-9.]+)_pads_abs", c["case"])
            if c["source"] == "PR-COST-b" and c["menu"] == menu and m:
                pts[float(m.group(1))] = (c["USD_per_m3"], c["regime"])
        xx = sorted(pts)
        axc.plot(xx, [pts[v][0] for v in xx], color=col, ls=ls, lw=0.9, label=lab_, zorder=2)
        for v in xx:
            mk = REG[pts[v][1]][0]
            axc.plot([v], [pts[v][0]], marker=mk, ms=3.6, mfc=col, mec="black", mew=0.3, ls="", zorder=3)
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlim(0.45, 11)
    axc.set_ylim(0.5, 40)
    axc.set_xticks([0.5, 1, 2, 3, 4, 6, 10])
    axc.set_xticklabels(["0.5", "1", "2", "3", "4", "6", "10"])
    axc.set_yticks([1, 2, 5, 10, 20])
    axc.set_yticklabels(["1", "2", "5", "10", "20"])
    axc.xaxis.set_minor_formatter(NullFormatter())
    axc.yaxis.set_minor_formatter(NullFormatter())
    axc.set_xlabel("Dry-cooler price [multiple of the central $120 per kW]")
    axc.set_ylabel("Cost of the ban [$ per m³ of water saved]")
    axc.legend(loc="upper right", borderaxespad=0.1, handlelength=2.0)
    K.panel_label(axc, "C", dx=-0.2)

    # (D) numerator and denominator (243 cost cases)
    axd = fig.add_subplot(gs[1, 1])
    mech = K.jload("sb/sb5_mechanism.json")["cases"]
    for p, tx in ((1, 640), (3, 640), (10, 265), (30, 86)):
        x0, x1 = max(10, 100 / p), min(700, 3000 / p)
        xs_ = np.logspace(np.log10(x0), np.log10(x1), 40)
        axd.plot(xs_, p * xs_, color=K.C["lightgrey"], lw=0.6, zorder=0)
        axd.text(tx, p * tx * 1.12, "$%d" % p, fontsize=7, color=K.C["grey"], ha="right", va="bottom")
    colmap = {"uncorrected|historical": K.C["hist"], "uncorrected|stress_plus1.2K": K.C["stress"],
              "cmip_ssp245_median|ssp245_median": K.C["c2040"]}
    for c in mech:
        mk = REG[c["regime"]][0]
        axd.plot([c["water_saved_GL_yr"]], [c["ban_added_cost_MUSD_yr"]], marker=mk, ms=3.0, ls="", mfc="none",
                 mec=colmap[c["menu"]], mew=0.6, zorder=2)
    axd.set_xscale("log")
    axd.set_yscale("log")
    axd.set_xlim(10, 700)
    axd.set_ylim(100, 3000)
    axd.set_xticks([10, 20, 50, 100, 200, 500])
    axd.set_xticklabels(["10", "20", "50", "100", "200", "500"])
    axd.set_yticks([100, 200, 500, 1000, 2000])
    axd.set_yticklabels(["100", "200", "500", "1,000", "2,000"])
    axd.xaxis.set_minor_formatter(NullFormatter())
    axd.yaxis.set_minor_formatter(NullFormatter())
    axd.set_xlabel("Water saved by the ban [GL per year]")
    axd.set_ylabel("Added cost of the ban [$ million per year]")
    hs = [Line2D([], [], ls="", marker=mk, mfc="none", mec="black", mew=0.6, ms=4, label=l) for mk, l in REG.values()]
    axd.legend(handles=hs, loc="lower right", borderaxespad=0.1, handletextpad=0.2)
    K.panel_label(axd, "D", dx=-0.16)
    K.save(fig, "fig5")


if __name__ == "__main__":
    main()
