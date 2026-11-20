"""Fig. 5 (Applied Energy) -- sizing for warming during the plant's life (supporting design tool, Section 4.5).

Artifacts: experiments/phase1/out/oos_b_summary.json (training-only least-cost designs), oos_c_summary.json (drift-aware
sizing), per-cell oos_b/ and oos_c/ records (paired water-free stores). The held-out climate-model-family test is in
the Supplementary material.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

import common as K
from drift_helpers import paired_store


def main():
    K.style()
    ob = K.jload("oos_b_summary.json")
    oc = K.jload("oos_c_summary.json")
    ps = paired_store()
    sc2 = K.jload("ae/sc2_finite_sample.json")
    fig = K.plt.figure(figsize=(K.ONEHALF, 58 * K.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42], left=0.235, right=0.99, bottom=0.2, top=0.97, wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    rows = [("Water-free store,\nsites it keeps", ps["obs"][0], ps["scen"][0], ps["obs"][1], ps["scen"][1]),
            ("Least-cost,\nwater allowed", ob["b"]["obs"]["pooled_weighted"], ob["b"]["scen"]["pooled_weighted"],
             oc["b"]["obs"]["pooled_weighted"], oc["b"]["scen"]["pooled_weighted"]),
            ("Least-cost,\nwater banned", ob["c"]["obs"]["pooled_weighted"], ob["c"]["scen"]["pooled_weighted"],
             oc["c"]["obs"]["pooled_weighted"], oc["c"]["scen"]["pooled_weighted"])]
    ax.axvline(5, color=K.C["grey"], lw=0.6, ls=":")
    for i, (lab, s_obs, s_40, d_obs, d_40) in enumerate(rows):
        y = 2 - i
        for off, s, d, col, ls in ((0.14, s_obs, d_obs, K.C["hist"], "-"), (-0.14, s_40, d_40, K.C["c2040"], (0, (3, 1.5)))):
            if s is not None:
                ax.plot([100 * d, 100 * s], [y + off, y + off], color=col, lw=0.8, ls=ls, zorder=1)
                ax.plot([100 * s], [y + off], marker="o", ms=4, mfc="white", mec=col, mew=0.9, ls="", zorder=2)
            ax.plot([100 * d], [y + off], marker="o", ms=4, mfc=col, mec=col, ls="", zorder=3)
            print(lab.replace("\n", " "), "off", off, "stationary", s, "drift", d)
        if i == 0:  # SC-2: stores sized on the largest training year, same sites (finite-sample stationary reference)
            for off, v, col in ((0.14, sc2["largest"]["obs_drift_cohort"]["pooled_weighted"], K.C["hist"]),
                                (-0.14, sc2["largest"]["scen_drift_cohort"]["pooled_weighted"], K.C["c2040"])):
                ax.plot([100 * v], [y + off], marker="s", ms=3.6, mfc="white", mec=col, mew=0.9, ls="", zorder=2)
                print(lab.replace("\n", " "), "off", off, "largest", v)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.55, 2.55)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 60)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Site-years with a coolant-limit violation [%]")
    hs = [Line2D([], [], ls="", marker="o", mfc="white", mec="black", mew=0.9, ms=4, label="Second-largest\ntraining year"),
          Line2D([], [], ls="", marker="s", mfc="white", mec="black", mew=0.9, ms=3.6, label="Largest\ntraining year"),
          Line2D([], [], ls="", marker="o", mfc="black", mec="black", ms=4, label="Plus warming margin"),
          Line2D([], [], color=K.C["hist"], lw=1.5, label="Tested 2015–2025"),
          Line2D([], [], color=K.C["c2040"], lw=1.5, ls=(0, (3, 1.5)), label="Tested around 2040")]
    axl = fig.add_subplot(gs[0, 1])
    axl.axis("off")
    axl.legend(handles=hs, loc="center left", bbox_to_anchor=(0.05, 0.5), borderaxespad=0.0, handletextpad=0.3,
               labelspacing=0.5)
    K.save(fig, "fig5")


if __name__ == "__main__":
    main()
