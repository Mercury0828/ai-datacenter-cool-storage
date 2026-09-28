"""Figure 6 — sizing cooling capacity for climatic drift (C5, C6).

Artifacts: experiments/phase1/out/oos_summary.json (PR-OOS (a): stores sized on 1990-2014), oos_b_summary.json (PR-OOS-b:
training-only least-cost designs), oos_c_summary.json (PR-OOS-c: drift-aware sizing), r2f/stage2_summary.json (PR-R2F:
held-out model-family test at the 5 % budget; pooled risk, cost, 90 % cluster-bootstrap risk intervals).
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

import common as K

NAMES = [("B-MON-q0.5", "Monthly quantile field, 0.5"), ("BL1-shift", "Design day + margin"),
         ("B-MON-q0.75", "Monthly quantile field, 0.75"), ("B-SAA", "Ensemble average"),
         ("DR-F", "Robust over model fields"), ("BL2-shift", "Uniform margin"), ("BL3", "Margin, no violating year"),
         ("B-QNT", "Ensemble quantile"), ("B-SCAL", "Summer-warming quantile"), ("B-ROB", "Worst training field"),
         ("B-MON-q0.9", "Monthly quantile field, 0.9"), ("B-MON-q1", "Monthly quantile field, 1")]


def paired_store():
    """Stationary and drift-aware water-free stores on the same sites: those the drift-aware store keeps within 8 h
    (per-cell records of PR-OOS-b (a) and PR-OOS-c (a)); pooled, capacity-weighted violation frequencies."""
    import json
    s = K.pd.read_csv(K.ROOT / "data" / "sites" / "hub_sites.csv")
    acc = {"obs": [0.0, 0.0, 0.0], "scen": [0.0, 0.0, 0.0]}
    for _, r in s.iterrows():
        b = json.loads((K.OUT / "oos_b" / f"{r.cell_id}.json").read_text(encoding="utf-8"))["a"]
        c = json.loads((K.OUT / "oos_c" / f"{r.cell_id}.json").read_text(encoding="utf-8"))["a"]
        for key, e, n in (("obs", np.asarray(b["E_hist"])[25:], 11), ("scen", np.asarray(b["E_scen"]), 36)):
            if c[key]["h_drift"] <= 8 + 1e-9:
                acc[key][0] += r.B_full_MW * (e > b["h_train"] + 1e-9).sum() / n
                acc[key][1] += r.B_full_MW * sum(c[key]["viol"]) / n
                acc[key][2] += r.B_full_MW
    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in acc.items()}


def main():
    K.style()
    ob = K.jload("oos_b_summary.json")
    oc = K.jload("oos_c_summary.json")
    ps = paired_store()
    r2 = K.jload("r2f/stage2_summary.json")["0.05"]
    fig = K.plt.figure(figsize=(K.FULL, 112 * K.MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.16, 0.34], height_ratios=[0.62, 1.0], left=0.2, right=0.99,
                          bottom=0.085, top=0.955, wspace=0.08, hspace=0.42)

    # (A) fleet transfer: designs sized on 1990-2014, tested on later weather
    ax = fig.add_subplot(gs[0, 0])
    rows = [("Water-free store,\nsites it keeps", ps["obs"][0], ps["scen"][0], ps["obs"][1], ps["scen"][1]),
            ("Least-cost,\nwater allowed", ob["b"]["obs"]["pooled_weighted"], ob["b"]["scen"]["pooled_weighted"],
             oc["b"]["obs"]["pooled_weighted"], oc["b"]["scen"]["pooled_weighted"]),
            ("Least-cost,\nwater banned", ob["c"]["obs"]["pooled_weighted"], ob["c"]["scen"]["pooled_weighted"],
             oc["c"]["obs"]["pooled_weighted"], oc["c"]["scen"]["pooled_weighted"])]
    ax.axvline(5, color=K.C["grey"], lw=0.6, ls=":")
    for i, (lab, s_obs, s_40, d_obs, d_40) in enumerate(rows):
        y = 2 - i
        for off, s, d, col in ((0.14, s_obs, d_obs, K.C["hist"]), (-0.14, s_40, d_40, K.C["c2040"])):
            if s is not None:
                ax.plot([100 * d, 100 * s], [y + off, y + off], color=col, lw=0.8, zorder=1)
                ax.plot([100 * s], [y + off], marker="o", ms=4, mfc="white", mec=col, mew=0.9, ls="", zorder=2)
            ax.plot([100 * d], [y + off], marker="o", ms=4, mfc=col, mec=col, ls="", zorder=3)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.55, 2.55)
    ax.set_xscale("log")
    ax.set_xlim(0.9, 60)
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Site-years violating the screen [%]")
    hs = [Line2D([], [], ls="", marker="o", mfc="white", mec="black", mew=0.9, ms=4, label="Sized on 1990–2014"),
          Line2D([], [], ls="", marker="o", mfc="black", mec="black", ms=4, label="Plus warming margin"),
          Line2D([], [], color=K.C["hist"], lw=1.5, label="Tested 2015–2025"),
          Line2D([], [], color=K.C["c2040"], lw=1.5, label="Tested circa 2040")]
    axl = fig.add_subplot(gs[0, 1:])
    axl.axis("off")
    axl.legend(handles=hs, loc="center left", bbox_to_anchor=(0.1, 0.5), borderaxespad=0.0, handletextpad=0.3,
               labelspacing=0.5)
    K.panel_label(ax, "A", dx=-0.27, dy=1.0)

    # (B) held-out climate-model families, 5 % budget
    axb = fig.add_subplot(gs[1, 0])
    P = r2["procedures"]
    ri = r2["bootstrap"]["risk_interval_90"]
    order = sorted([n for n, _ in NAMES], key=lambda n: P[n]["K"])
    lab = dict(NAMES)
    axb.axvline(5, color=K.C["grey"], lw=0.6, ls=":")
    for j, n in enumerate(order):
        y = len(order) - 1 - j
        v = 100 * P[n]["V"]
        lo, hi = 100 * ri[n][0], 100 * ri[n][1]
        comp = P[n]["V"] <= 0.05
        col = K.C["c2040"] if n == "DR-F" else ("black" if comp else K.C["grey"])
        axb.plot([lo, hi], [y, y], color=col, lw=0.8)
        axb.plot([v], [y], marker="D" if n == "DR-F" else "o", ms=3.8, mfc=col if comp else "white", mec=col, mew=0.8,
                 ls="")
    axb.set_yticks(range(len(order)))
    axb.set_yticklabels([lab[n] for n in order[::-1]])
    axb.set_ylim(-0.6, len(order) - 0.4)
    axb.set_xscale("log")
    axb.set_xlim(0.4, 30)
    axb.set_xticks([0.5, 1, 2, 5, 10, 20])
    axb.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    axb.xaxis.set_minor_formatter(NullFormatter())
    axb.set_xlabel("Held-out violation [%], 90 % interval")
    K.panel_label(axb, "B", dx=-0.27, dy=1.0)

    axc = fig.add_subplot(gs[1, 1])
    axc.axis("off")
    axc.set_ylim(-0.6, len(order) - 0.4)
    axc.set_xlim(0, 1)
    axc.text(0.0, len(order) - 0.1, "$ per MW-yr", fontsize=7, va="bottom", ha="left")
    for j, n in enumerate(order):
        y = len(order) - 1 - j
        axc.text(0.0, y, "{:,.0f}".format(P[n]["K"]), fontsize=7, va="center", ha="left")
    K.save(fig, "fig6")


if __name__ == "__main__":
    main()
