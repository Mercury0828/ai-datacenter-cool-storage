"""Figure 3 — warming around 2040, the spread across climate models, and the priced 1 K design lever.

Artifacts: experiments/phase1/out/sb/sb23_robustness.json (PR-SB SB-2/SB-3: 33 SSP2-4.5 model fields, approach
variants, robustness classes; CDS data of record), sb/sb4_price1k.json (SB-4: break-even of the 1 K lever against the
longer store), cmip/cmip_summary.json scenario values rebuilt on CDS (C2), docs/cost_basis.md (SB-6: derived plate-HX
purchase premium $28.9/kW_th, $23-38 across U = 3-5 kW m-2 K-1).
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

import common as K

HIST, S12, C40 = K.C["hist"], K.C["stress"], K.C["c2040"]
CLS = [("core", "Robust core, 22.5 GW", "#0072B2", "o"),
       ("sensitive", "Assumption-sensitive, 13.2 GW", "#E69F00", "D"),
       ("out", "Infeasible in every test, 1.6 GW", "#D55E00", "^")]
SHORT = {"cell_485_331": "Temple", "cell_487_293": "Stargate New Mexico", "cell_490_321": "Abilene (3 sites)",
         "cell_490_332": "Midlothian", "cell_490_333": "Red Oak", "cell_493_331": "Denton",
         "cell_495_317": "Helios", "cell_505_339": "Pryor"}


def main():
    K.style()
    sb = K.jload("sb/sb23_robustness.json")
    pr = K.jload("sb/sb4_price1k.json")
    fig = K.plt.figure(figsize=(K.FULL, 118 * K.MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], left=0.075, right=0.985, bottom=0.075, top=0.955,
                          wspace=0.30, hspace=0.45)

    # (A) climate fields
    ax = fig.add_subplot(gs[0, 0])
    vals = np.array(sorted(sb["SB2"]["F_dry_8h_GW_per_model"].values()))
    rng = np.random.default_rng(1)
    jit = rng.uniform(-0.16, 0.16, len(vals))
    q10, q50, q90 = sb["SB2"]["family_balanced_q10_q50_q90"]
    xs = {"hist": 0, "s12": 1, "models": 2.3, "p10": 3.5, "p90": 4.3}
    ax.plot([xs["hist"]], [34.29], marker="o", ms=5, color=HIST, ls="")
    ax.plot([xs["s12"]], [28.19], marker="s", ms=4.5, color=S12, ls="")
    ax.add_patch(K.plt.Rectangle((xs["models"] - 0.28, q10), 0.56, q90 - q10, fc=C40, alpha=0.18, lw=0))
    ax.scatter(xs["models"] + jit, vals, s=7, c="white", edgecolors=C40, linewidths=0.6, zorder=3)
    ax.plot([xs["models"] - 0.28, xs["models"] + 0.28], [29.05, 29.05], color=C40, lw=1.2)
    ax.plot([xs["p10"]], [34.14], marker="v", ms=4.5, color=C40, ls="")
    ax.plot([xs["p90"]], [26.22], marker="^", ms=4.5, color="#8B2E00", ls="")
    ax.set_xticks([xs[k] for k in ("hist", "s12", "models", "p10", "p90")])
    ax.set_xticklabels(["1990–\n2025", "+1.2 K\nstress", "33 model\nfields", "p10\nfield", "SSP5-8.5\np90"])
    ax.set_xlim(-0.5, 4.8)
    ax.set_ylim(20, 38)
    ax.set_yticks([20, 25, 30, 35])
    ax.set_ylabel("Capacity within 8 h of storage [GW]")
    ax.text(xs["models"] + 0.34, 29.05, "median\nfield", fontsize=7, va="center", color=C40)
    ax.text(-0.35, 21.0, "SSP2-4.5, circa 2040, unless labelled", fontsize=7, color=K.C["grey"])
    K.panel_label(ax, "A", dx=-0.2)

    # (B) total design approach
    axb = fig.add_subplot(gs[0, 1])
    series = [("historical", "1990–2025 weather", HIST, "o", "-"),
              ("ssp245_median", "SSP2-4.5, circa 2040", C40, "D", "-"),
              ("stress_plus1.2K", "+1.2 K stress", S12, "s", (0, (3, 1.5)))]
    for cl, lab, col, mk, ls in series:
        d = sb["SB3"][cl]["F_dry_8h_GW"]
        ks = sorted(d, key=lambda k: float(k[:-1]))
        axb.plot([float(k[:-1]) for k in ks], [d[k] for k in ks], color=col, marker=mk, ms=3.5, ls=ls, lw=1.0,
                 label=lab)
    axb.text(5.07, 34.5, "34.1", fontsize=7, color=C40, va="bottom", ha="left")
    axb.text(6.0, 29.55, "29.1", fontsize=7, color=C40, va="bottom", ha="center")
    axb.text(6.12, 36.9, "Around 2040, one kelvin less\napproach restores 5.1 GW", fontsize=7, ha="left",
             va="center")
    axb.set_xlim(4.8, 7.2)
    axb.set_xticks([5, 5.5, 6, 6.5, 7])
    axb.set_xticklabels(["5", "5.5", "6\n(base)", "6.5", "7"])
    axb.set_ylim(24, 38)
    axb.set_xlabel("Total design approach [K]")
    axb.set_ylabel("Capacity within 8 h of storage [GW]")
    axb.legend(loc="lower left", borderaxespad=0.2, handlelength=2.2)
    K.panel_label(axb, "B", dx=-0.13)

    # (C) robustness classes around 2040 against the store each site needs with 1990-2025 weather (all 93 sites);
    # the site map of the store classes is Figure 2B, so this panel shows only how the two classifications relate
    s = K.site_hstar("historical")
    core = set(sb["joint_2040"]["robust_core_cells"])
    out = set(sb["joint_2040"]["firmly_out_cells"])
    s["cls"] = s.cell_id.map(lambda c: "core" if c in core else ("out" if c in out else "sensitive"))
    rows = [("None", lambda h: h <= 1e-9), ("0–2 h", lambda h: (h > 1e-9) & (h <= 2 + 1e-9)),
            ("2–8 h", lambda h: (h > 2 + 1e-9) & (h <= 8 + 1e-9)), ("> 8 h", lambda h: h > 8 + 1e-9)]
    axm = fig.add_subplot(gs[1, 0])
    yy = np.arange(len(rows))[::-1]
    for (lab, f), yv in zip(rows, yy):
        g = s[f(s.h_star.to_numpy(float))]
        left = 0.0
        for key, _, col, _ in CLS:
            w = g.loc[g.cls == key, "B_full_MW"].sum() / 1e3
            if w > 0:
                axm.barh(yv, w, left=left, height=0.62, color=col, edgecolor="white", linewidth=0.6, zorder=2)
            left += w
        axm.text(left + 0.3, yv, "%.1f GW" % left, va="center", ha="left", fontsize=7)
    axm.set_yticks(yy)
    axm.set_yticklabels([r[0] for r in rows])
    axm.set_ylabel("Store needed, 1990–2025")
    axm.set_xlabel("IT capacity by class around 2040 [GW]")
    axm.set_xlim(0, 25)
    axm.set_ylim(-0.6, len(rows) - 0.1)
    hs = [Patch(facecolor=col, edgecolor="white", label=lab) for _, lab, col, _ in CLS]
    axm.legend(handles=hs, loc="upper right", borderaxespad=0.2, handlelength=1.2, labelspacing=0.3)
    K.panel_label(axm, "C", dx=-0.2, dy=1.02)

    # (D) the priced lever
    axd = fig.add_subplot(gs[1, 1])
    cells = sorted(pr["ssp245_median"]["cells"], key=lambda c: c["B_storage_capex_per_kW"])
    yy = np.arange(len(cells))
    axd.axvspan(23, 38, color=K.C["lightgrey"], lw=0)
    axd.plot([28.9, 28.9], [-0.7, len(cells) - 0.3], color="black", lw=0.8)
    axd.text(29.8, len(cells) - 0.55, "Purchase premium of the\nlarger heat exchanger,\n$28.9/kW (23–38)",
             fontsize=7, va="top", ha="left")
    for i, c in enumerate(cells):
        axd.scatter([c["B_storage_capex_per_kW"]], [i], s=6 + c["MW"] * 0.05, c=C40, edgecolors="black",
                    linewidths=0.35, zorder=3)
    axd.set_yticks(yy)
    axd.set_yticklabels([SHORT[c["cell"]] for c in cells])
    axd.set_xscale("log")
    axd.set_xlim(15, 600)
    axd.set_xticks([20, 50, 100, 200, 500])
    axd.set_xticklabels(["20", "50", "100", "200", "500"])
    axd.xaxis.set_minor_formatter(NullFormatter())
    axd.set_ylim(-0.7, len(cells) - 0.3)
    axd.set_xlabel("Break-even premium against a longer store [$ per kW of IT]")
    K.panel_label(axd, "D", dx=-0.36)
    K.save(fig, "fig3")


if __name__ == "__main__":
    main()
