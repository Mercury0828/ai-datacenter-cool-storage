"""Figure 4 — two weather statistics place a site on the 8 h storage boundary (the climate ruler, C7).

Artifacts: experiments/phase1/out/ruler/points.csv (80 hub cells x 3 climates; x = 0.4 % dry-bulb minus the chain's
limit temperature, y = mean daily range on days reaching the 0.4 % dry-bulb; h* at k_max), ruler/external.csv (15 ERA5
cells outside the frame x 2 climates, never fitted), ruler/ruler_summary.json (fitted logistic models, PR-RULER).
"""
import numpy as np
from matplotlib.lines import Line2D

import common as K

CLIM = {"historical": ("o", "1990–2025"), "stress_plus1.2K": ("s", "+1.2 K stress"),
        "ssp245_median": ("D", "SSP2-4.5, circa 2040")}
CLS = [("No store", lambda h: h <= 1e-9, "#0072B2"), ("Within 8 h", lambda h: (h > 1e-9) & (h <= 8 + 1e-9), "#E69F00"),
       ("Beyond 8 h", lambda h: h > 8 + 1e-9, "#D55E00")]


def contour_x(model, p, y):
    b, mu, sd = model["b"], model["mu"], model["sd"]
    lg = np.log(p / (1 - p))
    return mu[0] + sd[0] / b[1] * (lg - b[0] - b[2] * (y - mu[1]) / sd[1])


def draw(ax, df, rs, hollow=False):
    yy = np.linspace(3.5, 21.5, 50)
    m8, m0 = rs["h8"]["model_all_points"], rs["h0"]["model_all_points"]
    ax.fill_betweenx(yy, contour_x(m8, 0.9, yy), contour_x(m8, 0.1, yy), color=K.C["lightgrey"], lw=0, zorder=0)
    ax.plot(contour_x(m8, 0.5, yy), yy, color="black", lw=0.9, zorder=1)
    ax.plot(contour_x(m0, 0.5, yy), yy, color="black", lw=0.9, ls=(0, (4, 2)), zorder=1)
    for cl, (mk, _) in CLIM.items():
        g = df[df.climate == cl]
        for _, f, col in CLS:
            m = f(g.h_star.to_numpy(float))
            if hollow:
                ax.scatter(g.x[m], g.y[m], s=16, marker=mk, facecolors="white", edgecolors=col, linewidths=0.9, zorder=3)
            else:
                ax.scatter(g.x[m], g.y[m], s=12, marker=mk, c=col, edgecolors="black", linewidths=0.3, zorder=3)
    if "cell" in df:
        ring = df[df.cell.isin(["cell_490_321", "cell_531_368"])]
        ax.scatter(ring.x, ring.y, s=70, facecolors="none", edgecolors="black", linewidths=1.0, zorder=4)
    ax.set_xlim(-22, 12)
    ax.set_ylim(3.5, 21.5)
    ax.set_xlabel("0.4 % dry-bulb minus chain limit temperature [K]")


def main():
    K.style()
    rs = K.jload("ruler/ruler_summary.json")
    pts = K.pd.read_csv(K.OUT / "ruler" / "points.csv")
    ext = K.pd.read_csv(K.OUT / "ruler" / "external.csv")
    fig = K.plt.figure(figsize=(K.FULL, 78 * K.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.66], left=0.065, right=0.995, bottom=0.14, top=0.93,
                          wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    draw(ax, pts, rs)
    ax.set_ylabel("Daily range on those hot days [K]")
    ax.set_title("80 tracked grid cells, three climates", fontsize=7, pad=2)
    K.panel_label(ax, "A", dx=-0.17, dy=1.0)
    axb = fig.add_subplot(gs[0, 1])
    draw(axb, ext, rs, hollow=True)
    axb.set_title("15 grid cells outside the fitted set", fontsize=7, pad=2)
    K.panel_label(axb, "B", dx=-0.1, dy=1.0)
    axl = fig.add_subplot(gs[0, 2])
    axl.axis("off")
    h1 = [Line2D([], [], ls="", marker="o", mfc=col, mec="black", mew=0.3, ms=4.5, label=lab) for lab, _, col in CLS]
    l1 = axl.legend(handles=h1, title="Store needed", loc="upper left", bbox_to_anchor=(-0.05, 1.0), borderaxespad=0,
                    handletextpad=0.2, labelspacing=0.35, title_fontsize=7)
    l1._legend_box.align = "left"
    axl.add_artist(l1)
    h2 = [Line2D([], [], ls="", marker=mk, mfc="white", mec="black", mew=0.5, ms=4.5, label=lab)
          for mk, lab in CLIM.values()]
    l2 = axl.legend(handles=h2, title="Climate", loc="center left", bbox_to_anchor=(-0.05, 0.5), borderaxespad=0,
                    handletextpad=0.2, labelspacing=0.35, title_fontsize=7)
    l2._legend_box.align = "left"
    axl.add_artist(l2)
    h3 = [Line2D([], [], color="black", lw=0.9, label="8 h boundary, p = 0.5"),
          Line2D([], [], color=K.C["lightgrey"], lw=5, label="Marginal band,\np = 0.1–0.9"),
          Line2D([], [], color="black", lw=0.9, ls=(0, (4, 2)), label="No-store boundary,\np = 0.5"),
          Line2D([], [], ls="", marker="o", mfc="none", mec="black", mew=1.0, ms=7,
                 label="Abilene (right) and\nMount Pleasant (left)")]
    l3 = axl.legend(handles=h3, loc="lower left", bbox_to_anchor=(-0.05, 0.0), borderaxespad=0, handlelength=1.8,
                    labelspacing=0.5)
    l3._legend_box.align = "left"
    K.save(fig, "fig4")


if __name__ == "__main__":
    main()
