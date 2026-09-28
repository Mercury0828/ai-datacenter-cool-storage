"""Figure 2 — the water-free, compressor-free boundary of the tracked build-out and the storage it takes.

Artifacts: experiments/phase1/out/m2/cds_rebuild/cell_*.json (h* per hub cell on the CDS data of record, DD-017);
sb/sb1_storage.json (storage totals and medians, PR-SB SB-1); st_summary.json (simulated tank volumes, PR-ST);
data/sites/hub_sites.csv (Epoch AI hub, B-full IT MW).
Band in (A): DD-003 capacity-estimate interval, the DE-2 method (independent per-site lognormal multipliers,
sigma = ln 1.4 / 1.2816, 10,000 draws, seed 20260925), 10-90 % range of the share, scaled to the 37.3 GW total.
"""
import numpy as np
from matplotlib.lines import Line2D

import common as K

CL = [("historical", "1990–2025 weather", K.C["hist"], "-"),
      ("ssp245_median", "SSP2-4.5, circa 2040", K.C["c2040"], "-"),
      ("stress_plus1.2K", "+1.2 K stress", K.C["stress"], (0, (3, 1.5)))]
CLASSES = [("No store", lambda h: h <= 1e-9, "#0072B2", "o"),
           ("≤ 2 h", lambda h: (h > 1e-9) & (h <= 2 + 1e-9), "#56B4E9", "s"),
           ("2–8 h", lambda h: (h > 2 + 1e-9) & (h <= 8 + 1e-9), "#E69F00", "D"),
           ("> 8 h", lambda h: h > 8 + 1e-9, "#D55E00", "^")]


def fdry_curve(h, w, grid):
    return np.array([w[h <= g + 1e-9].sum() for g in grid]) / 1e3


def band(h, w, grid):
    rng = np.random.default_rng(20260925)
    z = rng.standard_normal((10000, len(w)))
    W = w * np.exp(np.log(1.4) / 1.2816 * z)
    tot = W.sum(1)
    lo, hi = [], []
    for g in grid:
        sh = (W * (h <= g + 1e-9)).sum(1) / tot
        lo.append(np.quantile(sh, 0.1))
        hi.append(np.quantile(sh, 0.9))
    return np.array(lo) * w.sum() / 1e3, np.array(hi) * w.sum() / 1e3


def main():
    K.style()
    fig = K.plt.figure(figsize=(K.FULL, 120 * K.MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.28, 0.30], height_ratios=[1.0, 1.0],
                          left=0.075, right=0.995, bottom=0.085, top=0.955, wspace=0.10, hspace=0.42)
    grid = np.linspace(0, 12, 1201)

    # (A) F_dry(h)
    ax = fig.add_subplot(gs[0, 0])
    for cl, lab, col, ls in CL:
        s = K.site_hstar(cl)
        h, w = s.h_star.to_numpy(float), s.B_full_MW.to_numpy(float)
        if cl == "historical":
            lo, hi = band(h, w, grid)
            ax.fill_between(grid, lo, hi, color=col, alpha=0.18, lw=0, step="post")
            i8 = np.searchsorted(grid, 8.0)
            print("band at 8 h (GW):", lo[i8], hi[i8], "share", lo[i8] / (w.sum() / 1e3), hi[i8] / (w.sum() / 1e3))
        ax.step(grid, fdry_curve(h, w, grid), where="post", color=col, ls=ls, lw=1.1, label=lab)
    ax.plot([0, 4.8], [37.295, 37.295], color=K.C["grey"], lw=0.5, ls=":")
    ax.text(0.2, 37.9, "Tracked capacity, 37.3 GW", fontsize=7, color=K.C["grey"], va="bottom")
    ax.plot([8, 8], [22.5, 41], color=K.C["grey"], lw=0.5, ls=":")
    ax.plot([8, 8], [34.29, 29.05], ls="", marker="o", ms=2.5, mfc="white", mec="black", mew=0.5, zorder=5)
    ax.text(5.1, 35.3, "34.3 GW", fontsize=7, color=K.C["hist"], va="bottom")
    ax.text(8.3, 25.4, "29.1 GW", fontsize=7, color=K.C["c2040"], va="top")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 41)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_xlabel("Cool-water store [hours of full IT heat]")
    ax.set_ylabel("Water-free, compressor-free capacity [GW]")
    ax.legend(loc="lower right", handlelength=2.2, borderaxespad=0.2)
    K.panel_label(ax, "A", dx=-0.2)

    # (B) US map, historical store class
    s = K.site_hstar("historical")
    axm = fig.add_subplot(gs[0, 1])
    K.draw_conus(axm)
    us = s[s.country == "United States"].sort_values("B_full_MW", ascending=False)
    x, y = K.albers(us.lon.to_numpy(), us.lat.to_numpy())
    h = us.h_star.to_numpy(float)
    for lab, f, col, mk in CLASSES:
        m = f(h)
        axm.scatter(x[m], y[m], s=K.msize(us.B_full_MW.to_numpy()[m]), c=col, marker=mk, edgecolors="black",
                    linewidths=0.35, alpha=0.9, zorder=3)
    axm.set_title("United States (77 sites)", fontsize=7, pad=2)
    K.panel_label(axm, "B", dx=0.0, dy=1.0)
    axl = fig.add_subplot(gs[0, 2])
    axl.axis("off")
    hs = [Line2D([], [], ls="", marker=mk, mfc=col, mec="black", mew=0.35, ms=5, label=lab)
          for lab, f, col, mk in CLASSES]
    leg1 = axl.legend(handles=hs, title="Store needed,\n1990–2025", loc="upper left", bbox_to_anchor=(-0.1, 1.0),
                      handletextpad=0.2, borderaxespad=0.0, labelspacing=0.35, title_fontsize=7)
    leg1._legend_box.align = "left"
    axl.add_artist(leg1)
    sz = [Line2D([], [], ls="", marker="o", mfc="white", mec="black", mew=0.35,
                 ms=np.sqrt(K.msize(v)), label=f"{v / 1e3:g} GW") for v in (250, 1000, 2000)]
    leg2 = axl.legend(handles=sz, title="IT capacity", loc="lower left", bbox_to_anchor=(-0.1, 0.0), handletextpad=0.4,
                      borderaxespad=0.0, labelspacing=0.9, title_fontsize=7)
    leg2._legend_box.align = "left"

    # (C) storage distribution
    axc = fig.add_subplot(gs[1, 0])
    edges = [(0, 0, "0"), (0, 1, "0–1"), (1, 2, "1–2"), (2, 4, "2–4"), (4, 6, "4–6"),
             (6, 8, "6–8"), (8, np.inf, "> 8")]
    xs = np.arange(len(edges))
    for j, (cl, lab, col, ls) in enumerate(CL[:2]):
        s2 = K.site_hstar(cl)
        hh, ww = s2.h_star.to_numpy(float), s2.B_full_MW.to_numpy(float)
        vals = []
        for a, b, _ in edges:
            if b == 0:
                m = hh <= 1e-9
            elif np.isinf(b):
                m = hh > 8 + 1e-9
            else:
                m = (hh > a + 1e-9) & (hh <= b + 1e-9)
            vals.append(ww[m].sum() / 1e3)
        print(cl, [round(v, 2) for v in vals])
        if j == 0:
            axc.bar(xs - 0.19, vals, width=0.36, color=col, lw=0, label=lab)
        else:
            axc.bar(xs + 0.19, vals, width=0.36, color="white", edgecolor=col, lw=0.8, label=lab)
    axc.plot([5.5, 5.5], [0, 9.6], color=K.C["grey"], lw=0.5, ls=":")
    axc.set_xticks(xs)
    axc.set_xticklabels([e[2] for e in edges])
    axc.set_xlabel("Store needed [hours of full IT heat]")
    axc.set_ylabel("IT capacity [GW]")
    axc.set_ylim(0, 17)
    axc.set_yticks([0, 5, 10, 15])
    axc.legend(loc="upper left", borderaxespad=0.1)
    sb = K.jload("sb/sb1_storage.json")["historical"]
    st = K.jload("st_summary.json")  # PR-ST: simulated tanks, the smaller piping per site and series piping
    axc.text(3.55, 15.6, "Median 1.7 h (2.0 h around 2040)\nAll 34.3 GW: %.0f GWh,\n%.1f to %.1f million m³ of water"
             % (sb["storage_GWh_th_at_h_star"], st["fleet_volume_Mm3|min|h_star<=8"],
                st["fleet_volume_Mm3|series|h_star<=8"]),
             fontsize=7, va="top", ha="left")
    K.panel_label(axc, "C", dx=-0.2)

    # (D) sites outside the United States
    axw = fig.add_subplot(gs[1, 1:])
    K.draw_world(axw)
    ot = s[s.country != "United States"]
    hh = ot.h_star.to_numpy(float)
    for lab, f, col, mk in CLASSES:
        m = f(hh)
        axw.scatter(ot.lon.to_numpy()[m], ot.lat.to_numpy()[m], s=K.msize(ot.B_full_MW.to_numpy()[m]) + 4, c=col,
                    marker=mk, edgecolors="black", linewidths=0.35, zorder=3)
    axw.set_title("Other countries (16 sites)", fontsize=7, pad=2)
    K.panel_label(axw, "D", dx=0.0, dy=1.0)
    K.save(fig, "fig2")


if __name__ == "__main__":
    main()
