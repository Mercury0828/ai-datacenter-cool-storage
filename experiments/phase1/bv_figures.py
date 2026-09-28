"""Diagnostic figures for package 1 (not paper figures): capacity-weighted F_dry(h) on the full frame (B-full) for each
weather tag × climate, and the storage–dry-cooler trade-off F_dry(8 h) by dry-cooler size k. Reads bv_summary.json."""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/phase1/out"


def main():
    r = json.loads((OUT / "bv_summary.json").read_text(encoding="utf-8"))
    h = r["h_grid"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    styles = {("uncorrected", "historical"): ("C0", "-"), ("uncorrected", "stress_plus1.2K"): ("C3", "-"),
              ("corrected", "historical"): ("C0", "--"), ("corrected", "stress_plus1.2K"): ("C3", "--")}
    for (tag, cl), (c, ls) in styles.items():
        rec = r.get(tag, {}).get(cl, {})
        if "F_dry_curve" in rec:
            ax[0].plot(h, rec["F_dry_curve"]["B_full_MW"], color=c, ls=ls, label=f"{tag}, {cl}")
            ks = rec["tradeoff_F_dry_8h_by_k|B_full_MW"]
            ax[1].plot([float(k) for k in ks], list(ks.values()), color=c, ls=ls, marker="o", ms=3, label=f"{tag}, {cl}")
    ax[0].axvline(8, color="grey", lw=0.8)
    ax[0].set(xlabel="storage h (hours of full IT heat)", ylabel="F_dry (capacity share, B-full)", ylim=(0, 1.02),
              title="Dry-only, compressor-free share vs storage")
    ax[1].set(xlabel="dry-cooler size k (grid)", ylabel="F_dry(8 h)", ylim=(0, 1.02), title="Storage–dry-cooler trade-off")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bv_Fdry_h.png", dpi=150)
    print("wrote", OUT / "fig_bv_Fdry_h.png")


if __name__ == "__main__":
    main()
