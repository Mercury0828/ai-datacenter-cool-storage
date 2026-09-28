"""PR-SB SB-1 (ledger PR-SB, declared before any run): storage requirement and physical scale of the water-free, compressor-free
boundary, from the CDS rebuild h* (80 hub cells; historical, +1.2 K, SSP2-4.5 median around 2040; B-full capacity weights).
- F_dry(h) at 0, 1, 2, 4, 6, 8 h; the capacity with h* in (4, 8] and (6, 8] h;
- installed storage of the feasible capacity E = sum w h* (GWh_th; per-unit design heat = IT power) and at the 8 h cap;
- water volume at the declared 10 K swing: 1 MWh_th = 3.6e9 J / (4.18e6 J m-3 K-1 x 10 K) = 86.1 m3.
Writes experiments/phase1/out/sb/sb1_storage.json. Usage: python experiments/phase1/sb_storage.py
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/phase1/out/sb"
M3_PER_MWH = 3.6e9 / (4.18e6 * 10.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    rec = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in (ROOT / "experiments/phase1/out/m2/cds_rebuild").glob("cell_*.json")}
    w = sites.B_full_MW.astype(float).to_numpy()
    res = {"m3_per_MWh_th": M3_PER_MWH, "frame_GW": float(w.sum() / 1e3)}
    for cl in ("historical", "stress_plus1.2K", "ssp245_median"):
        h = sites.cell_id.map(lambda c: rec[c][cl]["h_star"]).to_numpy(float)
        feas = h <= 8 + 1e-9
        E_feas = float(np.sum(w[feas] * h[feas]) / 1e3)  # GWh_th (MW x h / 1e3)
        E_cap = float(np.sum(w[feas] * 8.0) / 1e3)
        res[cl] = {
            "F_dry_GW": {f"{x:g}h": float(w[h <= x + 1e-9].sum() / 1e3) for x in (0, 1, 2, 4, 6, 8)},
            "GW_h_in_4_8": float(w[(h > 4 + 1e-9) & feas].sum() / 1e3),
            "GW_h_in_6_8": float(w[(h > 6 + 1e-9) & feas].sum() / 1e3),
            "storage_GWh_th_at_h_star": E_feas, "storage_GWh_th_at_8h_cap": E_cap,
            "volume_million_m3_at_h_star": E_feas * 1e3 * M3_PER_MWH / 1e6,
            "volume_million_m3_at_8h_cap": E_cap * 1e3 * M3_PER_MWH / 1e6,
            "capacity_weighted_mean_h_star_of_feasible": float(np.sum(w[feas] * h[feas]) / w[feas].sum()),
            # corrected after internal review strengthen-boundary R1 (B2): inverse weighted empirical CDF; the interpolated value is kept, labelled
            "capacity_weighted_median_h_star_of_feasible": float(np.sort(h[feas])[np.searchsorted(
                np.cumsum(w[feas][np.argsort(h[feas])]) / w[feas].sum(), 0.5 - 1e-12)]),
            "interpolated_quantile_estimate_h": float(np.interp(0.5, np.cumsum(w[feas][np.argsort(h[feas])]) / w[feas].sum(),
                                                                np.sort(h[feas]))),
        }
        print(cl, json.dumps(res[cl], indent=None)[:600], flush=True)
    (OUT / "sb1_storage.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
