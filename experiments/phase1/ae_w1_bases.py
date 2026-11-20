"""AE transfer W1: the water-free, compressor-free boundary (h* <= 8 h) on the three capacity bases (B-full, B-2027, B-now), from the
CDS rebuild of record (out/m2/cds_rebuild/<cell>.json). No simulation; re-weights frozen per-cell h*.
Usage: python experiments/phase1/ae_w1_bases.py   Output: experiments/phase1/out/ae/w1_bases.json
"""
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
rec = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in (ROOT / "experiments/phase1/out/m2/cds_rebuild").glob("cell_*.json")}
res = {}
for cl in ("historical", "stress_plus1.2K", "ssp245_median"):
    h = sites.cell_id.map(lambda c: rec[c][cl]["h_star"])
    res[cl] = {}
    for b in ("B_full_MW", "B_2027_MW", "B_now_MW"):
        w = sites[b].astype(float)
        res[cl][b] = {"total_GW": w.sum() / 1e3, "F_dry_8h_GW": w[h <= 8 + 1e-9].sum() / 1e3, "F_dry_0h_GW": w[h <= 1e-9].sum() / 1e3,
                      "share_8h": w[h <= 8 + 1e-9].sum() / w.sum()}
out = ROOT / "experiments/phase1/out/ae/w1_bases.json"
out.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
