"""PR-AS part D (cost assumptions) and the part-A aggregation.
Part D: re-price the PR-FF menu (both tags) with chiller capex 200 / 500 k$/MW, dry-cooler capex 50 / 150 k$/MW per module, capital
recovery at 6 % / 10 % over 15 years; re-select the ban and water-allowed least cost per site; report the ban's $ per m³ saved,
banned Z3 share, installed / physically necessary / economic compressor capacity and water-allowed Z3 share (B-full).
Part A: F_dry at 8 h (B-full) for every stress variant in experiments/phase1/out/as/.
Writes experiments/phase1/out/as_summary.json.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import ff_aggregate as A  # noqa: E402

OUT = ROOT / "experiments/phase1/out"
CL = ("historical", "stress_plus1.2K")
BASE = {"chiller": 300e3, "dc": 80e3}
VARIANTS = {"base": {}, "chiller_200k": {"chiller": 200e3}, "chiller_500k": {"chiller": 500e3}, "dc_50k": {"dc": 50e3},
            "dc_150k": {"dc": 150e3}, "crf_6pct": {"crf": 0.06 / (1 - 1.06 ** -15)}, "crf_10pct": {"crf": 0.10 / (1 - 1.10 ** -15)}}


def reprice(d, pr, rk, v):
    c = d["capex"]
    sc = v.get("chiller", BASE["chiller"]) / BASE["chiller"]
    sd = v.get("dc", BASE["dc"]) / BASE["dc"]
    capex = (c["dc"] + c["ad"]) * sd + c["tank_MWh"] * pr["tank"] + c["chiller"] * sc + c["tower"]
    ox = A.ox_of(d, rk)
    return v.get("crf", pr["crf"]) * capex + ox["elec_MWh"] * pr["elec"] + ox["water_m3"] * pr["water"]


def part_d():
    res = {}
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    for tag in ("uncorrected", "corrected"):
        fs = pd.read_csv(OUT / f"ff_sites_{tag}.csv")
        items = {}
        for f in (OUT / "ff" / tag).glob("*.json"):
            it = json.loads(f.read_text(encoding="utf-8"))
            items[(it["cell"], it["climate"])] = it
        res[tag] = {}
        for name, v in VARIANTS.items():
            for cl in CL:
                rows = []
                for (cell, c2), it in items.items():
                    if c2 != cl:
                        continue
                    pr = A.base_prices(it)
                    rk = A.ratio_key(it, pr)
                    comp = [d for d in it["designs"] if d["compliant"]]
                    lc = min(comp, key=lambda d: reprice(d, pr, rk, v))
                    ban = min((d for d in comp if A.is_ban(d)), key=lambda d: reprice(d, pr, rk, v))
                    rows.append({"cell_id": cell, "lc_cost": reprice(lc, pr, rk, v), "ban_cost": reprice(ban, pr, rk, v),
                                 "lc_water": A.ox_of(lc, rk)["water_m3"], "ban_c_ch": ban["c_ch"], "lc_c_ch": lc["c_ch"]})
                df = sites.merge(pd.DataFrame(rows), on="cell_id")
                phys = df.merge(fs[fs.climate == cl][["site_id", "cphys_ban_h8"]], on="site_id").cphys_ban_h8.fillna(0).to_numpy()
                w = df.B_full_MW.astype(float).to_numpy()
                dcost = (w * (df.ban_cost - df.lc_cost)).sum()
                dwat = (w * df.lc_water).sum()
                res[tag][f"{name}|{cl}"] = {
                    "USD_per_m3": float(dcost / dwat), "ban_added_cost_MUSD_yr": float(dcost / 1e6),
                    "banned_Z3_share": float(w[df.ban_c_ch > 1e-9].sum() / w.sum()),
                    "installed_GW_th": float((w * df.ban_c_ch).sum() / 1e3), "physical_GW_th": float((w * phys).sum() / 1e3),
                    "economic_GW_th": float((w * np.clip(df.ban_c_ch - phys, 0, None)).sum() / 1e3),
                    "water_allowed_Z3_share": float(w[df.lc_c_ch > 1e-9].sum() / w.sum())}
    return res


def part_a():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    recs = {}
    for f in (OUT / "as").glob("cell_*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        recs[(r["cell"], r["climate"])] = r["h_star"]
    res = {"items": len(recs)}
    if not recs:
        return res
    keys = [k for k in next(iter(recs.values())) if not k.endswith("_kmax")]
    for cl in CL:
        w = sites.B_full_MW.astype(float)
        for k in keys:
            h = sites.cell_id.map(lambda c: recs.get((c, cl), {}).get(k, np.nan)).astype(float)
            res[f"{k}|{cl}"] = {"F_dry_8h_GW": float(w[h <= 8 + 1e-9].sum() / 1e3), "F_dry_8h_share": float(w[h <= 8 + 1e-9].sum() / w.sum()),
                                "missing": int(h.isna().sum())}
    return res


def main():
    res = {"A": part_a(), "D": part_d()}
    (OUT / "as_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    a = res["A"]
    for cl in CL:
        base = a.get(f"base|{cl}", {}).get("F_dry_8h_GW")
        print(cl, "A (F_dry 8 h GW):", {k.split("|")[0]: round(v["F_dry_8h_GW"], 2) for k, v in a.items() if isinstance(v, dict) and k.endswith(cl)})
    for tag in ("uncorrected", "corrected"):
        for k, v in res["D"][tag].items():
            print(tag, k, {kk: round(vv, 3) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
