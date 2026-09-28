"""BV-4 (ledger PR-BV): risk interpretation — in-sample screen vs out-of-sample reliability, and the R1–R2 design crosswalk.

R1's least-cost designs (Phase 0, 15 sites, headline prototype, historical climate) were selected and screened on the same 36
site-years (1990–2025): every one is sized in-sample. R2 applied the same empirical rule (SAA, m' = 1) to the training years
1990–2014 only and evaluated the chosen design on 2015–2025 (F0 as observed; F1–F4 warmed). This script lines the two up per site
and reports exact one-sided 95 % binomial (Clopper–Pearson) upper bounds for the empirical screen.
Output: experiments/phase1/out/bv4/bv4_crosswalk.json and .csv. Reads Phase-0 outputs only (no new model runs).
"""
import json
import pathlib

import pandas as pd
from scipy.stats import beta

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/phase1/out/bv4"


def cp_upper(k, n, conf=0.95):
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r1 = pd.read_csv(ROOT / "experiments/phase0/out/r1/r1_sites.csv")
    r1 = r1[(r1.variant == "headline") & (r1.climate == "current")].set_index("site")
    r2 = json.loads((ROOT / "experiments/phase0/out/r2/r2_results.json").read_text(encoding="utf-8"))
    rows = []
    for site, s in r2["per_site"]["SAA_m1"].items():
        a = r1.loc[site]
        r1_design = (a.liq_family, round(float(a.liq_k), 4), bool(a.liq_ad), float(a.liq_tank_h), float(a.liq_kct))
        fam, k, ad, th, kct = s["design"]
        r2_design = (fam, round(float(k), 4), bool(ad), float(th), float(kct))
        rows.append({"site": site, "w_full_MW": float(a.w_full), "R1_zone": a.zone, "R1_design_in_sample_36y": str(r1_design),
                     "R1_chiller_per_unit": float(a.liq_c_ch), "R2_SAA_m1_design_train_25y": str(r2_design),
                     "R2_chiller_per_unit": float(s["C"]),
                     # internal review boundary-validity R1 finding 2: equality must include the chiller capacity (compressor presence)
                     "same_non_chiller_configuration": r1_design == r2_design,
                     "same_full_design_incl_chiller": r1_design == r2_design and abs(float(a.liq_c_ch) - float(s["C"])) <= 1e-6,
                     "compressor_presence_differs": (float(a.liq_c_ch) > 1e-9) != (float(s["C"]) > 1e-9),
                     **{f"R2_test_viol_{f}": float(v) for f, v in s["viol"].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bv4_crosswalk.csv", index=False)
    w = df.w_full_MW
    res = {"n_sites": len(df), "n_same_non_chiller_configuration": int(df.same_non_chiller_configuration.sum()),
           "n_same_full_design_incl_chiller": int(df.same_full_design_incl_chiller.sum()),
           "sites_compressor_presence_differs": df.site[df.compressor_presence_differs].tolist(),
           "same_non_chiller_configuration_MW": float(w[df.same_non_chiller_configuration].sum()),
           "same_full_design_MW": float(w[df.same_full_design_incl_chiller].sum()), "total_MW": float(w.sum()),
           "SAA_m1_pooled_test_violation_weighted": r2["methods"]["weighted"]["SAA_m1"]["by_family"],
           "SAA_m1_pooled_test_violation_unweighted": r2["methods"]["unweighted"]["SAA_m1"]["by_family"],
           "binomial_upper_95_label": "nominal fixed-design, iid-binomial reference bounds; no selection-adjusted coverage is claimed "
                                      "for the R1 designs (selected and screened on the same years) (internal review boundary-validity R1 finding 3)",
           "binomial_upper_95": {"0/36": cp_upper(0, 36), "1/36": cp_upper(1, 36), "0/25": cp_upper(0, 25),
                                 "1/25": cp_upper(1, 25), "0/11": cp_upper(0, 11), "1/11": cp_upper(1, 11)},
           "note": "R1 designs are all in-sample (selected and screened on the same 36 years); R2 SAA m'=1 is the same rule on "
                   "25 training years with 11 held-out years (F0 observed; F1-F4 warmed +0.6/+1.2/+1.8 K and hot-tail amplified)."}
    (OUT / "bv4_crosswalk.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(df[["site", "R1_zone", "R1_design_in_sample_36y", "R1_chiller_per_unit", "R2_SAA_m1_design_train_25y", "R2_chiller_per_unit",
              "same_full_design_incl_chiller",
              "R2_test_viol_F0", "R2_test_viol_F2"]].to_string())
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
