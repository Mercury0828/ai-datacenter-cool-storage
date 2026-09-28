"""PR-FF-CDS comparison: fleet quantities of the design menu on the CDS data of record against the earlier ERA5 copy
(historical climate). Reads ff_summary_{uncorrected,cds}.json and ff_sites_{uncorrected,cds}.csv; writes ff_cds_compare.json and ff_cds_site_designs.json.
Usage: python experiments/phase1/ff_cds_compare.py
"""
import json
import pathlib

import pandas as pd

OUT = pathlib.Path(__file__).resolve().parents[2] / "experiments/phase1/out"


def fleet(tag):
    f = json.loads((OUT / f"ff_summary_{tag}.json").read_text(encoding="utf-8"))["fleet"]["B_full_MW"]["historical"]
    lc, ban, tw = f["lc_scale"], f["ban_scale"], f["tower_scale"]
    added = (ban["cost_BUSD_yr"] - lc["cost_BUSD_yr"]) * 1e3
    saved = lc["water_GL_yr"] - ban["water_GL_yr"]
    # the registered $/m3 nets out the water purchases at $2/m3, which are part of the water-allowed designs' cost
    return {"water_allowed_GL": lc["water_GL_yr"], "tower_water_GL": tw["water_GL_yr"], "ban_added_cost_MUSD": added,
            "ban_USD_per_m3": added / saved, "ban_chillers_GWth": ban["chiller_GW_th"],
            "envelope_min_GWth": f["cphys_GW_th"]["cphys_ban_h8"], "lc_Z1": f["lc_zone_share"]["Z1"],
            "lc_Z2": f["lc_zone_share"]["Z2"], "lc_Z3": f["lc_zone_share"]["Z3"], "ban_Z1": f["ban_zone_share"]["Z1"],
            "ban_Z3": f["ban_zone_share"]["Z3"], "lc_elec_TWh": lc["elec_TWh_yr"], "ban_elec_TWh": ban["elec_TWh_yr"],
            "lc_peak_GW": lc["peak_GW_e"], "ban_peak_GW": ban["peak_GW_e"]}


def site_designs(sa, sb):
    """Site-level design changes per route (water allowed 'lc', ban, tower): full design strings, family or pads, chiller
    on/off, dry-cooler size, tower size, and relative site-cost changes. Written to ff_cds_site_designs.json."""
    res = {}
    for route in ("lc", "ban", "tower"):
        ch = [s for s in sa.index if str(sa.loc[s, f"{route}_design"]) != str(sb.loc[s, f"{route}_design"])]
        fam = [s for s in sa.index if (sa.loc[s, f"{route}_family"], sa.loc[s, f"{route}_ad"])
               != (sb.loc[s, f"{route}_family"], sb.loc[s, f"{route}_ad"])]
        chiller = [s for s in sa.index if (sa.loc[s, f"{route}_c_ch"] > 1e-9) != (sb.loc[s, f"{route}_c_ch"] > 1e-9)]
        rc = (sb[f"{route}_cost"] - sa[f"{route}_cost"]) / sa[f"{route}_cost"]
        top = rc.abs().sort_values(ascending=False).head(4)
        res[route] = {"design_changes": ch, "design_change_MW": float(sa.loc[ch, "B_full_MW"].sum()),
                      "family_or_pads_changes": fam, "chiller_on_off_changes": chiller,
                      "max_abs_site_cost_change": float(rc.abs().max()),
                      "k_changes": [s for s in sa.index if abs(sa.loc[s, f"{route}_k"] - sb.loc[s, f"{route}_k"]) > 1e-6],
                      "kct_changes": [s for s in sa.index if str(sa.loc[s, f"{route}_kct"]) != str(sb.loc[s, f"{route}_kct"])],
                      "top_cost_changes_pct": {sa.loc[s, "site"]: float(100 * rc[s]) for s in top.index},
                      "median_abs_cost_change_pct": float(100 * rc.abs().median())}
    (OUT / "ff_cds_site_designs.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main():
    a, b = fleet("uncorrected"), fleet("cds")
    rows = {k: {"earlier_copy": a[k], "cds": b[k], "rel_change": (b[k] - a[k]) / a[k] if a[k] else None} for k in a}
    sa = pd.read_csv(OUT / "ff_sites_uncorrected.csv")
    sb = pd.read_csv(OUT / "ff_sites_cds.csv")
    sa, sb = sa[sa.climate == "historical"].set_index("site_id"), sb[sb.climate == "historical"].set_index("site_id")
    zone_changes = [s for s in sa.index if (sa.loc[s, "lc_zone"], sa.loc[s, "ban_zone"]) != (sb.loc[s, "lc_zone"], sb.loc[s, "ban_zone"])]
    res = {"fleet": rows, "sites_changing_zone": zone_changes,
           "max_abs_rel_change": max(abs(r["rel_change"]) for r in rows.values() if r["rel_change"] is not None)}
    (OUT / "ff_cds_compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    site_designs(sa, sb)
    for k, r in rows.items():
        rc = "" if r["rel_change"] is None else f"{100 * r['rel_change']:+.2f} %"
        print(f"{k:22s} {r['earlier_copy']:10.4f} {r['cds']:10.4f} {rc}")
    print("sites changing zone:", zone_changes)


if __name__ == "__main__":
    main()
