"""PR-SB SB-4 (ledger PR-SB, declared before any run; clarification 2): pricing the 1 K lever as a break-even premium.
Rescued cells: h*(6 K) > 8 h and h*(5 K) <= 8 h -
  SSP2-4.5 median around 2040: SB-3 (CDS tables, headline h* path): hist base = CDS rebuild ssp245_median h*, 5 K = sb/screen median_5K;
  historical: the existing AS2-2 grid (floor 4 K, HX 2 K vs HX 1 K), as declared for SB-3.
Break-even annualized premium per MW of IT: B = C_alt - C_dry5K, with
  C_alt   = least-cost design at the base approach from the full-frame menus: lc_cost (water allowed) and ban_cost (water banned)
            (ff_sites_uncorrected.csv historical rows; ff_sites_cmip_ssp245_median.csv);
  C_dry5K = annualized cost of the dry-only design at k = max(k_grid) with a tank of h*(5 K) hours, evaluated at the 5 K approach
            (plate-HX 1 K) on the same weather as the menus (mirror tables; scenario operator as the menu), no HX premium
            (the model does not price the dry route's plate HX): D.annual_cost(P5, D.capex_dry(P5, k, False, h, 0), opex_dry).
The tank h*(5 K) for C_dry5K is recomputed on that weather with the headline path; disagreements with the data-of-record class
are reported. Capital-equivalent premium = B / CRF ($/MW -> $/kW of IT).
Writes experiments/phase1/out/sb/sb4_price1k.json. Usage: python experiments/phase1/sb_price1k.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import ast  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _grid(x):
    """AS2-2 'grid' is stored as a JSON object (older files may hold its repr as a string)."""
    return ast.literal_eval(x) if isinstance(x, str) else x

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from bv1_min_storage import need_spare  # noqa: E402
from cmip_scenarios import scenario_weather  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / "experiments/phase1/out/sb"
MIRROR = pathlib.Path("D:/_caches/era5/weather_hub")


def h_star(P, W, lim):
    _, need, spare, _ = need_spare(P, W, max(D.k_grid(P, lim)), lim)
    ey = np.array([E.sequent_peak(need[y], spare[y])[0] for y in range(need.shape[0])])
    return float(np.sort(ey)[::-1][D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])])


def c_dry(P, W, lim, h):
    k = max(D.k_grid(P, lim))
    ev = E.eval_dry(P, W, k, False, h, lim)
    ox = E.opex_dry(P, W, ev, 0.0, lim)
    return D.annual_cost(P, D.capex_dry(P, k, False, h, 0.0), ox), int(np.sum(ev["cch_req"] > 1e-9))


def rescued(climate):
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    out = []
    for c in cells:
        if climate == "historical":  # PR-SB clarification 3: CDS data of record
            h6 = json.loads((ROOT / f"experiments/phase1/out/m2/cds_rebuild/{c}.json").read_text(encoding="utf-8"))["historical"]["h_star"]
            h5 = json.loads((OUT / "screen" / f"{c}__hist_5K.json").read_text(encoding="utf-8"))["h_star"]
        else:
            h6 = json.loads((ROOT / f"experiments/phase1/out/m2/cds_rebuild/{c}.json").read_text(encoding="utf-8"))["ssp245_median"]["h_star"]
            h5 = json.loads((OUT / "screen" / f"{c}__median_5K.json").read_text(encoding="utf-8"))["h_star"]
        if h6 > 8 + 1e-9 and h5 <= 8 + 1e-9:
            out.append((c, h6, h5))
    return out


def main():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    cap = sites.groupby("cell_id").B_full_MW.sum()
    Pb = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(Pb)
    P5 = PR.params(ch_etaII=0.5076, dT_HX_design=1.0)
    crf = Pb["crf"]
    dall = json.loads((ROOT / "experiments/phase1/out/cmip/deltas_all.json").read_text(encoding="utf-8"))
    res = {"crf": crf}
    for climate, ff_file in (("historical", "ff_sites_uncorrected.csv"), ("ssp245_median", "ff_sites_cmip_ssp245_median.csv")):
        ff = pd.read_csv(ROOT / f"experiments/phase1/out/{ff_file}")
        ff = ff[ff.climate == climate].groupby("cell").first()
        rows = []
        for c, h6, h5 in rescued(climate):
            W = E.load_weather(MIRROR / f"{c}.npz")
            if climate != "historical":
                W = scenario_weather(W, np.array(dall["ssp245"]["median"])[:, dall["cells"].index(c)], True)
            h5m = h_star(P5, W, lim)
            h6m = h_star(Pb, W, lim)
            cd, viol = c_dry(P5, W, lim, h5m)
            cd6, viol6 = c_dry(Pb, W, lim, h6m)  # PR-SB amendment: the storage alternative (dry-only, base approach, h*(6 K) store)
            lc, ban = float(ff.loc[c, "lc_cost"]), float(ff.loc[c, "ban_cost"])
            rows.append({"cell": c, "MW": float(cap[c]), "h6_record": h6, "h5_record": h5, "h6_menu_weather": h6m, "h5_menu_weather": h5m,
                         "class_agrees": bool(h6m > 8 + 1e-9 and h5m <= 8 + 1e-9), "C_dry5K": cd, "dry5K_violating_years": viol,
                         "C_lc": lc, "C_ban": ban, "B_lc_per_MWyr": lc - cd, "B_ban_per_MWyr": ban - cd,
                         "B_lc_capex_per_kW": (lc - cd) / crf / 1e3, "B_ban_capex_per_kW": (ban - cd) / crf / 1e3,
                         "C_dry6K_storage": cd6, "dry6K_violating_years": viol6, "B_storage_per_MWyr": cd6 - cd,
                         "B_storage_capex_per_kW": (cd6 - cd) / crf / 1e3})
            print(climate, rows[-1]["cell"], round(rows[-1]["MW"]), "B_storage $/kW", round(rows[-1]["B_storage_capex_per_kW"], 1),
                  "B_lc $/kW", round(rows[-1]["B_lc_capex_per_kW"], 1),
                  "B_ban $/kW", round(rows[-1]["B_ban_capex_per_kW"], 1), "agree", rows[-1]["class_agrees"], flush=True)
        df = pd.DataFrame(rows)
        summ = {"rescued_GW": float(df.MW.sum() / 1e3) if len(df) else 0.0, "cells": rows}
        if len(df):
            for key in ("B_lc_capex_per_kW", "B_ban_capex_per_kW", "B_storage_capex_per_kW"):
                o = np.argsort(df[key].to_numpy())
                v, w = df[key].to_numpy()[o], df.MW.to_numpy()[o]
                cw = np.cumsum(w) / w.sum()
                summ[key] = {"min": float(v.min()), "capacity_weighted_median": float(v[np.searchsorted(cw, 0.5)]), "max": float(v.max())}
                summ[f"{key}_curve_GW_paying_if_premium_below"] = {f"{x:g}": float(df.MW[df[key] >= x].sum() / 1e3)
                                                                    for x in (0, 10, 25, 50, 100, 200, 400)}
        res[climate] = summ
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sb4_price1k.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cells"} for k, v in res.items() if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
