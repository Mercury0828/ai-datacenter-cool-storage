"""M1 calibration against the targets pre-registered in ledger PR-M1 (2026-09-24, commit 14e8913).

Writes experiments/phase0/out/m1_calibration.json and prints a table. Tolerances are the pre-registered ones; a miss is
reported as a miss (diagnosed afterwards, never tuned away).
"""
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import components as C  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / "experiments/phase0/out"
OUT.mkdir(parents=True, exist_ok=True)
res = {}

# ---------------- M1-ii: chiller eta_II from ASHRAE 90.1-2019 air-cooled >=150 ton Path A (10.1 EER -> COP 2.96)
cop_ahri = 10.1 / 3.412
te, tc = 6.7 - 4.0 + 273.15, 35.0 + 15.0 + 273.15
eta2 = cop_ahri / (te / (tc - te))
P = PR.params(ch_etaII=round(eta2, 4))
cop_7 = float(C.chiller_cop(35.0, 7.0, P))
cop_10 = float(C.chiller_cop(35.0, 10.0, P))  # registered validation condition (PR-M1: 15/10 degC water, 35 degC)
res["M1-ii"] = {"etaII_fitted": P["ch_etaII"], "cop_AHRI_target": cop_ahri,
                "validation_optichill_EER_upto": 3.21, "model_cop_at_35C_10C_registered": cop_10,
                "rel_err": cop_10 / 3.21 - 1, "tolerance": 0.15, "pass": abs(cop_10 / 3.21 - 1) <= 0.15,
                "secondary_model_cop_at_35C_7C": cop_7,
                "note": "R1 run validated at 7 degC by mistake (departure from PR-M1); corrected to the registered 10 degC",
                "model_cop_trim_at_45C_amb_41C_lvg": float(C.chiller_cop(45.0, 41.0, P)),
}

# ---------------- M1-i-a: Airedale DR10 table (water range 5 K, airflow 2.3 m3/s, constant density 1.2 kg/m3)
table = {(35, 30): {25: 11.0}, (40, 35): {25: 19.6, 30: 11.0}, (45, 40): {25: 28.1, 30: 19.6, 35: 11.1},
         (50, 45): {25: 36.6, 30: 28.0, 35: 19.6, 40: 11.1}, (55, 50): {25: 45.0, 30: 36.4, 35: 27.9, 40: 19.5, 45: 11.1}}
c_air = 2.3 * 1.2 * 1.006  # kW/K


def q_dr10(ua, tw_in, t_air):
    """Solve Q with water range fixed at 5 K (C_w = Q/5): Q = eps(UA/Cmin, Cr) * Cmin * ITD."""
    itd = tw_in - t_air
    lo, hi = 1e-6, c_air * itd
    for _ in range(200):
        q = 0.5 * (lo + hi)
        c_w = q / 5.0
        cmin, cmax = min(c_air, c_w), max(c_air, c_w)
        f = float(C.eps_crossflow_unmixed(ua / cmin, cmin / cmax)) * cmin * itd - q
        lo, hi = (q, hi) if f > 0 else (lo, q)
    return 0.5 * (lo + hi)


lo, hi = 0.01, 50.0
for _ in range(200):
    ua = 0.5 * (lo + hi)
    lo, hi = (lo, ua) if q_dr10(ua, 45.0, 35.0) > 11.1 else (ua, hi)
ua_fit = 0.5 * (lo + hi)
rows = []
for (tin, tout), d in table.items():
    for ta, qv in d.items():
        qm = q_dr10(ua_fit, float(tin), float(ta))
        rows.append({"water": f"{tin}/{tout}", "ambient": ta, "catalog_kW": qv, "model_kW": round(qm, 2),
                     "rel_err": round(qm / qv - 1, 4)})
worst = max(abs(r["rel_err"]) for r in rows)
res["M1-i-a"] = {"UA_fit_kW_per_K": ua_fit, "points": rows, "max_abs_rel_err": worst, "tolerance": 0.10,
                 "pass": worst <= 0.10}

# ---------------- M1-i-b: reference module / P5 floor (construction constraint)
lim = PR.fws_max(P)
kmax = D.k_max(P, lim)
res["M1-i-b"] = {"k_max": kmax, "dc_approach_at_k1": C.DryChain(P, 1.0, lim).dc_approach(),
                 "departure_note": "PR-M1 says k in [1, k_max] spans ~10 K to 4 K. At k = 1 the module gives 10 K only at "
                                   "its rating (5 K glycol range); in the chain (10 K glycol range) k = 1 gives 8.1 K, so "
                                   "the grid starts at k = 0.6 (16.4 K) to cover the ~10 K end. Recorded departure.",
                 "dc_approach_at_kmax": C.DryChain(P, kmax, lim).dc_approach(),
                 "chain_approach_at_kmax_incl_HX": C.DryChain(P, kmax, lim).design_approach(),
                 "k_grid": D.k_grid(P, lim),
                 "pass": abs(C.DryChain(P, kmax, lim).dc_approach() - P["dc_approach_floor"]) < 1e-3
                 and C.DryChain(P, min(D.k_grid(P, lim)), lim).dc_approach() >= 10.0}

# ---------------- M1-iii-a: Mokkapati & Das design-point verdicts with their quoted DBTs
quoted = {("Phoenix", 0.4): 44.0, ("Phoenix", 2.0): 41.7, ("Las Vegas", 0.4): 42.9, ("Las Vegas", 2.0): 40.5,
          ("Tucson", 0.4): 41.1, ("Tucson", 1.0): 39.9, ("Tucson", 2.0): 38.7, ("El Paso", 0.4): 38.8}
their = {1.1: {"Phoenix": "INF", "Las Vegas": "INF", ("Tucson", 0.4): "INF"},
         2.8: {("Tucson", 1.0): "INF"}}


def md_label(site, pct, appr):
    if site in ("Phoenix", "Las Vegas"):
        return "INF"
    if site == "Tucson" and pct == 0.4:
        return "INF"
    if site == "Tucson" and pct == 1.0 and appr >= 2.8:
        return "INF"
    return "FEAS"


rows, mism = [], []
for (site, pct), dbt in quoted.items():
    for appr in (1.1, 1.7, 2.8):
        ours = "FEAS" if D.design_point_dry_feasible(dbt, 41.7, appr) else "INF"
        theirs = md_label(site, pct, appr)
        rows.append({"site": site, "pct": pct, "dbt": dbt, "min_approach": appr, "ours": ours, "theirs": theirs})
        if ours != theirs:
            mism.append(rows[-1])
expected_diff = [r for r in mism if r["site"] == "Las Vegas" and r["pct"] == 2.0 and r["min_approach"] == 1.1]
res["M1-iii-a"] = {"rows": rows, "mismatches": mism, "pass": len(mism) == len(expected_diff),
                   "note": "Las Vegas 2 % at 1.1 K: rule says feasible (40.5+1.1=41.6<=41.7); authors call it practically "
                           "infeasible (~20 units). Documented modelling difference (PR-M1)."}

# ---------------- M1-iii-b: ERA5 design DBTs at their 12 hubs (1998-2022, ASHRAE-like 25-year window)
import pandas as pd  # noqa: E402

hubs = pd.read_csv(ROOT / "data/sites/md_hubs.csv")
names = {"md_phoenix": "Phoenix", "md_las_vegas": "Las Vegas", "md_tucson": "Tucson", "md_el_paso": "El Paso"}
rows = []
WIN = (1990, 2025)  # registered window (PR-M1); 1998-2022 reported as a secondary
for _, h in hubs.iterrows():
    W = E.load_weather(ROOT / f"data/weather_mdhubs/{h.site_id}.npz")
    sel = (W["years"] >= WIN[0]) & (W["years"] <= WIN[1])
    t = W["t_db"][sel].ravel()
    for pct in (0.4, 1.0, 2.0):
        dbt = float(np.percentile(t, 100 - pct))
        site = names.get(h.site_id, h.site_id)
        for appr in (1.1, 1.7, 2.8):
            ours = "FEAS" if D.design_point_dry_feasible(dbt, 41.7, appr) else "INF"
            rows.append({"hub": h.site_id, "pct": pct, "dbt_era5": round(dbt, 2),
                         "dbt_ashrae_quoted": quoted.get((site, pct)), "min_approach": appr, "ours_era5": ours,
                         "theirs": md_label(site, pct, appr)})
sec = []
for _, h in hubs.iterrows():
    W = E.load_weather(ROOT / f"data/weather_mdhubs/{h.site_id}.npz")
    t2 = W["t_db"][(W["years"] >= 1998) & (W["years"] <= 2022)].ravel()
    for pct in (0.4, 1.0, 2.0):
        sec.append({"hub": h.site_id, "pct": pct, "dbt_era5_1998_2022": round(float(np.percentile(t2, 100 - pct)), 2)})
agree = sum(r["ours_era5"] == r["theirs"] for r in rows)
bias = [r["dbt_era5"] - r["dbt_ashrae_quoted"] for r in rows if r["dbt_ashrae_quoted"] and r["min_approach"] == 1.1]
res["M1-iii-b"] = {"rows": rows, "agreement": f"{agree}/{len(rows)}",
                   "era5_minus_ashrae_dbt_K": {"n": len(bias), "mean": float(np.mean(bias)), "min": float(np.min(bias)),
                                               "max": float(np.max(bias))},
                   "mismatches": [r for r in rows if r["ours_era5"] != r["theirs"]], "window": WIN,
                   "secondary_1998_2022": sec}

# ---------------- M1-iv: tower water at Abilene (current climate), kct = 1
W_ab = E.load_weather(ROOT / "data/weather/openai_stargate_abilene.npz")
ev = E.eval_tower(P, W_ab, 1.0, lim)  # re-run with the coupled tower model (M1 R1 finding 1)
ox = E.opex_tower(P, W_ab, ev, 0.0, lim)
l_per_kwh = float(np.mean(ox["water_m3"])) * 1000.0 / (8760 * 1000.0)
res["M1-iv"] = {"tower_L_per_kWh_IT": l_per_kwh, "target": 1.12, "band": [0.56, 2.24],
                "pass": 0.56 <= l_per_kwh <= 2.24, "tower_violating_years_no_chiller": int(np.sum(ev["cch_req"] > 1e-9))}

# ---------------- expectation checks (Abilene, Lulea)
W_lu = E.load_weather(ROOT / "data/weather/lulea.npz")
lu = E.eval_dry(P, W_lu, 1.0, False, 0.0, lim)
ab_dry = E.eval_dry(P, W_ab, kmax, False, 0.0, lim)
ab_ad = {k: int(np.sum(E.eval_dry(P, W_ab, k, True, 0.0, lim)["cch_req"] > 1e-9)) for k in D.k_grid(P, lim)}
ab_tank = {th: int(np.sum(E.eval_dry(P, W_ab, kmax, False, th, lim)["cch_req"] > 1e-9)) for th in P["tank_hours_grid"]}
m = D.allowed_violations(P["eta_risk"], len(W_ab["years"]))
res["expectations"] = {
    "allowed_violating_years_m": m,
    "lulea_dry_k1_violating_years": int(np.sum(lu["cch_req"] > 1e-9)),
    "abilene_dry_kmax_no_tank_violating_years": int(np.sum(ab_dry["cch_req"] > 1e-9)),
    "abilene_adiabatic_no_tank_violating_years_by_k": ab_ad,
    "abilene_dry_kmax_violating_years_by_tank_hours": ab_tank,
}
res["params"] = P
res["config_hash"] = PR.config_hash(P)
(OUT / "m1_calibration.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
for key in ("M1-ii", "M1-i-a", "M1-i-b", "M1-iii-a", "M1-iv"):
    r = res[key]
    print(key, "PASS" if r.get("pass") else "MISS", {k: v for k, v in r.items() if k not in ("rows", "points", "k_grid")})
print("M1-iii-b", res["M1-iii-b"]["agreement"], res["M1-iii-b"]["era5_minus_ashrae_dbt_K"])
for r in res["M1-iii-b"]["mismatches"]:
    print("   mismatch", r)
print("expectations", res["expectations"])
