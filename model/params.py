"""Declared parameters of the HOTLOOP Phase-0 model, with provenance and sensitivity ranges.

Every value here is a *declared modelling choice* (DESIGN_DECISIONS DD-006/DD-007) or a *calibrated* value (PR-M1).
Nothing is tuned toward an expected zone. `ch_etaII` is filled by the M1 calibration (experiments/phase0/m1_calibration.py).
"""
import copy
import hashlib
import json

DEFAULTS = {
    # ---- chain (guide §4, P1, P2)
    "T_TCS_max": 45.0,       # degC rack coolant supply limit (P1); variants 40 / 50
    "dT_CDU": 4.0,           # K CDU approach (P1: 45 -> 41 degC)
    "dT_fw": 10.0,           # K facility-water range at design (P2: rack coolant 45 -> ~55 degC)
    "dT_HX_design": 2.0,     # K plate-HX approach at design duty (guide: declared 1-3 K); sens 1-3
    "dT_glycol": 10.0,       # K glycol-loop range at design (declared)
    # ---- dry cooler reference module (k = 1), per MW design heat (PR-M1 M1-i-b)
    "dc_ref_approach": 10.0,  # K at rating: glycol out 35 - air 25 (Thermokey standard conditions)
    "dc_ref_itd": 15.0,       # K at rating: glycol in 40 - air 25
    "dc_ref_air_dT": 10.0,    # K air-side temperature rise at rating (sets C_air,ref; declared)
    "dc_ua_air_exp": 0.6,     # UA ~ (air mass flow)^0.6 (air-side-dominated finned coil; declared, sens 0.5-0.8)
    "dc_fan_frac_ref": 0.015, # fan power at full speed per unit heat at k = 1 (EC fans ~1.4-1.8 %; sens 0.01-0.03)
    "dc_min_speed": 0.2,
    "dc_approach_floor": 4.0, # K, P5 "Minimum 4 degC": dry-cooler approach (glycol out - air in) at design duty,
                              # 35 degC air, sea level -> upper bound k_max on the module count
    "eta_ad": 0.75,           # adiabatic pad saturation efficiency (EVAPCO WP: ~72-78 % measured); sens 0.65-0.90
    "ad_overspray": 1.15,     # supplied / evaporated water for pads with bleed (declared; sens 1.0-1.3)
    # ---- trim chiller (air-cooled, series on the facility loop; PR-M1 M1-ii)
    "ch_etaII": None,         # fitted: ASHRAE 90.1-2019 air-cooled >=150 ton Path A 10.1 EER at AHRI conditions
    "ch_dT_evap": 4.0,        # K evaporator approach
    "ch_dT_cond": 15.0,       # K condensing temperature above ambient (air-cooled)
    "ch_cop_max": 6.0,        # small-lift cap (declared; sens 4-8)
    # ---- cooling tower (open tower + plate HX)
    "ct_dT_cw": 10.0,         # K condenser-water range
    "ct_ma_ref": 25.0,        # kg/s dry air per MW design at kct = 1 (declared; L/G ~ 1 at 10 K range)
    "ct_ntu_ref": 1.3,        # tower NTU (mass-transfer conductance / air flow) at kct = 1, full fan (Merkel ~1-2.5)
    "ct_ua_exp": 0.6,         # tower conductance ~ (air flow)^0.6 at part fan
    "ct_fan_frac_ref": 0.010, # fan power at full speed per unit heat at kct = 1
    "ct_min_speed": 0.2,
    "ct_cycles": 4.0,         # cycles of concentration (blowdown = evap/(CoC-1))
    # ---- hybrid air path (85/15 variant; guide §4)
    "f_air": 0.0,             # air-side fraction of IT heat (0 headline; 0.15 GB200 / 0.10 GB300, P3)
    "T_sa_max": 32.0,         # degC supply-air limit = ASHRAE A1 allowable upper bound (declared; sens 27)
    "dT_coil": 3.0,           # K air-path coil approach (supply air - air-path water supply)
    # ---- storage (sensible cool-water tank at facility-supply temperature)
    "tank_charge_max": 1.0,   # max charge rate, per-unit of design heat
    # ---- design bounds and grids
    "tank_hours_grid": [0.0, 2.0, 4.0, 8.0],   # MWh_th per MW design heat
    "kct_grid": [0.5, 1.0, 1.5, 2.0],
    # ---- costs (Phase-0 central values; ranges are flip-point axes, Q4)
    "crf": 0.1168,            # capital recovery factor, 8 %, 15 years
    "elec_price": 80.0,       # $/MWh (US industrial average order; sens 50-120)
    "water_price": 2.0,       # $/m3 (sens 0.5-6; water-constrained variant = ban)
    "capex_dc_ref": 80e3,     # $/MW design heat for one reference module (k = 1) installed (sens 50-150k)
    "capex_ad_frac": 0.25,    # adiabatic add-on as a fraction of the dry-cooler capex (declared)
    "capex_chiller": 300e3,   # $/MW of trim-chiller cooling capacity, installed (sens 200-500k)
    "capex_tower": 60e3,      # $/MW at kct = 1 incl. plate HX and water treatment (sens 40-100k)
    "capex_tank_per_mwh": 35e3,  # $/MWh_th (~$400/m3 at 10 K swing; sens 17-70k)
    "pi_c": 2000.0,           # $/MWh of IT energy not delivered (S1 only; sens 500-5000)
    # ---- standards
    "eta_risk": 1.0 / 20.0,   # 1-in-20-year risk standard (headline)
    "service": "S0",          # S0 no capping (Phase-0 headline)
}


def params(**over):
    P = copy.deepcopy(DEFAULTS)
    for k, v in over.items():
        if k not in P:
            raise KeyError(k)
        P[k] = v
    return P


def fws_max(P):
    return P["T_TCS_max"] - P["dT_CDU"]


def config_hash(P):
    return hashlib.sha256(json.dumps(P, sort_keys=True).encode()).hexdigest()[:12]
