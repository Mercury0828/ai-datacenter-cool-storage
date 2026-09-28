"""Vectorized moist-air psychrometrics (SI units), following ASHRAE Handbook—Fundamentals (2017), Ch. 1.

Temperatures in degC, pressures in Pa, humidity ratio W in kg water / kg dry air, enthalpy in kJ/kg dry air.
Verified against PsychroLib (tests/test_psychro.py; tolerance 0.05 K on wet-bulb, PR-M1 M1-v).
"""
import numpy as np

R_DA = 287.042  # J/(kg K), dry air
CP_DA = 1.006  # kJ/(kg K)
CP_V = 1.86  # kJ/(kg K), water vapour
H_FG0 = 2501.0  # kJ/kg at 0 degC
CP_W = 4.186  # kJ/(kg K), liquid water
K0 = 273.15


def p_ws(t):
    """Saturation vapour pressure [Pa] over liquid water (t > 0.01) or ice (t <= 0.01 degC, triple point), Hyland & Wexler (ASHRAE 2017)."""
    t = np.asarray(t, dtype=float)
    T = t + K0
    ice = t <= 0.01  # triple point, as PsychroLib
    ln_w = (-5.8002206e3 / T + 1.3914993 - 4.8640239e-2 * T + 4.1764768e-5 * T**2
            - 1.4452093e-8 * T**3 + 6.5459673 * np.log(T))
    ln_i = (-5.6745359e3 / T + 6.3925247 - 9.6778430e-3 * T + 6.2215701e-7 * T**2
            + 2.0747825e-9 * T**3 - 9.4840240e-13 * T**4 + 4.1635019 * np.log(T))
    return np.exp(np.where(ice, ln_i, ln_w))


def w_from_pw(pw, p):
    return 0.621945 * pw / (p - pw)


def w_sat(t, p):
    return w_from_pw(p_ws(t), p)


def w_from_dewpoint(td, p):
    return w_from_pw(p_ws(td), p)


def enthalpy(t, w):
    return CP_DA * t + w * (H_FG0 + CP_V * t)


def h_sat(t, p):
    """Enthalpy of saturated air at temperature t [kJ/kg dry air]."""
    return enthalpy(t, w_sat(t, p))


def density(t, w, p):
    """Moist-air density [kg moist air / m3]."""
    v = R_DA * (t + K0) * (1 + 1.607858 * w) / p  # m3 / kg dry air
    return (1 + w) / v


def _w_psychrometric(t, tstar, p):
    """Humidity ratio implied by dry-bulb t and a candidate wet-bulb t* (ASHRAE 2017 eqs. 33/35)."""
    ws = w_sat(tstar, p)
    above = (H_FG0 - 2.326 * tstar) * ws - CP_DA * (t - tstar)
    above = above / (H_FG0 + CP_V * t - CP_W * tstar)
    below = (2830.0 - 0.24 * tstar) * ws - CP_DA * (t - tstar)
    below = below / (2830.0 + CP_V * t - 2.1 * tstar)
    return np.where(tstar >= 0.0, above, below)


def wet_bulb(t, td, p, iters=48):
    """Thermodynamic wet-bulb [degC] from dry-bulb t, dew point td [degC] and pressure p [Pa]; vectorized bisection."""
    t, td, p = np.broadcast_arrays(np.asarray(t, float), np.asarray(td, float), np.asarray(p, float))
    td = np.minimum(td, t)
    w = w_from_dewpoint(td, p)
    lo, hi = td.copy(), t.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        # W implied by t* = mid decreases as mid falls; if it exceeds the true W the guess is too high
        too_high = _w_psychrometric(t, mid, p) > w
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return 0.5 * (lo + hi)


def rel_humidity(t, td):
    return p_ws(td) / p_ws(t)


def dewpoint_from_w(w, p, iters=48):
    """Dew point [degC] for humidity ratio w at pressure p (bisection on p_ws)."""
    pw = w * p / (0.621945 + w)
    lo = np.full(np.shape(pw), -60.0)
    hi = np.full(np.shape(pw), 60.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        big = p_ws(mid) > pw
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    return 0.5 * (lo + hi)


def t_from_h_sat(h, p, iters=48):
    """Temperature of saturated air with enthalpy h [kJ/kg dry air] (bisection)."""
    lo = np.full(np.shape(h), -30.0)
    hi = np.full(np.shape(h), 80.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        big = h_sat(mid, p) > h
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    return 0.5 * (lo + hi)


# ---- tabulated saturation pressure for hot inner loops (linear interpolation on a 0.001 K grid, -50..100 degC;
#      relative error < 1e-9 vs p_ws). Used by the cooling-tower solver only.
_T_TAB = np.linspace(-50.0, 100.0, 150001)
_P_TAB = p_ws(_T_TAB)


def p_ws_fast(t):
    return np.interp(t, _T_TAB, _P_TAB)


def h_sat_fast(t, p):
    pw = p_ws_fast(t)
    return enthalpy(t, 0.621945 * pw / (p - pw))


def t_from_h_sat_fast(h, p, iters=24):
    lo = np.full(np.shape(h), -30.0)
    hi = np.full(np.shape(h), 80.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        big = h_sat_fast(mid, p) > h
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    return 0.5 * (lo + hi)
