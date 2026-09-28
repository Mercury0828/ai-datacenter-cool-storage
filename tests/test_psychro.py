"""PR-M1 M1-v: psychrometrics against PsychroLib (ASHRAE 2017) on a grid T -10..50 degC, RH 5..100 %, p 85..101 kPa."""
import numpy as np
import psychrolib as pl
import pytest

from model import psychro as ps

pl.SetUnitSystem(pl.SI)
T = np.arange(-10, 50.1, 2.5)
RH = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0])
P = np.array([85000.0, 93000.0, 101325.0])


def grid():
    for t in T:
        for rh in RH:
            for p in P:
                td = pl.GetTDewPointFromRelHum(float(t), float(rh))
                yield float(t), float(td), float(p)


def test_saturation_pressure():
    for t in T:
        assert ps.p_ws(t) == pytest.approx(pl.GetSatVapPres(float(t)), rel=1e-6)


def test_wet_bulb_within_0p05K():
    pts = list(grid())
    t, td, p = (np.array(x) for x in zip(*pts))
    ours = ps.wet_bulb(t, td, p)
    ref = np.array([pl.GetTWetBulbFromTDewPoint(a, b, c) for a, b, c in pts])
    assert np.max(np.abs(ours - ref)) < 0.05, np.max(np.abs(ours - ref))


def test_density_and_humidity_ratio():
    for t, td, p in list(grid())[::7]:
        w = ps.w_from_dewpoint(td, p)
        assert w == pytest.approx(pl.GetHumRatioFromTDewPoint(td, p), rel=1e-6)
        assert ps.density(t, w, p) == pytest.approx(pl.GetMoistAirDensity(t, w, p), rel=1e-6)


def test_inverse_helpers():
    p = 97000.0
    for t in (5.0, 20.0, 35.0):
        assert ps.t_from_h_sat(ps.h_sat(t, p), p) == pytest.approx(t, abs=1e-6)
        w = ps.w_sat(t, p)
        assert ps.dewpoint_from_w(w, p) == pytest.approx(t, abs=1e-6)
