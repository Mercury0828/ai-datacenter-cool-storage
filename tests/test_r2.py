"""R2 machinery (DD-010): type-1 Wasserstein worst-case violation probability, margins, frontier interpolation."""
import importlib.util
import pathlib

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("r2a", pathlib.Path(__file__).resolve().parents[1] / "experiments/phase0/r2_aggregate.py")
r2a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2a)


def test_wcp_closed_form_cases():
    # N = 20 margins; eta*N = 1: compliant iff min margin >= N*eps (derivation in DD-010 notes)
    d = np.array([2.0] + [5.0] * 19)
    assert r2a.wcp(d, 0.1) == pytest.approx(0.05)          # budget moves exactly one full mass (2.0/20 = 0.1)
    assert r2a.wcp(d, 0.05) == pytest.approx(0.025)        # half a mass
    assert r2a.wcp(np.array([0.0] + [5.0] * 19), 0.0) == pytest.approx(0.05)   # one already-violating year
    assert r2a.wcp(np.array([0.0] + [5.0] * 19), 0.01) > 0.05                  # any budget pushes more mass
    assert r2a.wcp(np.array([np.inf] * 20), 10.0) == 0.0                       # unreachable within the support
    # fractional knapsack: cheapest first
    d = np.array([1.0, 3.0, 3.0, 10.0])  # N = 4
    assert r2a.wcp(d, 0.25) == pytest.approx(0.25)          # moves year 1 fully (cost 1/4)
    assert r2a.wcp(d, 0.25 + 0.75) == pytest.approx(0.5)    # + year 2 (cost 3/4)


def test_margins_interpolation():
    deltas = np.array([0.0, 1.0, 2.0])
    Rj = np.array([[0.0, 0.1], [0.05, 0.3], [0.2, 0.5]])  # (G, years)
    d = r2a.margins(Rj, 0.1, np.array([0, 1]), deltas)
    assert d[0] == pytest.approx(1.0 + (0.1 - 0.05) / (0.2 - 0.05))  # crosses between 1 and 2 K
    assert d[1] == pytest.approx(0.0 + (0.1 - 0.1) / (0.3 - 0.1))    # r0 = C exactly -> not already violating
    assert r2a.margins(Rj, 0.6, np.array([0, 1]), deltas).tolist() == [np.inf, np.inf]


def test_frontier_interpolation():
    pts = [(0.5, 10.0), (0.2, 12.0), (0.05, 16.0), (0.3, 13.0)]  # (violation, cost); (0.3, 13) is dominated
    f = r2a.frontier(pts)
    assert f == [(0.5, 10.0), (0.2, 12.0), (0.05, 16.0)]
    assert r2a.cost_at_violation(pts, 0.125) == pytest.approx(12.0 + (0.2 - 0.125) / 0.15 * 4.0)
    assert r2a.cost_at_violation(pts, 0.01) is None
    assert r2a.violation_at_cost(pts, 14.0) == pytest.approx(0.2 - (14.0 - 12.0) / 4.0 * 0.15)
    assert r2a.violation_at_cost(pts, 9.0) is None
