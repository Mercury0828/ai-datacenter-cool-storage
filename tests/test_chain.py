"""PR-M1 M1-v: chain energy balance, storage kernel and monotonicity checks."""
import numpy as np
import pytest

from model import components as C
from model import evaluate as E
from model import params as PR

P = PR.params(ch_etaII=0.5)


@pytest.mark.parametrize("k", [0.6, 1.0, 1.4, 2.0])
def test_dry_chain_energy_balance(k):
    rng = np.random.default_rng(0)
    ch = C.DryChain(P, k, 41.0)
    t_a = rng.uniform(-10, 45, 200)
    p = rng.uniform(85000, 102000, 200)
    s = rng.uniform(0.2, 1.0, 200)
    q = rng.uniform(0.1, 1.5, 200)
    rho = C.air_density_dry(t_a, p)
    flow = s * rho / C.RHO_REF
    c_air = k * ch.c_air_ref * flow
    ua = k * ch.ua_ref * flow ** P["dc_ua_air_exp"]
    cmin = np.minimum(c_air, ch.C_g)
    eps_dc = C.eps_crossflow_unmixed(ua / cmin, cmin / np.maximum(c_air, ch.C_g))
    R = ch.resistance(s, rho)
    t1 = t_a + q * R                                  # facility return (hot)
    t_gc = t_a + q * (1 / (eps_dc * cmin) - 1 / ch.C_g)  # glycol cold (into HX)
    t_gh = t_gc + q / ch.C_g                          # glycol hot (into dry cooler)
    q_hx = ch.eps_hx * ch.cmin_hx * (t1 - t_gc)
    q_dc = eps_dc * cmin * (t_gh - t_a)
    assert np.allclose(q_hx, q, rtol=1e-9) and np.allclose(q_dc, q, rtol=1e-9)
    fws = t1 - q / ch.C_f
    assert np.allclose(fws, t_a + q * ch.A(s, rho), rtol=1e-9)
    # second law: facility supply above glycol cold, glycol above air
    assert np.all(fws > t_gc - 1e-9) and np.all(t_gc > t_a - 1e-9)


def test_trim_need_consistency():
    ch = C.DryChain(P, 1.0, 41.0)
    t = np.linspace(20, 48, 57)
    p = np.full_like(t, 101325.0)
    qmax = ch.q_max(t, p)
    need = ch.trim_need(1.0, t, p)
    assert np.all(need[qmax >= 1.0] == 0.0)
    assert np.all(need[qmax < 1.0] > 0.0)
    assert np.all(np.diff(need) >= -1e-12)  # hotter -> never less trim


def test_tank_kernel():
    need = np.array([0.0, 0.5, 0.5, 0.0, 0.3])
    spare = np.array([1.0, 0.0, 0.0, 0.2, 0.0])
    worst, dis, chg, ch = E._tank_year(need, spare, 0.6, 1.0, 0.0)
    # starts full (0.6): covers 0.5, then 0.1 of 0.5 -> unmet 0.4; recharge 0.2; then 0.3 -> covers 0.2, unmet 0.1
    assert worst == pytest.approx(0.4)
    assert dis == pytest.approx(0.6 + 0.2)
    assert chg == pytest.approx(0.2)
    c = E._min_chiller_with_tank(need, spare, 0.6, 1.0)
    assert E._tank_year(need, spare, 0.6, 1.0, c)[0] <= 1e-9
    assert E._tank_year(need, spare, 0.6, 1.0, c - 1e-3)[0] > 0


def test_monotone_in_warming():
    W = E.load_weather("data/weather/openai_stargate_abilene.npz")
    W = {k: (v[:6] if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == len(W["years"]) else v) for k, v in W.items()}
    W2 = E.shift_weather(W, 1.0)
    for k, ad, th in [(1.0, False, 0.0), (1.4, True, 0.0), (1.2, False, 4.0)]:
        a = E.eval_dry(P, W, k, ad, th, 41.0)
        b = E.eval_dry(P, W2, k, ad, th, 41.0)
        assert np.all(b["cch_req"] >= a["cch_req"] - 1e-9)
        oa = E.opex_dry(P, W, a, 0.3, 41.0)
        ob = E.opex_dry(P, W2, b, 0.3, 41.0)
        cost = lambda o: o["elec_MWh"] * P["elec_price"] + o["water_m3"] * P["water_price"]
        assert np.all(cost(ob) >= cost(oa) - 1e-6)
        assert np.all(ob["unmet_max"] >= oa["unmet_max"] - 1e-9)


@pytest.mark.parametrize("lim", [41.0, 29.0])
@pytest.mark.parametrize("kct", [0.5, 1.0, 2.0, 20.0])
def test_tower_never_below_wetbulb(lim, kct):
    """M1 R1 finding 1: leaving tower water must stay above the entering wet-bulb; duty falls as wet-bulb rises."""
    tw = C.TowerChain(P, kct, lim)
    twb = np.linspace(5, 34, 59)
    p = np.full_like(twb, 98000.0)
    fwr = lim + 1.0 / tw.C_f
    qd = tw.chain_duty(fwr, twb, p)
    a = 1.0 / (tw.eps_hx * tw.cmin_hx) - 1.0 / tw.C_cw
    t_cold = fwr - qd * a - qd / tw.C_cw
    assert np.all(t_cold >= twb - 1e-6)
    assert np.all(np.diff(qd) <= 1e-9)
    # a limit below the wet-bulb plus the HX approach can never be met without a compressor
    need = tw.trim_need(1.0, twb, p)
    hx_app = 1.0 * (1.0 / (tw.eps_hx * tw.cmin_hx) - 1.0 / tw.C_f)
    assert np.all(need[twb + hx_app >= lim] > 0)


def test_charging_uses_available_return_temperature():
    """M1 R1 finding 2: charging duty at fixed flows is bounded by the design return temperature (lim + q/C_f)."""
    from model import design as D
    k = D.k_max(P, 41.0)
    ch = C.DryChain(P, k, 41.0)
    fwr = 41.0 + 1.0 / ch.C_f
    spare = float(ch.duty_fixed_inlet(np.array([30.0]), np.array([101325.0]), fwr)[0]) - 1.0
    assert spare == pytest.approx(0.3207, abs=2e-3)  # independently reproduced bound at 30 degC, sea level
    t_out = fwr - (1.0 + spare) / ch.C_f            # water sent to the store
    assert 30.0 < t_out < 41.0                       # colder than the limit, warmer than the air


def test_sequent_peak_equals_minimum_greedy_tank():
    """BV-1: the sequent-peak storage equals the smallest tank for which the greedy dispatch has no unmet need."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        need = np.where(rng.random(500) < 0.15, rng.random(500) * 0.4, 0.0)
        spare = np.where(need > 0, 0.0, rng.random(500) * 0.3)
        kmax, dur, n = E.sequent_peak(need, spare)
        assert n == int(np.sum(need > 0))
        if kmax > 0:
            assert E._tank_year(need, spare, kmax, 10.0, 0.0)[0] <= 1e-9
            assert E._tank_year(need, spare, kmax * (1 - 1e-6) - 1e-9, 10.0, 0.0)[0] > 0
