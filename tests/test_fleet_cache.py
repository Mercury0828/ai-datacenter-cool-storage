"""The cached menu evaluator (model/fleet.py) must reproduce evaluate.eval_dry / opex_dry bit for bit."""
import pathlib

import numpy as np
import pytest

from model import design as D
from model import evaluate as E
from model import fleet as F
from model import params as PR

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _weather(site, years=8, dT=0.0):
    W = E.load_weather(ROOT / f"data/weather/{site}.npz")
    W = {k: (v[-years:] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == W["t_db"].shape[0] else v)
         for k, v in W.items()}
    return E.shift_weather(W, dT) if dT else W


@pytest.mark.parametrize("site,dT", [("openai_stargate_abilene", 0.0), ("openai_stargate_abilene", 1.2), ("goodnight", 0.0)])
def test_cached_menu_is_bit_identical(site, dT):
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = _weather(site, dT=dT)
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    cache = F.SiteCache(P, W, lim)
    ks = D.k_grid(P, lim)
    for k, ad, th in [(ks[-1], False, 8.0), (ks[-1], True, 2.0), (ks[3], True, 4.0), (ks[0], True, 0.0), (ks[3], False, 0.0),
                      (ks[5], False, 4.0)]:
        ref = E.eval_dry(P, W, k, ad, th, lim)
        new = F.eval_dry_c(cache, k, ad, th)
        for key in ("need", "need_dry", "ok_dry", "spare", "qmax_dry", "cch_req"):
            assert np.array_equal(ref[key], new[key]), (k, ad, th, key)
        if ad:
            assert np.array_equal(ref["qmax_ad"], new["qmax_ad"]) and np.array_equal(ref["tad"], new["tad"])
        cc = D.chiller_for_risk(ref["cch_req"], m)
        for c_ch in (0.0, cc, 0.5 * cc):
            o_ref = E.opex_dry(P, W, ref, c_ch, lim)
            o_new = F.opex_dry_c(cache, new, c_ch)
            for key in o_ref:
                assert np.array_equal(o_ref[key], o_new[key]), (k, ad, th, c_ch, key)
            if o_ref["viol"].any():
                d = F.opex_dry_c(cache, new, c_ch, detail=True)
                assert np.allclose(d["unmet_h"].max(axis=1), o_ref["unmet_max"], atol=1e-9)
