"""Counterfactual for the Phase-0 scale estimate: annual on-site water of a conventional tower-based design (BL4-like,
open tower kct = 1 + plate HX, all heat evaporative) at each weighted site, current climate and 2040 stress.
Output: experiments/phase0/out/r1/counterfactual_tower_water.csv (m3 per MW-yr of liquid-loop heat; L/kWh-IT)."""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

P = PR.params(ch_etaII=0.5076)
lim = PR.fws_max(P)
rows = []
for _, s in pd.read_csv(ROOT / "data/sites/phase0_sites.csv").iterrows():
    W0 = E.load_weather(ROOT / f"data/weather/{s.site_id}.npz")
    for climate, dT in (("current", 0.0), ("2040", 1.2)):
        W = E.shift_weather(W0, dT) if dT else W0
        ev = E.eval_tower(P, W, 1.0, lim)
        ox = E.opex_tower(P, W, ev, 1e9, lim)
        m3 = float(np.mean(ox["water_m3"]))
        rows.append({"site": s.site_id, "climate": climate, "B_full_MW": s.B_full_MW, "tower_water_m3_per_MW_yr": m3,
                     "tower_L_per_kWh": m3 * 1000 / (8760 * 1000), "tower_fan_MWh_per_MW_yr": float(np.mean(ox["fan_MWh"]))})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "experiments/phase0/out/r1/counterfactual_tower_water.csv", index=False)
