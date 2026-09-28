"""M2 follow-up (DD-015 addendum): sensitivity of the headline classifications to the derived surface pressure.
(1) Every hub cell: table pressure × 0.99 (wet-bulb recomputed), historical and +1.2 K, uncorrected weather → h*(k_max) and the
Phase-0 exact short-circuit screen (dry k_max + 8 h store → F_dry; else adiabatic k_max + 8 h; else tower kct_max → F_nc), compared
with the unperturbed BV-1 / hub-screen outputs. (2) Keflavik (cell_616_630): pressure re-derived from ERA5 mean-sea-level pressure
with ERA5's orography (41.75 m) instead of the missing value, same outputs.
Usage: python experiments/phase1/m2_pressure_sensitivity.py [--workers 2]
Output: experiments/phase1/out/m2/pressure/<case>__<cell>__<climate>.json and pressure_summary.json
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from bv1_min_storage import CLIMATES, COMMIT  # noqa: E402
from hw_screen import screen_path  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402
from model import psychro as ps  # noqa: E402

OUT = ROOT / "experiments/phase1/out/m2/pressure"
WX = pathlib.Path("D:/_caches/era5/weather_hub")
KEF = "cell_616_630"
KEF_OROG = 41.75  # m, ERA5 geopotential_at_surface / g (ARCO), DD-015


def keflavik_pressure(W):
    """Invert the tables' derivation p = p_msl·f(z) with the missing z = −999 m, then re-apply it with z = 41.75 m."""
    t = W["t_db"]
    f = lambda z: (1 - 0.0065 * z / (t + 0.0065 * z + 273.15)) ** 5.257  # noqa: E731
    return W["p"] / f(-999.0) * f(KEF_OROG)


def run_item(args):
    case, cell, climate, dT = args
    out = OUT / f"{case}__{cell}__{climate}.json"
    if out.exists():
        return case, cell, climate, "skip"
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    W = E.load_weather(WX / f"{cell}.npz")
    W = dict(W)
    W["p"] = W["p"] * 0.99 if case == "p_minus_1pct" else keflavik_pressure(W)
    W["t_wb"] = ps.wet_bulb(W["t_db"], W["t_dp"], W["p"])
    if dT:
        W = E.shift_weather(W, dT)
    m = D.allowed_violations(P["eta_risk"], W["t_db"].shape[0])
    r = screen_path(P, W, lim, m)
    rec = {"case": case, "cell": cell, "climate": climate, "commit": COMMIT, **r}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return case, cell, climate, "done"


def summarize():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    res = {}
    for case in ("p_minus_1pct", "keflavik_orography"):
        for cl, hub_cl in (("historical", "current"), ("stress_plus1.2K", "2040")):
            ch_dry, ch_nc, n = [], [], 0
            for f in OUT.glob(f"{case}__*__{cl}.json"):
                r = json.loads(f.read_text(encoding="utf-8"))
                n += 1
                bv1 = json.loads((ROOT / f"experiments/phase1/out/bv1/uncorrected/{r['cell']}__{cl}.json").read_text(encoding="utf-8"))
                h0 = bv1["k"][sorted(bv1["k"], key=float)[-1]]["h_star"]
                hub = json.loads((ROOT / f"experiments/phase0/out/hub/items/{r['cell']}__{hub_cl}.json").read_text(encoding="utf-8"))
                if (h0 <= 8 + 1e-9) != r["F_dry"]:
                    ch_dry.append(r["cell"])
                if bool(hub["F_nc"]) != r["F_nc"]:
                    ch_nc.append(r["cell"])
            mw = lambda cs: float(sites.B_full_MW[sites.cell_id.isin(cs)].sum())  # noqa: E731
            res[f"{case}|{cl}"] = {"items": n, "F_dry_changes": ch_dry, "F_dry_changed_MW": mw(ch_dry),
                                   "F_nc_changes": ch_nc, "F_nc_changed_MW": mw(ch_nc)}
    (OUT.parent / "pressure_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    items = [("p_minus_1pct", c, cl, dT) for c in cells for cl, dT in CLIMATES.items()]
    items += [("keflavik_orography", KEF, cl, dT) for cl, dT in CLIMATES.items()]
    items = [i for i in items if not (OUT / f"{i[0]}__{i[1]}__{i[2]}.json").exists()]
    print(f"{len(items)} items pending; commit {COMMIT}", flush=True)
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(n, r, flush=True)
    summarize()


if __name__ == "__main__":
    main()
