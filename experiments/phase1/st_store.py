"""PR-ST (ledger 2026-09-28, declared before any fleet run): hydraulic closure of the cool-water store.

For every hub cell (CDS data of record, 1990-2025, base 6 K design, dry chain at k_max) and year, the store is simulated
as an ideal stratified plug-flow tank with explicit water temperatures and flows, in two piping arrangements:

- series (reference): the store sits downstream of the dry chain like the trim chiller, so the chain always carries the
  design facility flow from the 51 C return and its duty is the Eq. (1) value of the energy model. Discharge: a tank pump
  draws cold water from the bottom into the supply main and an equal flow of chain-outlet water (41 + 10n C) enters the
  top. Charging at full fan: chain-outlet water (41 - 10s C) enters the bottom and an equal flow of warm water leaves the
  top into the supply main. The supply temperature follows from the mixing balance.
- decoupler (comparison): the tank sits between the supply and return mains. In discharge the load keeps the design flow
  at 41 C, the chain runs at reduced facility flow f (glycol loop and fans at design), the tank supplies 1 - f and takes
  51 C return at the top. Charging at design chain flow banks s with water at 41 - 10s C.

Both tanks are last-in-first-out at each end, start each year full of water at 41 - 10 x (charge cap) C, and are sized by
the largest cold-water volume drawn in a year (in U, design-flow hours: 86.1 m3 per MW at the 10 K range); the screen takes
the second-largest year, as for h*. Writes experiments/phase1/out/st/<cell>.json, the Abilene 2024 trace
out/st/trace_cell_490_321_2024.json and experiments/phase1/out/st_summary.json.
Usage: python experiments/phase1/st_store.py [--workers 3] [--cells cell_490_321]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from bv1_min_storage import need_spare  # noqa: E402
from model import components as C  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / "experiments/phase1/out/st"
WX = pathlib.Path("D:/_caches/era5/cds/weather_hub_cds")
U_M3_PER_MW = 86.1
TRACE = ("cell_490_321", 2024)


def series_year(need, spare, lim, t_init, trace=False):
    """Series arrangement for one site-year. Returns (max cold volume drawn [U], max supply temperature in discharge hours,
    max |heat delivered - need|, max tank flow fraction, min supply temperature in charging hours, trace dict or None)."""
    cold = [[math.inf, t_init]]  # the last element is the bottom of the tank (newest cold water)
    warm = []  # the last element is the top of the tank (newest warm water)
    vdef = vmax = 0.0
    t_sup_max = -math.inf
    t_sup_min = math.inf
    e_err = 0.0
    phi_max = 0.0
    tr = {k: [] for k in ("t_chain", "t_supply", "t_cold", "t_warm", "phi", "v_drawn", "q_store")} if trace else None
    for h in range(need.shape[0]):
        n, s = float(need[h]), float(spare[h])
        t_chain = t_sup = lim + 10.0 * n if n > 0 else lim - 10.0 * s if (s > 0 and warm) else lim
        t_cold = t_warm = math.nan
        phi = q_store = 0.0
        if n > 0:
            rem, vol, heat_vt = n, 0.0, 0.0
            while rem > 1e-12:
                v, ts = cold[-1]
                de = (t_chain - ts) / 10.0
                dv = min(v, rem / de)
                rem -= dv * de
                vol += dv
                heat_vt += dv * ts
                if dv >= v:
                    cold.pop()
                else:
                    cold[-1][0] = v - dv
            warm.append([vol, t_chain])
            vdef += vol
            phi = vol
            t_cold = heat_vt / vol
            t_warm = t_chain
            t_sup = (1.0 - vol) * t_chain + heat_vt  # mixing balance of the supply main
            e_err = max(e_err, abs(vol * (t_chain - t_cold) / 10.0 - n))
            t_sup_max = max(t_sup_max, t_sup)
            q_store = n
        elif s > 0 and warm:
            rem, vol, heat_vt = s, 0.0, 0.0
            while rem > 1e-12 and warm:
                v, tw = warm[-1]
                de = (tw - t_chain) / 10.0
                dv = min(v, rem / de)
                rem -= dv * de
                vol += dv
                heat_vt += dv * tw
                if dv >= v - 1e-15:
                    warm.pop()
                else:
                    warm[-1][0] = v - dv
            cold.append([vol, t_chain])
            vdef = vdef - vol if warm else 0.0
            phi = vol
            t_cold = t_chain
            t_warm = heat_vt / vol if vol > 0 else math.nan
            # staged control in the hour the tank fills: charge at full fan for the fraction alpha of the hour that the
            # remaining warm water allows, then hold the limit for the load alone (internal review rev4-store R1)
            alpha = (s - rem) / s
            t_sup = (alpha - vol) * t_chain + heat_vt + (1.0 - alpha) * lim
            t_sup_min = min(t_sup_min, t_sup)
            q_store = -(s - rem)
        vmax = max(vmax, vdef)
        phi_max = max(phi_max, phi)
        if trace:
            for k, x in (("t_chain", t_chain), ("t_supply", t_sup), ("t_cold", t_cold), ("t_warm", t_warm), ("phi", phi),
                         ("v_drawn", vdef), ("q_store", q_store)):
                tr[k].append(x)
    return vmax, t_sup_max, e_err, phi_max, t_sup_min, tr


class ChainFlow:
    """Duty of the dry chain at facility flow fraction f with the 51 C inlet (glycol loop and fans at design)."""

    def __init__(self, ch, t_air, p, t_in):
        rho = C.air_density_dry(t_air, p)
        self.r_full = np.asarray(ch.resistance(1.0, rho), float)
        self.dt = np.maximum(t_in - np.asarray(t_air, float), 0.0)
        self.ch = ch
        self.hx1 = 1.0 / (ch.eps_hx * ch.cmin_hx)

    def hx(self, f):
        ch = self.ch
        cf = f * ch.C_f
        cmin, cmax = min(cf, ch.C_g), max(cf, ch.C_g)
        cr, ntu = cmin / cmax, ch.ua_hx / cmin
        if abs(cr - 1.0) < 1e-9:
            eps = ntu / (1.0 + ntu)
        else:
            e = math.exp(-ntu * (1.0 - cr))
            eps = (1.0 - e) / (1.0 - cr * e)
        return 1.0 / (eps * cmin)

    def q(self, h, f):
        return float(self.dt[h]) / (float(self.r_full[h]) - self.hx1 + self.hx(f))


def decoupler_year(need, spare, cf, lim, t_init):
    """Decoupler arrangement for one site-year. Returns (max cold volume drawn [U], max heat deficit [h], min chain flow)."""
    t_ret = lim + 10.0
    cold = [[math.inf, t_init]]
    vdef = vmax = edef = emax = 0.0
    fmin = 1.0
    for h in range(need.shape[0]):
        n, s = float(need[h]), float(spare[h])
        if n > 0:
            def cold_heat(phi):  # sum of dv * T over the first phi of tank water, from the bottom
                acc, left = 0.0, phi
                for v, ts in reversed(cold):
                    dv = min(v, left)
                    acc += dv * ts
                    left -= dv
                    if left <= 1e-15:
                        break
                return acc
            def supply_excess(phi):  # supply - 41 C at chain flow 1 - phi; 10 n > 0 at phi = 0
                f = 1.0 - phi
                return t_ret * f - 10.0 * cf.q(h, f) + cold_heat(phi) - lim
            grid = [j / 50.0 for j in range(50)] + [1.0 - 1e-6]
            lo = hi = None
            for a_, b_ in zip(grid[:-1], grid[1:]):  # first sign change: the smallest tank flow that holds 41 C
                if supply_excess(b_) <= 0:
                    lo, hi = a_, b_
                    break
            if lo is None:
                raise RuntimeError(f"decoupler: no chain flow holds {lim} C at hour {h}")
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if supply_excess(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            phi = 0.5 * (lo + hi)
            f = 1.0 - phi
            drawn = 1.0 - cf.q(h, f)
            left = phi
            while left > 1e-15:
                v, ts = cold[-1]
                dv = min(v, left)
                left -= dv
                if dv >= v:
                    cold.pop()
                else:
                    cold[-1][0] = v - dv
            vdef += phi
            edef += drawn
            fmin = min(fmin, f)
        elif s > 0 and vdef > 0:
            vol = min(s / (1.0 + s), vdef)
            cold.append([vol, lim - 10.0 * s])
            vdef -= vol
            edef = max(0.0, edef - vol * (1.0 + s))
            if vdef <= 1e-12:
                vdef, edef = 0.0, 0.0
        vmax = max(vmax, vdef)
        emax = max(emax, edef)
    return vmax, emax, fmin


def run_item(cell):
    out = OUT / f"{cell}.json"
    if out.exists():
        return cell, "skip"
    t0 = time.time()
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    k0 = max(D.k_grid(P, lim))
    t_init = lim - 10.0 * P["tank_charge_max"]
    W = E.load_weather(WX / f"{cell}.npz")
    ch, need, spare, fwr = need_spare(P, W, k0, lim)
    years = [int(y) for y in np.asarray(W["years"])]
    m = D.allowed_violations(P["eta_risk"], need.shape[0])
    rec = {"cell": cell, "k": k0, "t_init": t_init, "m": m, "years": years, "E": [], "V_ser": [], "V_dec": [], "E_dec": [],
           "t_sup_max_ser": [], "t_sup_min_ser": [], "e_err_ser": [], "phi_max_ser": [], "f_min_dec": []}
    for y in range(need.shape[0]):
        e_y = E.sequent_peak(need[y], spare[y])[0]
        want = (cell, years[y]) == TRACE
        v_s, tsup, eerr, phimax, tsupmin, tr = series_year(need[y], spare[y], lim, t_init, trace=want)
        if want:
            (OUT / f"trace_{cell}_{years[y]}.json").write_text(json.dumps(
                {k: [None if (isinstance(x, float) and math.isnan(x)) else round(x, 5) for x in v] for k, v in tr.items()}),
                encoding="utf-8")
        if e_y > 0:
            cf = ChainFlow(ch, W["t_db"][y], W["p"][y], fwr)
            v_d, e_d, fmin = decoupler_year(need[y], spare[y], cf, lim, t_init)
        else:
            v_d, e_d, fmin = 0.0, 0.0, 1.0
        for k, x in (("E", e_y), ("V_ser", v_s), ("V_dec", v_d), ("E_dec", e_d), ("t_sup_max_ser", tsup if e_y > 0 else None),
                     ("t_sup_min_ser", tsupmin if e_y > 0 else None),
                     ("e_err_ser", eerr), ("phi_max_ser", phimax), ("f_min_dec", fmin)):
            rec[k].append(x)

    def second(a):
        return float(np.sort(np.asarray(a, float))[::-1][m])
    rec.update({"h_star": second(rec["E"]), "V_ser_star": second(rec["V_ser"]), "V_dec_star": second(rec["V_dec"]),
                "E_dec_star": second(rec["E_dec"]), "elapsed_s": round(time.time() - t0, 1)})
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, out)
    return cell, "done"


def summarize():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    mw = sites.groupby("cell_id").B_full_MW.sum()
    recs = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in OUT.glob("cell_*.json")}
    df = pd.DataFrame({c: {"h": r["h_star"], "vs": r["V_ser_star"], "vd": r["V_dec_star"], "ed": r["E_dec_star"]}
                       for c, r in recs.items()}).T
    df["MW"] = mw.reindex(df.index)
    df["vmin"] = df[["vs", "vd"]].min(axis=1)
    ok = df.h <= 8 + 1e-9
    res = {"cells": len(df),
           "cap_GW|h_star<=8": float(df.MW[ok].sum() / 1e3),
           "cap_GW|V_ser<=8": float(df.MW[df.vs <= 8 + 1e-9].sum() / 1e3),
           "cap_GW|V_dec<=8": float(df.MW[df.vd <= 8 + 1e-9].sum() / 1e3),
           "cap_GW|V_min<=8": float(df.MW[df.vmin <= 8 + 1e-9].sum() / 1e3),
           "fleet_heat_GWh|h_star<=8": float((df.MW * df.h)[ok].sum() / 1e3),
           "fleet_volume_Mm3|10K|h_star<=8": float((df.MW * df.h)[ok].sum() * U_M3_PER_MW / 1e6),
           "fleet_volume_Mm3|series|h_star<=8": float((df.MW * df.vs)[ok].sum() * U_M3_PER_MW / 1e6),
           "fleet_volume_Mm3|min|h_star<=8": float((df.MW * df.vmin)[ok].sum() * U_M3_PER_MW / 1e6),
           "t_sup_max_ser": max(max(x for x in r["t_sup_max_ser"] if x is not None) for r in recs.values()
                                if any(x is not None for x in r["t_sup_max_ser"])),
           "t_sup_min_ser": min(min(x for x in r["t_sup_min_ser"] if x is not None) for r in recs.values()
                                if any(x is not None for x in r["t_sup_min_ser"])),
           "e_err_ser_max": max(max(r["e_err_ser"]) for r in recs.values()),
           "phi_max_ser": max(max(r["phi_max_ser"]) for r in recs.values()),
           "f_min_dec": min(min(r["f_min_dec"]) for r in recs.values())}
    pos = ok & (df.h > 0)
    res["effective_swing_K|series"] = float(10.0 * (df.MW * df.h)[pos].sum() / (df.MW * df.vs)[pos].sum())
    res["effective_swing_K|min"] = float(10.0 * (df.MW * df.h)[pos].sum() / (df.MW * df.vmin)[pos].sum())
    r = (df.vs / df.h)[pos].sort_values()
    cum = df.MW[r.index].cumsum() / df.MW[r.index].sum()
    res["ratio_V_ser_over_h|capacity_weighted_median"] = float(r[cum >= 0.5].iloc[0])
    res["ratio_V_ser_over_h|min_max"] = [float(r.min()), float(r.max())]
    rd = (df.vmin / df.h)[pos]
    res["ratio_V_min_over_h|min_max"] = [float(rd.min()), float(rd.max())]
    res["cells_decoupler_smaller"] = int((df.vd < df.vs)[pos].sum())
    ab = df.loc["cell_490_321"]
    res["abilene"] = {"MW": float(ab.MW), "h_star": float(ab.h), "V_ser": float(ab.vs), "V_dec": float(ab.vd),
                      "E_dec": float(ab.ed), "volume_Mm3|10K": float(ab.MW * ab.h * U_M3_PER_MW / 1e6),
                      "volume_Mm3|series": float(ab.MW * ab.vs * U_M3_PER_MW / 1e6),
                      "volume_Mm3|decoupler": float(ab.MW * ab.vd * U_M3_PER_MW / 1e6)}
    res["crossing_8"] = {c: {"MW": float(df.MW[c]), "h": float(df.h[c]), "V_ser": float(df.vs[c]), "V_dec": float(df.vd[c])}
                         for c in df.index if (df.h[c] <= 8 + 1e-9) != (df.vmin[c] <= 8 + 1e-9)
                         or (df.h[c] <= 8 + 1e-9) != (df.vs[c] <= 8 + 1e-9)}
    (ROOT / "experiments/phase1/out/st_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cells", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    if a.cells:
        cells = [c for c in cells if c in a.cells.split(",")]
    items = [c for c in cells if not (OUT / f"{c}.json").exists()]
    print(f"{len(items)} items pending", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)} {r}", flush=True)
    if not a.cells:
        print(json.dumps(summarize(), indent=1), flush=True)


if __name__ == "__main__":
    main()
