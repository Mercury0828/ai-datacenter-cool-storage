"""R2 aggregation: DR / SAA / BL1 selection, leave-years-out radius rule, test evaluation and rules (a)/(b)
exactly as pre-registered in DESIGN_DECISIONS DD-010 and ledger PR-R2 (commit 007f9c0).

Reads experiments/phase0/out/r2/sites/*.npz; writes experiments/phase0/out/r2/r2_results.json and prints a summary.
"""
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import params as PR  # noqa: E402

P = PR.params(ch_etaII=0.5076)
ETA = P["eta_risk"]
CRF, CAPCH = P["crf"], P["capex_chiller"]
EPS_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2]
SAA_M = [0, 1, 2, 3]
FAMILIES = ["F0", "F1", "F2", "F3", "F4"]
IN = ROOT / "experiments/phase0/out/r2/sites"
OUT = ROOT / "experiments/phase0/out/r2"


def load():
    sites = pd.read_csv(ROOT / "data/sites/phase0_sites.csv")
    sites = sites[sites.weighted]
    data = {}
    for _, r in sites.iterrows():
        z = np.load(IN / f"{r.site_id}.npz", allow_pickle=False)
        R = np.maximum.accumulate(z["R"], axis=1)  # enforce monotonicity in the warming shift (numerical noise only)
        data[r.site_id] = {"R": R, "R4": z["R4"], "op_tr": z["op_tr"], "op_te": z["op_te"], "NB": z["NB"],
                           "cap0": z["capex_nochiller"], "deltas": z["deltas"], "years": z["years"],
                           "designs": [json.loads(x) for x in z["designs"]], "w": float(r.B_full_MW),
                           "bl1v": list(z["bl1_variants"])}
    return data


def margins(Rj, C, yrs, deltas):
    """d_i = smallest warming shift at which year i needs more than C (linear interpolation on the grid)."""
    r = Rj[:, yrs]  # (G, n)
    d = np.full(r.shape[1], np.inf)
    already = r[0] > C + 1e-12
    d[already] = 0.0
    for i in np.where(~already)[0]:
        g = np.argmax(r[:, i] > C + 1e-12)
        if r[g, i] <= C + 1e-12:
            continue  # never exceeds within 5 K
        r0, r1 = r[g - 1, i], r[g, i]
        d[i] = deltas[g - 1] + (C - r0) / max(r1 - r0, 1e-15) * (deltas[g] - deltas[g - 1])
    return d


def wcp(d, eps):
    """Worst-case violation probability over a type-1 Wasserstein ball of radius eps (fractional knapsack)."""
    n = len(d)
    mass, budget = 0.0, eps
    for di in np.sort(d):
        if di <= 0:
            mass += 1.0 / n
            continue
        if not np.isfinite(di):
            break
        take = min(1.0 / n, budget / di)
        mass += take
        budget -= take * di
        if take < 1.0 / n - 1e-15:
            break
    return mass


def sel_cost(s, j, C):
    return CRF * (s["cap0"][j] + C * CAPCH) + s["op_tr"][j, 0 if C <= 1e-9 else 1]


def test_cost(s, j, C):
    return CRF * (s["cap0"][j] + C * CAPCH) + s["op_te"][j, 0 if C <= 1e-9 else 1]


def dr_select(s, eps, yrs):
    best = None
    for j in range(len(s["designs"])):
        Rj = s["R"][j]
        if wcp(margins(Rj, 0.0, yrs, s["deltas"]), eps) <= ETA + 1e-12:
            C = 0.0
        else:
            lo, hi = 0.0, float(Rj[-1, yrs].max()) + 1e-9
            if wcp(margins(Rj, hi, yrs, s["deltas"]), eps) > ETA + 1e-12:
                continue  # cannot satisfy within the 5 K support with any chiller (should not happen)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if wcp(margins(Rj, mid, yrs, s["deltas"]), eps) <= ETA + 1e-12:
                    hi = mid
                else:
                    lo = mid
            C = hi
        c = sel_cost(s, j, C)
        if best is None or c < best[2]:
            best = (j, C, c)
    return best


def saa_select(s, m, yrs):
    best = None
    for j in range(len(s["designs"])):
        req = np.sort(s["R"][j, 0, yrs])[::-1]
        C = float(req[m]) if len(req) > m else 0.0
        C = C if C > 1e-9 else 0.0
        c = sel_cost(s, j, C)
        if best is None or c < best[2]:
            best = (j, C, c)
    return best


def bl1_select(s, v):
    vi = s["bl1v"].index(v)
    best = None
    for j in range(len(s["designs"])):
        C = float(s["NB"][j, vi])
        C = C if C > 1e-9 else 0.0
        c = sel_cost(s, j, C)
        if best is None or c < best[2]:
            best = (j, C, c)
    return best


def test_eval(s, j, C, test):
    dl = list(np.round(s["deltas"], 3))
    fam = {"F0": s["R"][j, dl.index(0.0), test], "F1": s["R"][j, dl.index(0.6), test],
           "F2": s["R"][j, dl.index(1.2), test], "F3": s["R"][j, dl.index(1.8), test], "F4": s["R4"][j, test]}
    return {f: float(np.mean(v > C + 1e-9)) for f, v in fam.items()}, test_cost(s, j, C)


def pooled(data, choice, weighted=True):
    ws = {k: (s["w"] if weighted else 1.0) for k, s in data.items()}
    W = sum(ws.values())
    fam = {f: sum(ws[k] * choice[k]["viol"][f] for k in data) / W for f in FAMILIES}
    cost = sum(ws[k] * choice[k]["cost"] for k in data) / W
    return {"cost": cost, "viol_test": float(np.mean([fam[f] for f in FAMILIES])),
            "viol_warm": float(np.mean([fam[f] for f in FAMILIES[1:]])), "by_family": fam}


def frontier(points):
    """Non-dominated (violation, cost) points sorted by cost ascending (violation strictly decreasing)."""
    pts = sorted(points, key=lambda p: (p[1], p[0]))
    out = []
    for v, c in pts:
        if not out or v < out[-1][0] - 1e-12:
            out.append((v, c))
    return out


def cost_at_violation(points, v):
    f = frontier(points)
    if v >= f[0][0]:
        return f[0][1]
    for (v1, c1), (v2, c2) in zip(f, f[1:]):
        if v2 <= v <= v1:
            return c1 + (v1 - v) / max(v1 - v2, 1e-15) * (c2 - c1)
    return None  # no point reaches that reliability


def violation_at_cost(points, c):
    f = frontier(points)
    if c < f[0][1] - 1e-9:
        return None
    for (v1, c1), (v2, c2) in zip(f, f[1:]):
        if c1 <= c <= c2:
            return v1 - (c - c1) / max(c2 - c1, 1e-15) * (v1 - v2)
    return f[-1][0]


def rules(dr, bl1_pts_test, bl1_pts_warm, bl2_pts_warm):
    c_bl1 = cost_at_violation(bl1_pts_test, dr["viol_test"])
    a = (c_bl1 is not None) and (dr["cost"] <= 0.97 * c_bl1)
    v1 = violation_at_cost(bl1_pts_warm, dr["cost"])
    v2 = violation_at_cost(bl2_pts_warm, dr["cost"])
    b = (dr["viol_warm"] <= ETA + 1e-12) and ((v1 is not None and v1 > ETA) or (v2 is not None and v2 > ETA))
    return {"a": bool(a), "b": bool(b), "bl1_cost_at_dr_violation": c_bl1,
            "dr_cost_saving_vs_bl1": (1 - dr["cost"] / c_bl1) if c_bl1 else None,
            "bl1_viol_warm_at_dr_cost": v1, "bl2_viol_warm_at_dr_cost": v2}


def main():
    data = load()
    years = next(iter(data.values()))["years"]
    train = np.where((years >= 1990) & (years <= 2014))[0]
    test = np.where((years >= 2015) & (years <= 2025))[0]
    folds = [train[i:i + 5] for i in range(0, 25, 5)]
    # ---- radius rule (training only)
    loyo = {}
    for eps in EPS_GRID:
        vals = []
        for f in folds:
            rest = np.setdiff1d(train, f)
            W = sum(s["w"] for s in data.values())
            v = 0.0
            for k, s in data.items():
                j, C, _ = dr_select(s, eps, rest)
                v += s["w"] * float(np.mean(s["R"][j, 0, f] > C + 1e-9))
            vals.append(v / W)
        loyo[eps] = {"fold_violation": vals, "mean": float(np.mean(vals))}
    ok = [e for e in EPS_GRID if loyo[e]["mean"] <= ETA + 1e-12]
    eps_star = min(ok) if ok else max(EPS_GRID)
    # ---- methods on full training, evaluated on test families
    methods = {}
    for eps in EPS_GRID:
        methods[f"DR_eps{eps}"] = {k: dict(zip(("j", "C", "sel"), dr_select(s, eps, train))) for k, s in data.items()}
    for m in SAA_M:
        methods[f"SAA_m{m}"] = {k: dict(zip(("j", "C", "sel"), saa_select(s, m, train))) for k, s in data.items()}
    for v in next(iter(data.values()))["bl1v"]:
        methods[f"BL1_{v}"] = {k: dict(zip(("j", "C", "sel"), bl1_select(s, v))) for k, s in data.items()}
    for name, ch in methods.items():
        for k, s in data.items():
            viol, cost = test_eval(s, ch[k]["j"], ch[k]["C"], test)
            ch[k].update(viol=viol, cost=cost, design=s["designs"][ch[k]["j"]])
    res = {"eps_star": eps_star, "loyo": loyo, "eta": ETA, "methods": {}, "verdict": {}}
    for weighted in (True, False):
        tag = "weighted" if weighted else "unweighted"
        pm = {name: pooled(data, ch, weighted) for name, ch in methods.items()}
        res["methods"][tag] = pm
        bl1_test = [(pm[n]["viol_test"], pm[n]["cost"]) for n in pm if n.startswith("BL1_")]
        bl1_warm = [(pm[n]["viol_warm"], pm[n]["cost"]) for n in pm if n.startswith("BL1_")]
        bl2_warm = [(pm[n]["viol_warm"], pm[n]["cost"]) for n in pm if n.startswith("SAA_")]
        sweep = {eps: rules(pm[f"DR_eps{eps}"], bl1_test, bl1_warm, bl2_warm) for eps in EPS_GRID}
        prim = sweep[eps_star]
        verdict = "PASS" if (prim["a"] or prim["b"]) else (
            "INCONCLUSIVE" if any(v["a"] or v["b"] for v in sweep.values()) else "NEGATIVE")
        res["verdict"][tag] = {"primary": prim, "sweep": {str(k): v for k, v in sweep.items()}, "verdict": verdict}
    res["per_site"] = {name: {k: {kk: vv for kk, vv in v.items() if kk != "sel"} for k, v in ch.items()}
                       for name, ch in methods.items()}
    (OUT / "r2_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("eps* =", eps_star, {e: round(loyo[e]["mean"], 4) for e in EPS_GRID})
    for tag in ("weighted", "unweighted"):
        print(f"--- {tag}: verdict {res['verdict'][tag]['verdict']}; primary {res['verdict'][tag]['primary']}")
        for n, m in res["methods"][tag].items():
            print(f"   {n:12s} cost ${m['cost']:9.0f}/MW-yr  viol_test {m['viol_test']:.3f}  viol_warm {m['viol_warm']:.3f}")


if __name__ == "__main__":
    main()
