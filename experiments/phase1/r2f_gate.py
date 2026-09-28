"""LEGACY DIAGNOSTIC ONLY (superseded by PR-R2F amendments 2-3): this hindsight-frontier headroom cannot terminate the experiment and
uses delta-grid interpolation; the terminating gate is the calibrated-comparator gate in r2f_stage1.py.
Original description: PR-R2F gate (ledger PR-R2F (8); PR-R2S rule): on the new test (held-out family worlds x 1960-1989 years), the multi-fold pooled
oracle vs the cheapest hindsight-selected executable point of the baselines computable from the delta-grid arrays:
  trend-only BL2-shift / BL3 / BL1-shift (8 variants) retargeted to 2060.5 (Delta_s^2060 = Theil-Sen JJA slope 1990-2014 x 58.5 yr),
  and B-SCAL per outer fold (family-balanced q-quantile of the training models' JJA-mean 2060 delta at the site's cell).
Aggregation: fold risk V_f = sum_s w~_s v_{s,f}, v_{s,f} = share of (world in family f, test year) with R > C; V = mean_f V_f;
cost K = mean_f sum_s w~_s K_s (registered objective). Oracle: per fold and site, every design x capacity breakpoint; exact MILP.
Gate passes at 5 % if 1 - K_oracle / K_comparator >= 3 %. The oracle selects nothing.
Writes experiments/phase1/out/r2f/gate_summary.json.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import theilslopes

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase0"))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402
from r2_aggregate import load  # noqa: E402
from r2_dro import bl1_need  # noqa: E402
from r2r_analysis import KAPPA, cost, select, shifted  # noqa: E402
from r2r_b_analysis import BL1_ABS, SHIFT_ABS  # noqa: E402
from r2r_c_analysis import shifted_req  # noqa: E402

OUT = ROOT / "experiments/phase1/out/r2f"
LEVELS = (0.025, 0.05, 0.10)
H2060 = 2060.5 - 2002.0
Q_SCAL = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def jja_slope(site):
    W = E.load_weather(ROOT / f"data/weather/{site}.npz")
    tr = (W["years"] >= 1990) & (W["years"] <= 2014)
    jja = W["t_db"][:, 151 * 24:243 * 24].mean(axis=1)
    return float(theilslopes(jja[tr], W["years"][tr])[0])


def wquantile(x, w, q):
    """Inverse-CDF (lower) quantile of a discrete distribution with weights w; q = 1 gives the maximum."""
    o = np.argsort(x)
    cw = np.cumsum(np.asarray(w)[o])
    return float(np.asarray(x)[o][min(np.searchsorted(cw, q * cw[-1] - 1e-12), len(x) - 1)])


def family_weights(models, fams):
    fam_of = {m: f for f, ms in fams.items() for m in ms}
    present = sorted({fam_of[m] for m in models})
    return np.array([1.0 / (len(present) * sum(fam_of[k] == fam_of[m] for k in models)) for m in models])


def trend_points(data, site, s, tr, exact_keys=frozenset()):
    """{knob: (j, C, cost, clipped)} for the trend-only baselines at one site. Shifts above 5 K use the 5 K grid row (a lower bound on
    requirements and cost, flagged clipped) unless the knob is in exact_keys, where they are recomputed exactly (cached)."""
    D = max(0.0, jja_slope(site)) * H2060
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    cond = json.loads(str(np.load(ROOT / f"experiments/phase0/out/r2/sites/{site}.npz")["cond"]))
    p_mean = float(np.mean(E.load_weather(ROOT / f"data/weather/{site}.npz")["p"]))
    pts = {}
    knobs = [(m, f"k{k:g}", k * D) for m in ("BL2-shift", "BL3") for k in KAPPA] + \
            [(m, f"a{a:g}", a) for m in ("BL2-shift", "BL3") for a in SHIFT_ABS] + \
            [("BL1-shift", f"k{k:g}", k * D) for k in (0.0, 1.0)] + [("BL1-shift", f"a{a:g}", a) for a in BL1_ABS]
    for m, lab, sig in knobs:
        key = f"{m}|{lab}"
        ex = key in exact_keys
        clipped = sig > s["deltas"][-1] + 1e-12 and not ex
        if m == "BL2-shift":
            pts[key] = (select(s, lambda j: float(np.sort(shifted_req(s, site, j, sig, ex)[tr])[::-1][1])) or (None,) * 3) + (clipped,)
        elif m == "BL3":
            pts[key] = (select(s, lambda j: float(shifted_req(s, site, j, sig, ex)[tr].max())) or (None,) * 3) + (clipped,)
        else:
            for v in s["bl1v"]:
                c = {"dry": (cond[v]["dry"][0] + sig, cond[v]["dry"][1] + sig), "wet": (cond[v]["wet"][0] + sig, cond[v]["wet"][1] + sig)}
                nb = [bl1_need(P, tuple(d), c, p_mean, lim) for d in s["designs"]]
                pts[f"{m}|{lab}|{v}"] = (select(s, lambda j: max(nb[j], 0.0)) or (None,) * 3) + (False,)
    return D, pts


def scal_points(s, site, tr, cell, dall, models, fams, exact_q=frozenset()):
    """{(fold, q): (j, C, cost, clipped)} for B-SCAL at one site: shift = family-balanced q-quantile over the fold's training models;
    above 5 K the 5 K row bounds the point from below (clipped) unless q is in exact_q."""
    jja = {m: float(np.mean(np.array(dall["delta_2060"][m])[5:8, dall["cells"].index(cell)])) for m in models}
    out = {}
    for f, fm in fams.items():
        trm = [m for m in models if m not in fm]
        w = family_weights(trm, {g: ms for g, ms in fams.items() if g != f})
        x = np.array([jja[m] for m in trm])
        for q in Q_SCAL:
            sig = wquantile(x, w, q)
            ex = q in exact_q
            sel = select(s, lambda j: float(np.sort(shifted_req(s, site, j, sig, ex)[tr])[::-1][1])) or (None,) * 3
            out[(f, q)] = sel + (sig > s["deltas"][-1] + 1e-12 and not ex,)
    return out


def fold_viol(Rt, fams, j, C):
    """{fold: share of (world in family, test year) with requirement > C} for one site and design."""
    return {f: float(np.mean([np.mean(Rt[m][j] > C + 1e-9) for m in ms])) for f, ms in fams.items()}


def oracle(data, Rtest, fams, wn, level):
    """Multi-fold pooled oracle: per fold and site, cheapest option for each violating (world, year) count; one option per group;
    mean over folds of the capacity-weighted violation share <= level."""
    F = len(fams)
    c, a, groups = [], [], []
    for gi, (f, ms) in enumerate(fams.items()):
        for site, s in data.items():
            n = len(ms) * 30
            best = np.full(n + 1, np.inf)
            for j in range(len(s["designs"])):
                allr = np.concatenate([Rtest[site][m][j] for m in ms])
                for C in np.unique(np.concatenate([[0.0], allr[allr > 1e-9]])):
                    v = int(np.sum(allr > C + 1e-9))
                    k = cost(s, j, C)
                    if k < best[v]:
                        best[v] = k
            best = np.minimum.accumulate(best)
            for v in range(n + 1):
                if np.isfinite(best[v]):
                    groups.append(gi * 1000 + list(data).index(site))
                    c.append(wn[site] * best[v] / F)
                    a.append(wn[site] * v / (n * F))
    groups = np.array(groups)
    ug = np.unique(groups)
    A = (groups[None, :] == ug[:, None]).astype(float)
    r = milp(np.array(c), constraints=[LinearConstraint(A, 1, 1), LinearConstraint(np.array(a)[None, :], -np.inf, level)],
             integrality=np.ones(len(c)), bounds=Bounds(0, 1), options={"mip_rel_gap": 1e-9})
    assert r.success, r.message
    x = np.round(r.x)
    return {"cost": float(np.array(c) @ x), "risk": float(np.array(a) @ x), "mip_gap": float(getattr(r, "mip_gap", np.nan)),
            "dual_bound": float(getattr(r, "mip_dual_bound", np.nan)), "status": int(r.status)}


def main():
    data = load()
    dall = json.loads((OUT / "deltas_2060_all.json").read_text(encoding="utf-8"))
    fams = {f: [m for m in ms if m in dall["valid_models"]] for f, ms in dall["families"].items()}
    models = [m for ms in fams.values() for m in ms]
    assert len(models) == 33 and len(fams) == 12
    W = sum(s["w"] for s in data.values())
    wn = {st: s["w"] / W for st, s in data.items()}
    Rtest = {st: {m: np.load(OUT / "test" / f"{st}__{m}.npz")["R"] for m in models} for st in data}
    def build(exact_trend, exact_q):
        pts, clip = {}, set()
        tD, pst, pss = {}, {}, {}
        for st, s in data.items():
            tr = np.where((s["years"] >= 1990) & (s["years"] <= 2014))[0]
            tD[st], pst[st] = trend_points(data, st, s, tr, exact_trend)
            pss[st] = scal_points(s, st, tr, dall["site_cell"][st], dall, models, fams, exact_q)
        for key in pst[next(iter(data))]:
            if any(pst[st].get(key) is None or pst[st][key][0] is None for st in data):
                continue
            Vf = {f: sum(wn[st] * fold_viol(Rtest[st], {f: fams[f]}, *pst[st][key][:2])[f] for st in data) for f in fams}
            pts[key] = {"cost": sum(wn[st] * pst[st][key][2] for st in data), "V_f": Vf, "V": float(np.mean(list(Vf.values())))}
            if any(pst[st][key][3] for st in data):
                clip.add(key)
        for q in Q_SCAL:
            Vf, Kf = {}, {}
            for f in fams:
                Vf[f] = sum(wn[st] * fold_viol(Rtest[st], {f: fams[f]}, *pss[st][(f, q)][:2])[f] for st in data)
                Kf[f] = sum(wn[st] * pss[st][(f, q)][2] for st in data)
            key = f"B-SCAL|q{q:g}"
            pts[key] = {"cost": float(np.mean(list(Kf.values()))), "V_f": Vf, "V": float(np.mean(list(Vf.values())))}
            if any(pss[st][(f, q)][3] for st in data for f in fams):
                clip.add(key)
        return tD, pts, clip

    trend_D, points, clipped = build(frozenset(), frozenset())
    # a clipped point's cost is a lower bound (cost is nondecreasing in the shift); recompute exactly only those that could be the
    # cheapest compliant point at some level (every clipped point if no exact point is compliant there)
    need = set()
    for lev in LEVELS:
        exact_ok = [v["cost"] for k, v in points.items() if k not in clipped and v["V"] <= lev + 1e-12]
        kstar = min(exact_ok) if exact_ok else np.inf
        need |= {k for k in clipped if points[k]["cost"] < kstar}
    print("clipped points", len(clipped), "recomputed exactly", sorted(need), flush=True)
    if need:
        trend_D, points, clipped = build(frozenset(k for k in need if not k.startswith("B-SCAL")),
                                         frozenset(float(k.split("|q")[1]) for k in need if k.startswith("B-SCAL")))
    for k in clipped:
        points[k]["clipped_lower_bound"] = True
    res = {"families": fams, "delta_s_2060_K": trend_D, "levels": {}}
    for lev in LEVELS:
        ok = {k: v for k, v in points.items() if v["V"] <= lev + 1e-12 and not v.get("clipped_lower_bound")}
        best = min(ok.items(), key=lambda kv: kv[1]["cost"]) if ok else None
        orc = oracle(data, Rtest, fams, wn, lev)
        rec = {"oracle": orc, "comparator": None if best is None else {"knob": best[0], **best[1]}}
        if best is not None:
            rec["headroom_pct"] = 100 * (1 - orc["cost"] / best[1]["cost"])
            rec["gate_pass_3pct"] = rec["headroom_pct"] >= 3.0
        res["levels"][f"{lev:g}"] = rec
        print(lev, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k != "comparator"},
              None if best is None else (best[0], round(best[1]["cost"], 1), round(best[1]["V"], 4)), flush=True)
    res["points"] = points
    (OUT / "gate_summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
