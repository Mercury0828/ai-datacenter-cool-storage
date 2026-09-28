"""PR-R2F Stage 2 (runs only if the Stage 1 gate did not rule out a cost WIN; amendments 2-3).

Procedures (each calibrated by exact nested enumeration on training data and locked by the outer refit, except the fixed-q B-MON):
  Stage 1 (re-used locks at 5 %): B-SAA, B-ROB, B-QNT.
  B-SCAL     constant field = rounded (0.1 K) family-balanced lower q-quantile of the fitting worlds' JJA-mean deltas; BL2-style.
  BL2-shift  raw weather; kappa path sigma = round(kappa * Delta_s(Y) * (2060.5 - mean Y)), kappa in [0, 2]; absolute path 0.1-6.0 K.
  BL3        as BL2-shift with no violating fitting year.
  BL1-shift  ASHRAE conditions from the fitting years (8 variants) + sigma (kappa path, absolute path 0.0-6.0 K); steady state.
  B-MON-qX   fixed q in {0.5, 0.75, 0.9, 1}: per-month quantile field of the outer fold's training models; BL2-style; no calibration.
  DR-F       normalized radius r in [0, 1] (exact path).
Every scalar shift uses the hourly arrays of the 0.1 K grid (R2 delta-grid values read raw from the R2 precompute; others from
out/r2f/shift). Budgets: 5 % (primary), 2.5 % and 10 % (secondary; eta replaced consistently, alpha scaled by alpha/eta).
Verdict (primary, 5 %), branch order: DR-F non-compliant -> FAIL; compliant with no compliant baseline -> reliability result;
else K_DR <= 0.97 K_min -> WIN, K_DR >= 1.03 K_min -> FAIL, otherwise INCONCLUSIVE. Bootstrap per F6.
Writes out/r2f/stage2_locks_<eta>.json (hashed, before test reads) and out/r2f/stage2_summary.json.
Usage: python experiments/phase1/r2f_stage2.py [--eta 0.05 0.025 0.10] [--part lock|evaluate|all]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import theilslopes  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase0"))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import r2f_fit as F  # noqa: E402
import r2f_paths as P  # noqa: E402
from model import components as C  # noqa: E402
from model import params as PR  # noqa: E402
from r2_dro import BL1_VARIANTS, ETA2, design_conditions  # noqa: E402

OUT = F.OUT
KAPPA_MAX = 2.0
H_REF = 2060.5
ABS_TREND = [round(0.1 * k, 1) for k in range(1, 61)]
ABS_BL1 = [round(0.1 * k, 1) for k in range(0, 61)]
BMON_Q = (0.5, 0.75, 0.9, 1.0)
R2_GRID = {0.0: 0, 0.3: 1, 0.6: 2, 0.9: 3, 1.2: 4, 1.5: 5, 1.8: 6, 2.4: 7, 3.0: 8, 4.0: 9, 5.0: 10}
SAA_RATIOS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # registered alpha / eta ratios (0, 0.01, ..., 0.05 at eta = 0.05)


def rnd(x):
    return np.floor(np.asarray(x, float) * 10 + 0.5) / 10


def bl2_count(eta, n):
    return int(np.floor(eta * n + 1e-9))


class Inputs2(F.Inputs):
    def __init__(self):
        super().__init__(need_test=False)
        from r2f_precompute import train_weather
        self._raw = {st: np.load(ROOT / f"experiments/phase0/out/r2/sites/{st}.npz")["R"] for st in self.sites}  # raw, no cummax
        self._shift = {}
        self.bmon_key = json.loads((OUT / "bmon_index.json").read_text(encoding="utf-8"))
        self.trend, self.cond, self.pmean = {}, {}, {}
        for st in self.sites:
            W = train_weather(st)
            yrs = W["years"]
            jja = W["t_db"][:, 151 * 24:243 * 24].mean(axis=1)
            for b in (None, 0, 1, 2, 3, 4):
                keep = np.ones(25, bool) if b is None else ~np.isin(np.arange(25), F.BLOCKS[b])
                sl = max(0.0, float(theilslopes(jja[keep], yrs[keep])[0]))
                self.trend[(st, b)] = sl * (H_REF - float(yrs[keep].mean()))
                self.cond[(st, b)] = design_conditions(W, keep)
                self.pmean[(st, b)] = float(W["p"][keep].mean())
        self.PP = PR.params(ch_etaII=ETA2)
        self.lim = PR.fws_max(self.PP)
        from r2_dro import design_list
        self.designs = design_list(self.PP, self.lim)

    def shift(self, st, sig):
        sig = float(round(float(sig), 1))
        key = (st, sig)
        if key not in self._shift:
            if sig in R2_GRID:
                self._shift[key] = np.asarray(self._raw[st][:, R2_GRID[sig], :25], float)
            else:
                self._shift[key] = np.load(OUT / "shift" / f"{st}__{sig:.1f}.npz")["R"]
        return self._shift[key]

    def bmon(self, st, f, q):
        return np.load(OUT / "bmon" / f"{st}__{self.bmon_key[f'{st}|{f}|{q:g}']}.npz")["R"]


def bl1_need_vec(inp, d, cond, p, sig):
    """Vectorized steady-state trim need of design d at condition + sigma (array); equals r2_dro.bl1_need pointwise."""
    fam, k, ad, th, kct = d
    sig = np.asarray(sig, float)
    pp = np.full(sig.shape, p)
    if fam == "tower":
        _, wb = cond["wet"]
        return C.TowerChain(inp.PP, kct, inp.lim).trim_need(1.0, wb + sig, pp)
    ch = C.DryChain(inp.PP, k, inp.lim)
    db, mcwb = cond["dry"]
    if not ad:
        return ch.trim_need(1.0, db + sig, pp)
    mcdb, wb = cond["wet"]
    t1 = C.adiabatic_air_temp(db + sig, mcwb + sig, inp.PP["eta_ad"])
    t2 = C.adiabatic_air_temp(mcdb + sig, wb + sig, inp.PP["eta_ad"])
    return np.maximum(ch.trim_need(1.0, t1, pp), ch.trim_need(1.0, t2, pp))


def need_at(inp, st, sig, years, count):
    """Per design: the (count+1)-th largest fitting-year requirement at uniform shift sig (0 if not positive)."""
    A = inp.shift(st, sig)[:, years]
    s = -np.sort(-A, axis=1)
    return np.maximum(s[:, count] if count < A.shape[1] else 0.0, 0.0)


# ---------------------------------------------------------------- continuous paths: per-design step functions on a state
def steps_scal(inp, si, idx, p, years, eta):
    st = inp.sites[si]
    atoms = rnd(inp.jja[idx, si])  # signed: a negative JJA delta is a (rounded) cooling shift (internal review stage2 R1, F1)
    o = np.argsort(inp.jja[idx, si], kind="stable")
    W = np.cumsum(p[o]) / p.sum()
    cnt = bl2_count(eta, len(years))
    caps = np.stack([need_at(inp, st, a, years, cnt) for a in atoms[o]], axis=1)  # (nd, K)
    thr = W[:-1] + 1e-12
    return [(thr, caps[j], "left") for j in range(inp.nd)]


def kappa_thresholds(Dh):
    if Dh <= 0:
        return np.array([]), np.array([0.0])
    kmax = int(np.floor(KAPPA_MAX * Dh * 10 + 0.5))
    thr = np.array([(k - 0.5) * 0.1 / Dh for k in range(1, kmax + 1)])
    sig = np.array([round(0.1 * k, 1) for k in range(0, kmax + 1)])
    return thr, sig


def steps_trend_k(inp, si, years, b, eta, robust):
    st = inp.sites[si]
    thr, sig = kappa_thresholds(inp.trend[(st, b)])
    cnt = 0 if robust else bl2_count(eta, len(years))
    caps = np.stack([need_at(inp, st, s, years, cnt) for s in sig], axis=1)
    return [(thr, caps[j], "right") for j in range(inp.nd)]


def steps_bl1_k(inp, si, b, variant):
    st = inp.sites[si]
    thr, sig = kappa_thresholds(inp.trend[(st, b)])
    cond, p = inp.cond[(st, b)][variant], inp.pmean[(st, b)]
    caps = np.stack([np.maximum(bl1_need_vec(inp, tuple(d), cond, p, sig), 0.0) for d in inp.designs])
    return [(thr, caps[j], "right") for j in range(inp.nd)]


class Stage2Cache(P.PathCache):
    """PathCache with the Stage 2 continuous paths; `spec` = (kind, *args); kind in DR-F, B-SAA, B-QNT, B-SCAL, BL2-k, BL3-k, BL1-k."""

    def path(self, spec, excl, b, si):
        key = (spec, tuple(sorted(excl)), b, si, self.eta)
        if key not in self.paths:
            inp = self.inp
            fams_fit = [f for f in inp.fams if f not in excl]
            idx, p = inp.weights(fams_fit)
            years = np.setdiff1d(np.arange(25), F.BLOCKS[b]) if b is not None else np.arange(25)
            st = inp.sites[si]
            kind = spec[0]
            lo, hi = domain(spec, self.eta)
            if kind in ("DR-F", "B-SAA", "B-QNT"):
                steps = P.state_steps(inp, kind, idx, p, years, self.eta, st)
            elif kind == "B-SCAL":
                steps = steps_scal(inp, si, idx, p, years, self.eta)
            elif kind in ("BL2-k", "BL3-k"):
                steps = steps_trend_k(inp, si, years, b, self.eta, kind == "BL3-k")
            elif kind == "BL1-k":
                steps = steps_bl1_k(inp, si, b, spec[1])
            else:
                raise ValueError(spec)
            self.paths[key] = P.site_path(steps, P.make_fitcost(inp, st, years), lo, hi)
        return self.paths[key]

    def fold_table(self, spec, f):
        inp = self.inp
        splits = [(g, b) for g in inp.fams if g != f for b in range(5)]
        lo, hi = domain(spec, self.eta)
        paths = {(g, b, si): self.path(spec, (f, g), b, si) for g, b in splits for si in range(len(inp.sites))}
        T = np.unique(np.concatenate([[lo, hi]] + [pth[0] for pth in paths.values()]))
        x = np.unique(np.concatenate([T, 0.5 * (T[:-1] + T[1:])]))
        risk, cost = np.zeros(len(x)), np.zeros(len(x))
        for (g, b, si), pth in paths.items():
            st = inp.sites[si]
            gi = [inp.midx[m] for m in inp.fams[g]]
            sols = P.path_solutions(pth)
            vr = np.array([float(np.mean(inp.Rtr[st][gi][:, j, F.BLOCKS[b]] > C_ + F.TOL)) for j, C_ in sols])
            vc = np.array([F.cost_on(inp, st, j, C_, F.BLOCKS[b]) for j, C_ in sols])
            q = P.query(pth, x)
            risk += inp.wn[si] * vr[q]
            cost += inp.wn[si] * vc[q]
        return x, risk / len(splits), cost / len(splits)

    def lock_path(self, spec, f, x):
        """Outer refit at knob x for a continuous path."""
        out = []
        for si, st in enumerate(self.inp.sites):
            pth = self.path(spec, (f,), None, si)
            sols = P.path_solutions(pth)
            out.append(sols[int(P.query(pth, np.array([x]))[0])])
        return out


def domain(spec, eta):
    kind = spec[0]
    if kind == "B-SAA":
        return 0.0, eta
    if kind in ("BL2-k", "BL3-k", "BL1-k"):
        return 0.0, KAPPA_MAX
    return 0.0, 1.0


# ---------------------------------------------------------------- discrete paths
def fit_discrete(inp, spec, knob, years, b, eta):
    """(j, C) per site for a discrete knob on the fitting years (families do not enter these procedures)."""
    kind = spec[0]
    out = []
    for si, st in enumerate(inp.sites):
        if kind in ("BL2-a", "BL3-a"):
            caps = need_at(inp, st, knob, years, 0 if kind == "BL3-a" else bl2_count(eta, len(years)))
        elif kind == "BL1-a":
            caps = bl1_grid(inp, st, b, spec[1])[:, ABS_BL1.index(round(float(knob), 1))]
        else:
            raise ValueError(spec)
        out.append(F.select_designs(inp, st, caps, years))
    return out


_BL1_MEMO = {}


def bl1_grid(inp, st, b, variant):
    """Steady-state needs (designs x ABS_BL1) at the fitting years' ASHRAE condition; memoized per (site, block, variant)."""
    key = (st, b, variant)
    if key not in _BL1_MEMO:
        cond, p = inp.cond[(st, b)][variant], inp.pmean[(st, b)]
        _BL1_MEMO[key] = np.stack([np.maximum(bl1_need_vec(inp, tuple(d), cond, p, np.array(ABS_BL1)), 0.0) for d in inp.designs])
    return _BL1_MEMO[key]


def discrete_table(inp, spec, knobs, f, eta, cache):
    rows = []
    for k in knobs:
        rs, cs = [], []
        for g in inp.fams:
            if g == f:
                continue
            gi = [inp.midx[m] for m in inp.fams[g]]
            for b in range(5):
                key = (spec, k, b, eta)
                if key not in cache:
                    cache[key] = fit_discrete(inp, spec, k, np.setdiff1d(np.arange(25), F.BLOCKS[b]), b, eta)
                fit = cache[key]
                r = c = 0.0
                for si, st in enumerate(inp.sites):
                    j, C_ = fit[si]
                    r += inp.wn[si] * float(np.mean(inp.Rtr[st][gi][:, j, F.BLOCKS[b]] > C_ + F.TOL))
                    c += inp.wn[si] * F.cost_on(inp, st, j, C_, F.BLOCKS[b])
                rs.append(r)
                cs.append(c)
        rows.append((k, float(np.mean(rs)), float(np.mean(cs))))
    return rows


# ---------------------------------------------------------------- procedures, selection, locking
PROCEDURES = {
    "B-SCAL": [("B-SCAL",)],
    "BL2-shift": [("BL2-k",), ("BL2-a",)],
    "BL3": [("BL3-k",), ("BL3-a",)],
    "BL1-shift": [(k, v) for v in BL1_VARIANTS for k in ("BL1-k", "BL1-a")],
    "DR-F": [("DR-F",)],
}


def calibrate_procedure(inp, cache2, dcache, name, f, eta):
    """Every candidate of every path of the procedure; declared selection and fallback; canonical order = ascending numerical
    knob, then path tag, then variant id."""
    cands = []  # (cost, risk, knob, path tag, variant, spec)
    for spec in PROCEDURES[name]:
        kind = spec[0]
        variant = spec[1] if len(spec) > 1 else ""
        if kind.endswith("-a"):
            knobs = ABS_BL1 if kind == "BL1-a" else ABS_TREND
            for k, r, c in discrete_table(inp, spec, knobs, f, eta, dcache):
                cands.append((c, r, float(k), kind, variant, spec))
        else:
            x, r, c = cache2.fold_table(spec, f)
            for xi, ri, ci in zip(x, r, c):
                cands.append((float(ci), float(ri), float(xi), kind, variant, spec))
    qual = [c for c in cands if c[1] <= eta + 1e-12]
    if qual:
        best = min(qual, key=lambda c: (c[0], c[1], c[2], c[3], c[4]))
        failure = False
    else:
        best = min(cands, key=lambda c: (c[1], c[0], c[2], c[3], c[4]))
        failure = True
    spec, knob = best[5], best[2]
    if spec[0].endswith("-a"):
        designs = fit_discrete(inp, spec, knob, np.arange(25), None, eta)
    else:
        designs = cache2.lock_path(spec, f, knob)
    return {"path": spec[0], "variant": best[4], "knob": knob, "inner_risk": best[1], "inner_cost": best[0],
            "calibration_failure": failure, "n_candidates": len(cands), "designs": [[int(j), float(c)] for j, c in designs]}


def lock_bmon(inp, f, q, eta):
    out = []
    for st in inp.sites:
        A = inp.bmon(st, f, q)
        s = -np.sort(-A, axis=1)
        cnt = bl2_count(eta, 25)
        caps = np.maximum(s[:, cnt], 0.0)
        out.append(F.select_designs(inp, st, caps, np.arange(25)))
    return [[int(j), float(c)] for j, c in out]


def stage1_locks_at(inp, eta):
    """Stage 1 procedures at budget eta (re-used from stage1_locks.json at 5 %; recomputed at secondary budgets)."""
    if abs(eta - 0.05) < 1e-12:
        rec = json.loads((OUT / "stage1_locks.json").read_text(encoding="utf-8"))
        return {m: {f: v["designs"] for f, v in per.items()} for m, per in rec["locks"].items()}
    pc = P.PathCache(inp, eta)
    cal = F.Calibrator(inp, eta)
    out, meta = {}, {}
    for m in ("B-SAA", "B-QNT"):
        out[m] = {}
        for f in inp.fams:
            sel = pc.select(m, f)  # the B-SAA domain is [0, eta] inside PathCache
            out[m][f] = [[int(j), float(c)] for j, c in P.lock(inp, m, f, sel["knob"], eta)]
            meta.setdefault(m, {})[f] = sel
    out["B-ROB"] = {}
    counts = rob_counts(eta)
    for f in inp.fams:
        table = {c: cal.inner_metrics("B-ROB", c, f) for c in counts}
        qual = [c for c in counts if table[c][0] <= eta + 1e-12]
        k = min(qual, key=lambda c: (table[c][1], table[c][0], c)) if qual else min(counts, key=lambda c: (table[c][0], table[c][1], c))
        out["B-ROB"][f] = [[int(j), float(C_)] for j, C_ in cal.lock("B-ROB", k, f)]
        meta.setdefault("B-ROB", {})[f] = {"knob": int(k), "calibration_failure": not qual}
    out["_meta"] = meta
    return out


def rob_counts(eta):
    """Every attainable B-ROB count for alpha in [0, eta] (internal review stage2 R1, F2): c = 0..floor(eta * 20); knob c stands for the
    smallest alpha with inner count c (alpha = c / 20, ascending-knob ties), whose 25-year outer count floor(1.25 c) equals c."""
    cs = list(range(0, int(np.floor(eta * 20 + 1e-9)) + 1))
    assert all(int(np.floor(c / 20 * 25 + 1e-9)) == c for c in cs)
    return cs


def lock_budget(inp, eta):
    t0 = time.time()
    cache2 = Stage2Cache(inp, eta)
    dcache = {}
    locks = {"stage1": stage1_locks_at(inp, eta), "calibrated": {}, "fixed": {}}
    for name in PROCEDURES:
        locks["calibrated"][name] = {}
        for f in inp.fams:
            locks["calibrated"][name][f] = calibrate_procedure(inp, cache2, dcache, name, f, eta)
            print(eta, name, f, {k: v for k, v in locks["calibrated"][name][f].items() if k != "designs"}, flush=True)
    for q in BMON_Q:
        locks["fixed"][f"B-MON-q{q:g}"] = {f: lock_bmon(inp, f, q, eta) for f in inp.fams}
    blob = json.dumps(locks, sort_keys=True)
    rec = {"eta": eta, "locks": locks, "sha256": hashlib.sha256(blob.encode()).hexdigest(), "seconds": round(time.time() - t0, 1),
           "note": "written before any test array is read"}
    (OUT / f"stage2_locks_{eta:g}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def designs_of(rec):
    L = rec["locks"]
    out = {m: {f: [tuple(d) for d in v] for f, v in per.items()} for m, per in L["stage1"].items() if m != "_meta"}
    for name, per in L["calibrated"].items():
        out[name] = {f: [tuple(d) for d in v["designs"]] for f, v in per.items()}
    for name, per in L["fixed"].items():
        out[name] = {f: [tuple(d) for d in v] for f, v in per.items()}
    return out


def verdict(res, eta):
    dr = res["DR-F"]
    base = {m: v for m, v in res.items() if m != "DR-F"}
    comp = {m: v for m, v in base.items() if v["V"] <= eta + 1e-12}
    if dr["V"] > eta + 1e-12:
        v = "FAIL (DR-F not compliant)"
    elif not comp:
        v = "reliability result (DR-F compliant; no compliant baseline)"
    else:
        kmin = min(c["K"] for c in comp.values())
        if dr["K"] <= 0.97 * kmin + 1e-9:
            v = "WIN"
        elif dr["K"] >= 1.03 * kmin - 1e-9:
            v = "FAIL"
        else:
            v = "INCONCLUSIVE"
    trend = {m: v_ for m, v_ in comp.items() if m in ("BL2-shift", "BL3", "BL1-shift")}
    info = None
    if dr["V"] <= eta + 1e-12 and trend:
        kt = min(c["K"] for c in trend.values())
        if dr["K"] <= 0.97 * kt + 1e-9 and v != "WIN":
            info = "information-modelling result: DR-F >= 3 % cheaper than the compliant trend-only baselines, not a DRO win"
    return {"verdict": v, "compliant_baselines": sorted(comp), "cheapest_compliant": None if not comp else
            min(comp, key=lambda m: comp[m]["K"]), "information_modelling_note": info}


def evaluate_budget(inp, rec):
    blob = json.dumps(rec["locks"], sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == rec["sha256"], "locks changed after writing"
    if inp.Rte is None:
        inp.load_test()
    eta = rec["eta"]
    des = designs_of(rec)
    res = {}
    for name, d in des.items():
        ev = F.evaluate(inp, d)
        res[name] = {"V": ev["V"], "K": ev["K"], "V_f": ev["V_f"], "K_f": ev["K_f"], "detail": ev["detail"]}
    out = {"eta": eta, "procedures": res, **verdict(res, eta)}
    baselines = [m for m in des if m != "DR-F"]
    out["bootstrap"] = F.bootstrap(inp, des, "DR-F", baselines, out["compliant_baselines"], B=2000, seed=20260926)
    print(eta, out["verdict"], {m: (round(v["V"], 4), round(v["K"], 1)) for m, v in res.items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, nargs="+", default=[0.05, 0.025, 0.10])
    ap.add_argument("--part", default="all", choices=("lock", "evaluate", "all"))
    a = ap.parse_args()
    inp = Inputs2()
    recs = {}
    if a.part in ("lock", "all"):
        for eta in a.eta:
            recs[eta] = lock_budget(inp, eta)
    else:
        for eta in a.eta:
            recs[eta] = json.loads((OUT / f"stage2_locks_{eta:g}.json").read_text(encoding="utf-8"))
    if a.part in ("evaluate", "all"):
        summ = {f"{eta:g}": evaluate_budget(inp, recs[eta]) for eta in a.eta}
        (OUT / "stage2_summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
