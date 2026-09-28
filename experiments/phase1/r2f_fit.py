"""PR-R2F fitting library (ledger PR-R2F, amendments 1-2): sizing rules on a fitting state (families, years), nested calibration with
the declared selection / failure rules and bracketing refinement, locking by outer refit, and test evaluation.

Conventions
- Worlds are models; family-balanced weights p_m = 1 / (F_fit * n_f(m)) over the fitting families.
- Requirement arrays: train R[site][m] (68 designs x 25 raw training years 1990-2014), test R[site][m] (68 x 30 years 1960-1989).
- A capacity C violates year y of world m when R > C + 1e-9. eta = 0.05 unless a secondary budget is run.
- Fitting objective (choose the design per site): CRF * (cap0_j + C * CAPCH) + mean over fitting years of the per-year delta = 0
  operating cost (index 0 without chiller, 1 with). Inner scoring: the same with the omitted block's years. Outer refit and the
  reported K: all 25 years (= the registered objective op_tr). Design ties: lowest design id.
- A method is a function fit(state, site, knob) -> (C_j for every design) plus a knob path; `select_designs` applies the objective.
"""
import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase0"))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from r2_aggregate import CAPCH, CRF, load  # noqa: E402
from r2f_transport import worst_case  # noqa: E402

OUT = ROOT / "experiments/phase1/out/r2f"
TOL = 1e-9
BLOCKS = [np.arange(5 * b, 5 * b + 5) for b in range(5)]  # 1990-94, ..., 2010-14 (indices into the 25 training years)
ALL_YEARS = np.arange(25)


class Inputs:
    def __init__(self, need_test=False):
        data = load()
        self.sites = list(data)
        W = sum(s["w"] for s in data.values())
        self.wn = np.array([data[st]["w"] / W for st in self.sites])
        self.cap0 = {st: np.asarray(data[st]["cap0"], float) for st in self.sites}
        self.op_tr = {st: np.asarray(data[st]["op_tr"], float) for st in self.sites}
        self.nd = len(data[self.sites[0]]["designs"])
        dall = json.loads((OUT / "deltas_2060_all.json").read_text(encoding="utf-8"))
        self.fams = {f: [m for m in ms if m in dall["valid_models"]] for f, ms in dall["families"].items()}
        self.models = [m for ms in self.fams.values() for m in ms]
        self.fam_of = {m: f for f, ms in self.fams.items() for m in ms}
        self.midx = {m: i for i, m in enumerate(self.models)}
        # field metric D_mk = sum_s w~_s (1/12) sum_l |dT_m,s,l - dT_k,s,l| over the 15 R2 sites' cells
        X = np.stack([np.stack([np.array(dall["delta_2060"][m])[:, dall["cells"].index(dall["site_cell"][st])] for st in self.sites])
                      for m in self.models])  # (models, sites, 12)
        self.D = np.einsum("s,mksl->mk", self.wn, np.abs(X[:, None] - X[None, :])) / 12.0
        self.jja = X[:, :, 5:8].mean(axis=2)  # (models, sites)
        self.op_year = {st: np.load(OUT / "opyear" / f"{st}.npz")["op"] for st in self.sites}  # (68, 2, 25)
        self.Rtr = {st: np.stack([np.load(OUT / "train" / f"{st}__{m}.npz")["R"] for m in self.models]) for st in self.sites}
        self.Rte = None
        if need_test:
            self.load_test()

    def load_test(self):
        self.Rte = {st: np.stack([np.load(OUT / "test" / f"{st}__{m}.npz")["R"] for m in self.models]) for st in self.sites}

    def weights(self, fams):
        ms = [m for f in fams for m in self.fams[f]]
        return np.array([self.midx[m] for m in ms]), np.array([1.0 / (len(fams) * len(self.fams[self.fam_of[m]])) for m in ms])


def wquantile_lower(x, w, q):
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, q * cw[-1] - 1e-12), len(x) - 1)])


# ---------------------------------------------------------------- per-design capacities on a fitting state
def caps_saa(A, p, alpha):
    """A: (K, nd, nY) requirements; smallest C per design with sum_k p_k * share_k(R > C) <= alpha."""
    K, nd, nY = A.shape
    out = np.zeros(nd)
    for j in range(nd):
        v = A[:, j, :].ravel()
        wt = np.repeat(p / nY, nY)
        keep = v > TOL
        v, wt = v[keep], wt[keep]
        if wt.sum() <= alpha + 1e-12:
            continue
        o = np.argsort(-v, kind="stable")
        v, wt = v[o], np.cumsum(wt[o])
        # the largest prefix whose weight (values strictly above C) stays <= alpha; C = next distinct value
        t = np.searchsorted(wt, alpha + 1e-12, side="right")  # wt[:t] <= alpha
        # values strictly greater than C must be exactly the prefix; move t back to a distinct-value boundary
        while t > 0 and v[t - 1] == v[t]:
            t -= 1
        out[j] = v[t]
    return out


def caps_rob(A, count):
    """Smallest C per design with at most `count` violating fitting years in every world."""
    s = -np.sort(-A, axis=2)  # descending per world and design
    return np.maximum(s[:, :, count] if count < A.shape[2] else 0.0, 0.0).max(axis=0)


def caps_qnt(A, p, q, eta):
    """Per design: each world's BL2-style need (floor(eta * nY) violating years allowed); C = family-balanced lower q-quantile."""
    nY = A.shape[2]
    k = int(np.floor(eta * nY + 1e-9))
    need = np.maximum(-np.sort(-A, axis=2)[:, :, k], 0.0)  # (K, nd)
    return np.array([wquantile_lower(need[:, j], p, q) for j in range(A.shape[1])])


def caps_drf(A, p, Dfit, eps, eta):
    """Smallest C per design with the exact finite-support Wasserstein worst case <= eta (binary search over requirement values)."""
    K, nd, nY = A.shape
    out = np.zeros(nd)
    for j in range(nd):
        Aj = A[:, j, :]
        def psi(C):
            return worst_case((Aj > C + TOL).mean(axis=1), p, Dfit, eps)
        if psi(0.0) <= eta + 1e-12:
            continue
        cand = np.unique(Aj[Aj > TOL])
        lo, hi = 0, len(cand) - 1  # psi(cand[-1]) = 0 <= eta
        while lo < hi:
            mid = (lo + hi) // 2
            if psi(cand[mid]) <= eta + 1e-12:
                hi = mid
            else:
                lo = mid + 1
        out[j] = cand[lo]
    return out


def select_designs(inp, st, caps, years):
    """Least fitting-objective design for one site given per-design capacities; ties -> lowest design id."""
    opm = inp.op_year[st][:, :, years].mean(axis=2)  # (nd, 2)
    k = CRF * (inp.cap0[st] + caps * CAPCH) + np.where(caps > TOL, opm[:, 1], opm[:, 0])
    j = int(np.argmin(k))
    return j, float(caps[j])


def cost_on(inp, st, j, C, years):
    op = inp.op_year[st][j, 1 if C > TOL else 0, years].mean()
    return CRF * (inp.cap0[st][j] + C * CAPCH) + op


def registered_cost(inp, st, j, C):
    return CRF * (inp.cap0[st][j] + C * CAPCH) + inp.op_tr[st][j, 1 if C > TOL else 0]


# ---------------------------------------------------------------- methods
def fit_state(inp, method, knob, fams_fit, years, eta=0.05):
    """Fitted (j, C) per site for one method, knob and fitting state."""
    idx, p = inp.weights(fams_fit)
    res = []
    for si, st in enumerate(inp.sites):
        A = inp.Rtr[st][idx][:, :, years]
        if method == "B-SAA":
            caps = caps_saa(A, p, knob)
        elif method == "B-ROB":
            caps = caps_rob(A, int(knob))
        elif method == "B-QNT":
            caps = caps_qnt(A, p, knob, eta)
        elif method == "DR-F":
            Dfit = inp.D[np.ix_(idx, idx)]
            eps_max = float(np.max(p @ Dfit))
            caps = caps_drf(A, p, Dfit, knob * eps_max, eta)
        else:
            raise ValueError(method)
        res.append(select_designs(inp, st, caps, years))
    return res


def inner_states(fams, f):
    """Inner splits for outer fold f: (held-out family g, block b) with fitting families = all - {f, g}."""
    return [(g, b) for g in fams if g != f for b in range(5)]


class Calibrator:
    """Caches fits per (method, knob, unordered excluded pair, block) so that splits (f, g, b) and (g, f, b) share them."""

    def __init__(self, inp, eta=0.05):
        self.inp, self.eta = inp, eta
        self.cache = {}

    def fit_excluding(self, method, knob, excl, b):
        key = (method, knob, tuple(sorted(excl)), b)
        if key not in self.cache:
            fams_fit = [f for f in self.inp.fams if f not in excl]
            years = ALL_YEARS if b is None else np.setdiff1d(ALL_YEARS, BLOCKS[b])
            self.cache[key] = fit_state(self.inp, method, knob, fams_fit, years, self.eta)
        return self.cache[key]

    def inner_metrics(self, method, knob, f):
        """Family-balanced mean over (g, b) of the capacity-weighted validation violation and of the scored cost."""
        inp = self.inp
        risks, costs = [], []
        for g, b in inner_states(inp.fams, f):
            fit = self.fit_excluding(method, knob, (f, g), b)
            gi = [inp.midx[m] for m in inp.fams[g]]
            r = c = 0.0
            for si, st in enumerate(inp.sites):
                j, C = fit[si]
                r += inp.wn[si] * float(np.mean(inp.Rtr[st][gi][:, j, BLOCKS[b]] > C + TOL))
                c += inp.wn[si] * cost_on(inp, st, j, C, BLOCKS[b])
            risks.append(r)
            costs.append(c)
        return float(np.mean(risks)), float(np.mean(costs))

    def calibrate(self, method, knobs, f, protect_increasing, refine=True, tol_pct=0.25, max_iter=12):
        """Declared selection: min inner cost among knobs with inner risk <= eta, then min risk, then canonical order (ascending
        knob). Bracketing refinement between the cheapest qualifying knob and its less-protective neighbour until the fitted
        solutions coincide at both ends or the inner-cost gap is below tol_pct."""
        table = {k: self.inner_metrics(method, k, f) for k in knobs}
        certified = True
        if refine and method in ("B-SAA", "B-QNT", "DR-F"):
            for _ in range(max_iter):
                ks = sorted(table)
                q = [k for k in ks if table[k][0] <= self.eta + 1e-12]
                if not q:
                    break
                best = min(q, key=lambda k: (table[k][1], table[k][0], k))
                # less protective neighbour: next knob in the direction of decreasing protection
                i = ks.index(best)
                nb = ks[i - 1] if protect_increasing else (ks[i + 1] if i + 1 < len(ks) else None)
                if protect_increasing and i == 0:
                    nb = None
                if nb is None or table[nb][0] <= self.eta + 1e-12:
                    break
                same = all(self.fit_excluding(method, best, (f, g), b) == self.fit_excluding(method, nb, (f, g), b)
                           for g, b in inner_states(self.inp.fams, f))
                gap = 100 * (table[best][1] - table[nb][1]) / table[best][1]
                if same or gap < tol_pct:
                    break
                mid = 0.5 * (best + nb)
                if mid in table or abs(best - nb) < 1e-6:
                    certified = False
                    break
                table[mid] = self.inner_metrics(method, mid, f)
            else:
                certified = False
        ks = sorted(table)
        qual = [k for k in ks if table[k][0] <= self.eta + 1e-12]
        if qual:
            sel = min(qual, key=lambda k: (table[k][1], table[k][0], k))
            failure = False
        else:
            sel = min(ks, key=lambda k: (table[k][0], table[k][1], k))
            failure = True
        return {"knob": sel, "calibration_failure": failure, "certified": certified,
                "table": {f"{k:.6g}": list(v) for k, v in table.items()}}

    def lock(self, method, knob, f):
        return self.fit_excluding(method, knob, (f,), None)


# ---------------------------------------------------------------- test evaluation and the oracle
def evaluate(inp, designs_by_fold):
    """designs_by_fold: {fold: [(j, C) per site]} -> fold risks / costs, aggregate V and K, per-site and per-world detail."""
    Vf, Kf, detail = {}, {}, {}
    for f, des in designs_by_fold.items():
        wi = [inp.midx[m] for m in inp.fams[f]]
        v = np.array([float(np.mean(inp.Rte[st][wi][:, j, :] > C + TOL)) for (j, C), st in zip(des, inp.sites)])
        per_world = {m: float(sum(inp.wn[si] * np.mean(inp.Rte[st][inp.midx[m]][j, :] > C + TOL)
                                  for si, ((j, C), st) in enumerate(zip(des, inp.sites)))) for m in inp.fams[f]}
        Vf[f] = float(inp.wn @ v)
        Kf[f] = float(sum(inp.wn[si] * registered_cost(inp, st, j, C) for si, ((j, C), st) in enumerate(zip(des, inp.sites))))
        detail[f] = {"site_violation": dict(zip(inp.sites, v.tolist())), "world_risk": per_world}
    return {"V": float(np.mean(list(Vf.values()))), "K": float(np.mean(list(Kf.values()))), "V_f": Vf, "K_f": Kf, "detail": detail}


def oracle(inp, level):
    """Multi-fold pooled oracle on the test (every design x capacity breakpoint incl. C = 0; registered cost; primary weights):
    min mean_f sum_s w~_s K s.t. mean_f sum_s w~_s v <= level. Returns the incumbent and the certified dual bound."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    F = len(inp.fams)
    c, a, grp = [], [], []
    for gi, (f, ms) in enumerate(inp.fams.items()):
        wi = [inp.midx[m] for m in ms]
        for si, st in enumerate(inp.sites):
            R = inp.Rte[st][wi]  # (n_f, nd, 30)
            n = R.shape[0] * R.shape[2]
            best = np.full(n + 1, np.inf)
            for j in range(inp.nd):
                allr = R[:, j, :].ravel()
                for C in np.unique(np.concatenate([[0.0], allr[allr > TOL]])):
                    v = int(np.sum(allr > C + TOL))
                    k = registered_cost(inp, st, j, C)
                    if k < best[v]:
                        best[v] = k
            best = np.minimum.accumulate(best)
            keep = np.flatnonzero(np.isfinite(best))
            # drop dominated counts (same cost as a smaller count)
            keep = [v for v in keep if v == keep[0] or best[v] < best[v - 1] - 1e-12]
            for v in keep:
                grp.append(gi * 100 + si)
                c.append(inp.wn[si] * best[v] / F)
                a.append(inp.wn[si] * v / (n * F))
    grp = np.array(grp)
    ug = np.unique(grp)
    from scipy.sparse import csr_matrix
    rows = np.searchsorted(ug, grp)
    A = csr_matrix((np.ones(len(grp)), (rows, np.arange(len(grp)))), shape=(len(ug), len(grp)))
    r = milp(np.array(c), constraints=[LinearConstraint(A, 1, 1), LinearConstraint(np.array(a)[None, :], -np.inf, level)],
             integrality=np.ones(len(c)), bounds=Bounds(0, 1), options={"mip_rel_gap": 1e-9})
    # no assertion (internal review r2-field-dro R3, F1): report status and whatever certified quantities exist; the gate treats missing or
    # uncertified values as unresolved (proceed)
    out = {"success": bool(r.success), "status": int(r.status), "message": str(r.message), "options": int(len(c)),
           "incumbent": None, "risk": None, "dual_bound": None, "mip_gap": None}
    db = getattr(r, "mip_dual_bound", None)
    if db is not None and np.isfinite(db):
        out["dual_bound"] = float(db)
    if r.x is not None:
        x = np.round(r.x)
        feasible = bool(np.all(np.abs(A @ x - 1) < 1e-9) and float(np.array(a) @ x) <= level + 1e-12)
        if feasible:
            out["incumbent"] = float(np.array(c) @ x)
            out["risk"] = float(np.array(a) @ x)
    gap = getattr(r, "mip_gap", None)
    if gap is not None and np.isfinite(gap):
        out["mip_gap"] = float(gap)
    return out


def unordered_pairs(fams):
    return list(itertools.combinations(list(fams), 2))


# ---------------------------------------------------------------- F6: paired cluster bootstrap (amendment 3, verbatim rule)
TEST_BLOCKS = [np.arange(5 * b, 5 * b + 5) for b in range(6)]  # 1960-64, ..., 1985-89 (indices into the 30 test years)


def bootstrap(inp, designs, dr_key, baseline_keys, compliant_keys, B=2000, seed=20260926, level=0.05):
    """designs: {procedure: {fold: [(j, C) per site]}}. Families (folds) and the six synchronized 5-year test blocks are resampled
    with replacement, identically for every procedure, world and site; within-family averaging is preserved. Returns, for every
    prespecified baseline b, the saving 1 - K_DR / K_b with its one-sided (level / M) percentile lower bound (Bonferroni over the M
    contrasts), the claim flag on the ORIGINAL compliant set (no reselection; adverse replicates kept), and risk intervals.
    Conditional on the fitted procedures and the reference adjustment; approximate (12 clusters)."""
    rng = np.random.default_rng(seed)
    fams = list(inp.fams)
    procs = [dr_key] + list(baseline_keys)
    # per procedure: fold cost K_f and per-fold, per-block capacity-weighted violation shares
    Kf = {pk: np.array([sum(inp.wn[si] * registered_cost(inp, st, *designs[pk][f][si]) for si, st in enumerate(inp.sites))
                        for f in fams]) for pk in procs}
    Vfb = {}
    for pk in procs:
        arr = np.zeros((len(fams), 6))
        for fi, f in enumerate(fams):
            wi = [inp.midx[m] for m in inp.fams[f]]
            for si, st in enumerate(inp.sites):
                j, C = designs[pk][f][si]
                viol = (inp.Rte[st][wi][:, j, :] > C + TOL).mean(axis=0)  # per test year, averaged over the family's worlds
                arr[fi] += inp.wn[si] * np.array([viol[b].mean() for b in TEST_BLOCKS])
        Vfb[pk] = arr
    M = len(baseline_keys)
    sav = {b: np.empty(B) for b in baseline_keys}
    risk = {pk: np.empty(B) for pk in procs}
    for r in range(B):
        fi = rng.integers(0, len(fams), len(fams))
        bi = rng.integers(0, 6, 6)
        K = {pk: Kf[pk][fi].mean() for pk in procs}
        for b in baseline_keys:
            sav[b][r] = 1.0 - K[dr_key] / K[b]
        for pk in procs:
            risk[pk][r] = Vfb[pk][np.ix_(fi, bi)].mean()
    point = {b: float(1.0 - Kf[dr_key].mean() / Kf[b].mean()) for b in baseline_keys}
    lb = {b: float(np.quantile(sav[b], level / M)) for b in baseline_keys}
    claim = bool(compliant_keys) and all(lb[b] >= 0.03 for b in compliant_keys)
    return {"B": B, "seed": seed, "M": M, "point_saving": point, "simultaneous_lower_bound": lb,
            "at_least_3pct_vs_all_compliant": claim, "compliant_set": sorted(compliant_keys),
            "risk_interval_90": {pk: [float(np.quantile(risk[pk], 0.05)), float(np.quantile(risk[pk], 0.95))] for pk in procs}}
