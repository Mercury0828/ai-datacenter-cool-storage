"""PR-R2F amendment 3 (F4): exact enumeration of every distinct inner solution along a method's knob path.

For a fitting state (families, years) and a site, each design's fitted capacity is a step function of the knob with explicit
thresholds; the site's selected solution (design id, capacity) can change only at those thresholds. `site_path` evaluates the
solution at every threshold and at the midpoint of every gap and keeps a compact representation (T, at, after):
    query(x) = at[k] if x == T[k] else after[k], with k = the last index with T[k] <= x (T[0] = domain start).
`fold_path` merges the (split, site) paths of an outer fold: the inner risk and cost are evaluated at every change point and every
gap midpoint of the union, so every distinct combination of inner solutions is scored; `select` applies the declared rules.
"""
import numpy as np

from r2f_fit import BLOCKS, CAPCH, CRF, TOL, cost_on
from r2f_transport import radius_threshold

DOMAIN = {"B-SAA": (0.0, 0.05), "B-QNT": (0.0, 1.0), "DR-F": (0.0, 1.0)}
PROTECT_INCREASING = {"B-SAA": False, "B-QNT": True, "DR-F": True}


# ---------------------------------------------------------------- per-design step functions: (thr, caps, side)
# C(x) = caps[np.searchsorted(thr, x, side)] ; thr ascending ; len(caps) = len(thr) + 1
def steps_saa(Aj, p):
    nY = Aj.shape[1]
    v = Aj.ravel()
    w = np.repeat(p / nY, nY)
    keep = v > TOL
    v, w = v[keep], w[keep]
    if v.size == 0:
        return np.array([]), np.array([0.0]), "right"
    vals, inv = np.unique(v, return_inverse=True)
    agg = np.bincount(inv, weights=w)
    vd, wd = vals[::-1], agg[::-1]
    U = np.cumsum(wd)
    # alpha in [U_{t-1}, U_t) -> vd[t]; alpha >= U_last -> 0 ; the fit uses alpha + 1e-12 (caps_saa)
    return U - 1e-12, np.concatenate([vd, [0.0]]), "right"


def steps_qnt(Aj, p, eta):
    nY = Aj.shape[1]
    k = int(np.floor(eta * nY + 1e-9))
    need = np.maximum(-np.sort(-Aj, axis=1)[:, k], 0.0)
    o = np.argsort(need, kind="stable")
    W = np.cumsum(p[o])
    W = W / W[-1]
    # q in (W_{i-1}, W_i] -> need_(i); the fit uses q - 1e-12 against cumsum (wquantile_lower): threshold shifted accordingly
    return W[:-1] + 1e-12, need[o], "left"


def steps_drf(Aj, p, D, eps_max, eta):
    cand = np.concatenate([[0.0], np.unique(Aj[Aj > TOL])])
    def thr(i):
        return radius_threshold((Aj > cand[i] + TOL).mean(axis=1), p, D, eta)
    # first candidate feasible at eps = 0 and first feasible at eps_max (feasibility is monotone in the candidate)
    def first(eps):
        lo, hi = 0, len(cand) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if thr(mid) >= eps - 1e-15:
                hi = mid
            else:
                lo = mid + 1
        return lo
    i0, i1 = first(0.0), first(eps_max)
    if i1 <= i0:
        return np.array([]), np.array([cand[i0]]), "left"
    t = np.array([thr(i) for i in range(i0, i1)])
    t = np.maximum.accumulate(t) / eps_max  # radius units; capacity cand[i0 + i] on (t_{i-1}, t_i]
    return t, cand[i0:i1 + 1], "left"


def eval_steps(st, x):
    thr, caps, side = st
    if thr.size == 0:
        return np.full(len(x), caps[0])
    return caps[np.minimum(np.searchsorted(thr, x, side=side), len(caps) - 1)]


# ---------------------------------------------------------------- one (state, site): compact path
def site_path(steps, fitcost, lo, hi):
    """steps: list over designs of step functions; fitcost(j, C array) -> fitting objective array. Returns (T, at, after) with
    solutions as (j, C) tuples."""
    T_all = np.concatenate([s[0] for s in steps]) if steps else np.array([])
    pts = np.unique(np.concatenate([[lo, hi], T_all[(T_all > lo) & (T_all < hi)]]))
    mids = 0.5 * (pts[:-1] + pts[1:])
    def solve(x):
        caps = np.stack([eval_steps(s, x) for s in steps])  # (nd, P)
        cost = np.stack([fitcost(j, caps[j]) for j in range(len(steps))])
        j = np.argmin(cost, axis=0)  # ties -> lowest design id
        return list(zip(j.tolist(), caps[j, np.arange(len(x))].tolist()))
    sp, sm = solve(pts), (solve(mids) if len(mids) else [])
    T, at, after = [pts[0]], [sp[0]], [sm[0] if len(sm) else sp[0]]
    for i in range(1, len(pts)):
        a = sp[i]
        af = sm[i] if i < len(sm) else sp[i]
        if a != after[-1] or af != after[-1]:
            T.append(pts[i])
            at.append(a)
            after.append(af)
    return np.array(T), at, after


def query(path, x):
    """Indices into the path's solution list for query points x: 2k for at[k], 2k+1 for after[k]."""
    T = path[0]
    k = np.searchsorted(T, x, side="right") - 1
    k = np.maximum(k, 0)
    exact = np.isclose(T[k], x, rtol=0, atol=0)
    return 2 * k + np.where(exact, 0, 1)


def path_solutions(path):
    T, at, after = path
    out = []
    for a, b in zip(at, after):
        out += [a, b]
    return out


# ---------------------------------------------------------------- the method's steps on a fitting state
def state_steps(inp, method, idx, p, years, eta, site):
    A = inp.Rtr[site][idx][:, :, years]  # (K, nd, nY)
    if method == "B-SAA":
        return [steps_saa(A[:, j, :], p) for j in range(inp.nd)]
    if method == "B-QNT":
        return [steps_qnt(A[:, j, :], p, eta) for j in range(inp.nd)]
    if method == "DR-F":
        Dfit = inp.D[np.ix_(idx, idx)]
        eps_max = float(np.max(p @ Dfit))
        return [steps_drf(A[:, j, :], p, Dfit, eps_max, eta) for j in range(inp.nd)]
    raise ValueError(method)


def make_fitcost(inp, site, years):
    opm = inp.op_year[site][:, :, years].mean(axis=2)
    cap0 = inp.cap0[site]
    def fitcost(j, C):
        return CRF * (cap0[j] + C * CAPCH) + np.where(C > TOL, opm[j, 1], opm[j, 0])
    return fitcost


# ---------------------------------------------------------------- fold-level merge and selection
class PathCache:
    """Compact paths per (method, excluded pair, block, site) and their scored solutions; shared by folds f and g."""

    def __init__(self, inp, eta=0.05):
        self.inp, self.eta = inp, eta
        self.paths = {}

    def path(self, method, excl, b, si):
        key = (method, tuple(sorted(excl)), b, si)
        if key not in self.paths:
            inp = self.inp
            fams_fit = [f for f in inp.fams if f not in excl]
            idx, p = inp.weights(fams_fit)
            years = np.setdiff1d(np.arange(25), BLOCKS[b]) if b is not None else np.arange(25)
            st = inp.sites[si]
            lo, hi = DOMAIN[method]
            hi = hi if method != "B-SAA" else self.eta
            self.paths[key] = site_path(state_steps(inp, method, idx, p, years, self.eta, st), make_fitcost(inp, st, years), lo, hi)
        return self.paths[key]

    def fold_table(self, method, f):
        """Exact inner risk and cost at every change point and gap midpoint of the merged paths of fold f."""
        inp = self.inp
        splits = [(g, b) for g in inp.fams if g != f for b in range(5)]
        lo, hi = DOMAIN[method]
        hi = hi if method != "B-SAA" else self.eta
        paths = {(g, b, si): self.path(method, (f, g), b, si) for g, b in splits for si in range(len(inp.sites))}
        T = np.unique(np.concatenate([[lo, hi]] + [pth[0] for pth in paths.values()]))
        x = np.unique(np.concatenate([T, 0.5 * (T[:-1] + T[1:])]))
        risk = np.zeros(len(x))
        cost = np.zeros(len(x))
        for (g, b, si), pth in paths.items():
            st = inp.sites[si]
            gi = [inp.midx[m] for m in inp.fams[g]]
            sols = path_solutions(pth)
            vr = np.array([float(np.mean(inp.Rtr[st][gi][:, j, BLOCKS[b]] > C + TOL)) for j, C in sols])
            vc = np.array([cost_on(inp, st, j, C, BLOCKS[b]) for j, C in sols])
            q = query(pth, x)
            risk += inp.wn[si] * vr[q]
            cost += inp.wn[si] * vc[q]
        n = len(splits)
        return x, risk / n, cost / n

    def select(self, method, f):
        """Declared rule: min inner cost among knobs with inner risk <= eta, then min risk, then ascending knob; otherwise the
        calibration-failure fallback (min risk, then min cost, then ascending knob)."""
        x, r, c = self.fold_table(method, f)
        q = r <= self.eta + 1e-12
        if q.any():
            i = np.lexsort((x, r, np.where(q, c, np.inf)))[0]
            return {"knob": float(x[i]), "calibration_failure": False, "inner_risk": float(r[i]), "inner_cost": float(c[i]),
                    "n_points": int(len(x))}
        i = np.lexsort((x, c, r))[0]
        return {"knob": float(x[i]), "calibration_failure": True, "inner_risk": float(r[i]), "inner_cost": float(c[i]),
                "n_points": int(len(x))}


def lock(inp, method, f, x, eta=0.05):
    """Outer refit at the selected knob x: families other than f, all 25 training years; (j, C) per site."""
    fams_fit = [g for g in inp.fams if g != f]
    idx, p = inp.weights(fams_fit)
    years = np.arange(25)
    out = []
    for st in inp.sites:
        steps = state_steps(inp, method, idx, p, years, eta, st)
        fc = make_fitcost(inp, st, years)
        caps = np.array([eval_steps(s, np.array([x]))[0] for s in steps])
        cost = np.array([fc(j, np.array([caps[j]]))[0] for j in range(len(steps))])
        j = int(np.argmin(cost))
        out.append((j, float(caps[j])))
    return out
