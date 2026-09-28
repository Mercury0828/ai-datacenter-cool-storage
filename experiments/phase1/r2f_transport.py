"""PR-R2F exact worst case over the finite-support type-1 Wasserstein set (ledger PR-R2F (5)).

    Psi(eps) = max_{Gamma >= 0} sum_{m,k} Gamma_mk * ell_k   s.t.  sum_k Gamma_mk = p_m,  sum_{m,k} D_mk Gamma_mk <= eps,

with D_mm = 0. This is the LP relaxation of a multiple-choice knapsack: for each centre atom m the non-dominated moves are the upper
concave majorant of {(D_mk, ell_k)} starting at (0, ell_m); the LP optimum takes the majorant segments of all m in decreasing
order of slope (gain per kelvin of transport) until the budget is spent, the last one fractionally. `check()` verifies it against
scipy's LP on random instances.
"""
import numpy as np
from numba import njit


@njit(cache=False)
def worst_case(ell, p, D, eps):
    M = p.shape[0]
    K = ell.shape[0]
    base = 0.0
    # collect majorant segments: (slope, mass-weighted cost, mass-weighted gain)
    seg_slope = np.empty(M * K)
    seg_cost = np.empty(M * K)
    seg_gain = np.empty(M * K)
    ns = 0
    xs = np.empty(K)
    ys = np.empty(K)
    for m in range(M):
        l0 = ell[m]
        base += p[m] * l0
        # candidate targets that improve on staying: ell_k > l0; keep the best gain for each cost (sorted by cost)
        order = np.argsort(D[m])
        n = 0
        for t in range(K):
            k = order[t]
            if ell[k] > l0 + 1e-15 and D[m, k] > 0.0:
                xs[n] = D[m, k]
                ys[n] = ell[k] - l0
                n += 1
        if n == 0:
            continue
        # upper concave majorant from (0, 0): repeatedly take the point with the steepest slope from the current point,
        # ties resolved to the farthest point
        cx, cy = 0.0, 0.0
        start = 0
        while start < n:
            best = -1
            bs = -1.0
            for t in range(start, n):
                if xs[t] > cx and ys[t] > cy:
                    s = (ys[t] - cy) / (xs[t] - cx)
                    if s > bs + 1e-15 or (abs(s - bs) <= 1e-15 and best >= 0 and xs[t] > xs[best]):
                        bs = s
                        best = t
            if best < 0:
                break
            seg_slope[ns] = bs
            seg_cost[ns] = p[m] * (xs[best] - cx)
            seg_gain[ns] = p[m] * (ys[best] - cy)
            ns += 1
            cx, cy = xs[best], ys[best]
            start = best + 1
    if ns == 0 or eps <= 0.0:
        return base
    order = np.argsort(-seg_slope[:ns])
    budget = eps
    gain = 0.0
    for t in range(ns):
        i = order[t]
        if seg_cost[i] <= budget:
            budget -= seg_cost[i]
            gain += seg_gain[i]
        else:
            gain += seg_gain[i] * budget / seg_cost[i]
            break
    return base + gain


@njit(cache=False)
def _segments(ell, p, D):
    """Majorant segments of all atoms (slope, mass-weighted cost, mass-weighted gain) and the base value (as in worst_case)."""
    M = p.shape[0]
    K = ell.shape[0]
    base = 0.0
    seg_slope = np.empty(M * K)
    seg_cost = np.empty(M * K)
    seg_gain = np.empty(M * K)
    ns = 0
    xs = np.empty(K)
    ys = np.empty(K)
    for m in range(M):
        l0 = ell[m]
        base += p[m] * l0
        order = np.argsort(D[m])
        n = 0
        for t in range(K):
            k = order[t]
            if ell[k] > l0 + 1e-15 and D[m, k] > 0.0:
                xs[n] = D[m, k]
                ys[n] = ell[k] - l0
                n += 1
        if n == 0:
            continue
        cx, cy = 0.0, 0.0
        start = 0
        while start < n:
            best = -1
            bs = -1.0
            for t in range(start, n):
                if xs[t] > cx and ys[t] > cy:
                    s = (ys[t] - cy) / (xs[t] - cx)
                    if s > bs + 1e-15 or (abs(s - bs) <= 1e-15 and best >= 0 and xs[t] > xs[best]):
                        bs = s
                        best = t
            if best < 0:
                break
            seg_slope[ns] = bs
            seg_cost[ns] = p[m] * (xs[best] - cx)
            seg_gain[ns] = p[m] * (ys[best] - cy)
            ns += 1
            cx, cy = xs[best], ys[best]
            start = best + 1
    return base, seg_slope[:ns], seg_cost[:ns], seg_gain[:ns]


@njit(cache=False)
def radius_threshold(ell, p, D, eta):
    """sup{eps >= 0 : worst_case(ell, p, D, eps) <= eta}: -1 if infeasible at eps = 0, +inf if never exceeded (exact, from the
    piecewise-linear concave value function)."""
    base, sl, co, ga = _segments(ell, p, D)
    if base > eta + 1e-12:
        return -1.0
    need = eta + 1e-12 - base
    order = np.argsort(-sl)
    spent = 0.0
    got = 0.0
    for t in range(order.shape[0]):
        i = order[t]
        if got + ga[i] > need:
            return spent + co[i] * (need - got) / ga[i]
        got += ga[i]
        spent += co[i]
    return np.inf


def check(n=300, seed=0):
    """Compare worst_case with scipy.optimize.linprog on random instances; returns the largest absolute difference."""
    from scipy.optimize import linprog
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        M = int(rng.integers(2, 9))
        ell = rng.choice([0.0, 0.04, 0.08, 0.12, 0.2, 0.5, 1.0], size=M) * rng.random() if rng.random() < 0.3 else rng.random(M) * 0.3
        x = rng.random((M, 3))
        D = np.abs(x[:, None, :] - x[None, :, :]).sum(axis=2)
        p = rng.random(M)
        p /= p.sum()
        eps = float(rng.random() * D.max() * 0.6)
        c = -np.tile(ell, M)  # Gamma flattened row-major (m, k)
        A_eq = np.kron(np.eye(M), np.ones((1, M)))
        r = linprog(c, A_ub=D.ravel()[None, :], b_ub=[eps], A_eq=A_eq, b_eq=p, bounds=(0, None), method="highs")
        assert r.success
        worst = max(worst, abs(-r.fun - worst_case(ell.astype(float), p, D, eps)))
    return worst


def check_threshold(n=300, seed=1):
    """radius_threshold vs bisection on worst_case; returns the largest mismatch in eps (finite cases)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        M = int(rng.integers(2, 9))
        ell = rng.random(M) * 0.2
        x = rng.random((M, 3))
        D = np.abs(x[:, None, :] - x[None, :, :]).sum(axis=2)
        p = rng.random(M)
        p /= p.sum()
        eta = 0.05
        r = radius_threshold(ell, p, D, eta)
        if r < 0:
            assert worst_case(ell, p, D, 0.0) > eta
            continue
        if not np.isfinite(r):
            assert worst_case(ell, p, D, 1e6) <= eta + 1e-12
            continue
        lo, hi = 0.0, D.max() * 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if worst_case(ell, p, D, mid) <= eta + 1e-12:
                lo = mid
            else:
                hi = mid
        worst = max(worst, abs(lo - r))
    return worst


if __name__ == "__main__":
    print("max |greedy - LP| over random instances:", check())
    print("max |threshold - bisection| over random instances:", check_threshold())
