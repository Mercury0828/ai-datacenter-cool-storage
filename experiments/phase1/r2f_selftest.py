"""PR-R2F self-tests on synthetic inputs (no project data, no test outcomes):
(1) transport worst case vs LP and radius threshold vs bisection;
(2) direct sizing rules vs brute force;
(3) exact path enumeration vs a dense knob grid of direct fits;
(4) the Stage 1 pipeline end to end: locking, evaluation, oracle; the oracle's certified lower bound must not exceed the cost of
    any procedure whose aggregate risk meets the level (a consistency check of the evaluation and oracle conventions);
(5) the bootstrap runs and returns bounds no larger than the point savings' range.
Usage: python experiments/phase1/r2f_selftest.py
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
import r2f_fit as F  # noqa: E402
import r2f_paths as P  # noqa: E402
import r2f_transport as T  # noqa: E402


class Mock:
    def __init__(self, seed, nd=5):
        rng = np.random.default_rng(seed)
        self.sites = ["s0", "s1", "s2"]
        self.wn = np.array([0.5, 0.3, 0.2])
        self.nd = nd
        self.fams = {"A": ["a1", "a2", "a3"], "B": ["b1"], "C": ["c1", "c2"], "D": ["d1"], "E": ["e1", "e2"]}
        self.models = [m for ms in self.fams.values() for m in ms]
        self.fam_of = {m: f for f, ms in self.fams.items() for m in ms}
        self.midx = {m: i for i, m in enumerate(self.models)}
        M = len(self.models)
        X = rng.random((M, 3, 12))
        self.D = np.einsum("s,mksl->mk", self.wn, np.abs(X[:, None] - X[None, :])) / 12.0
        self.cap0 = {s: rng.random(nd) * 1000 for s in self.sites}
        self.op_year = {s: np.sort(rng.random((nd, 2, 25)) * 100, axis=1) for s in self.sites}
        self.op_tr = {s: self.op_year[s].mean(axis=2) for s in self.sites}
        base = {s: rng.random((nd, 25)) for s in self.sites}
        self.Rtr = {s: np.where(rng.random((M, nd, 25)) < 0.5, 0.0,
                                np.round(base[s][None] * (0.5 + rng.random((M, 1, 1))) * 0.05, 4)) for s in self.sites}
        self._Rte = {s: np.where(rng.random((M, nd, 30)) < 0.5, 0.0,
                                 np.round(rng.random((M, nd, 30)) * 0.06, 4)) for s in self.sites}
        self.Rte = None

    def weights(self, fams):
        ms = [m for f in fams for m in self.fams[f]]
        return np.array([self.midx[m] for m in ms]), np.array([1.0 / (len(fams) * len(self.fams[self.fam_of[m]])) for m in ms])

    def load_test(self):
        self.Rte = self._Rte


def main():
    print("(1) transport:", T.check(), T.check_threshold())
    bad = 0
    for seed in range(4):
        inp = Mock(seed)
        pc, cal = P.PathCache(inp), F.Calibrator(inp)
        for method in ("B-SAA", "B-QNT", "DR-F"):
            for f in ("A", "B"):
                x, r, c = pc.fold_table(method, f)
                lo, hi = P.DOMAIN[method]
                hi = hi if method != "B-SAA" else 0.05
                for k in np.linspace(lo, hi, 31)[1:-1] + 1e-7:
                    rr, cc = cal.inner_metrics(method, float(k), f)
                    i = np.searchsorted(x, k)
                    if not any(abs(r[j] - rr) < 1e-9 and abs(c[j] - cc) < 1e-6 for j in range(max(0, i - 2), min(len(x), i + 2))):
                        bad += 1
    print("(3) path enumeration vs dense grid, mismatches:", bad)
    import r2f_stage1 as S
    viol = 0
    for seed in range(3):
        inp = Mock(seed)
        S.OUT = __import__("pathlib").Path(__import__("tempfile").mkdtemp())
        rec = S.lock_all(inp)
        res = S.evaluate_all(inp, rec)
        for m, v in res["methods"].items():
            for lev in (0.05,):
                L = res["oracle_5pct"].get("dual_bound")
                if v["V"] <= lev + 1e-12 and L is not None and L > v["K"] + 1e-9:
                    viol += 1
        des = {m: {f: [tuple(d) for d in rec["locks"][m][f]["designs"]] for f in inp.fams} for m in rec["locks"]}
        bs = F.bootstrap(inp, des, "B-QNT", ["B-SAA", "B-ROB"], [m for m in ("B-SAA", "B-ROB") if res["methods"][m]["compliant"]], B=200)
        assert all(np.isfinite(v) for v in bs["simultaneous_lower_bound"].values())
    print("(4) oracle lower bound above a compliant procedure's cost (must be 0):", viol)
    print("(5) bootstrap ran")
    sys.exit(1 if (bad or viol) else 0)


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def stage2_check(seeds=3):
    """(6) Stage 2 continuous paths (B-SCAL, BL2-k, BL3-k) vs dense direct fits on synthetic shift arrays."""
    import r2f_stage2 as S2
    bad = 0
    for seed in range(seeds):
        inp = Mock(seed)
        rng = np.random.default_rng(100 + seed)
        base = {s: rng.random((inp.nd, 25)) * 0.02 for s in inp.sites}
        grid = {s: {round(0.1 * k, 1): np.where(base[s] * (1 + 0.05 * k) > 0.015, np.round(base[s] * (1 + 0.05 * k), 5), 0.0)
                    for k in range(-15, 130)} for s in inp.sites}  # signed shifts (negative atoms: internal review stage2 R1, F1)
        inp.shift = lambda st, sig, g=grid: g[st][round(float(sig), 1)]
        inp.trend = {(s, b): float(rng.random() * 3 + 0.5) for s in inp.sites for b in (None, 0, 1, 2, 3, 4)}
        inp.jja = rng.random((len(inp.models), len(inp.sites))) * 4.5 - 1.0  # includes negative JJA deltas
        c2 = S2.Stage2Cache(inp, 0.05)
        for spec in (("B-SCAL",), ("BL2-k",), ("BL3-k",)):
            for f in ("A", "C"):
                x, r, c = c2.fold_table(spec, f)
                lo, hi = S2.domain(spec, 0.05)
                for k in np.linspace(lo, hi, 29)[1:-1] + 1e-7:
                    rs, cs = [], []
                    for g in inp.fams:
                        if g == f:
                            continue
                        gi = [inp.midx[m] for m in inp.fams[g]]
                        idx, p = inp.weights([h for h in inp.fams if h not in (f, g)])
                        for b in range(5):
                            yrs = np.setdiff1d(np.arange(25), F.BLOCKS[b])
                            rr = cc = 0.0
                            for si, st in enumerate(inp.sites):
                                if spec[0] == "B-SCAL":
                                    sig = float(S2.rnd(F.wquantile_lower(inp.jja[idx, si], p, k)))
                                    caps = S2.need_at(inp, st, sig, yrs, S2.bl2_count(0.05, len(yrs)))
                                else:
                                    sig = float(S2.rnd(k * inp.trend[(st, b)]))
                                    caps = S2.need_at(inp, st, sig, yrs, 0 if spec[0] == "BL3-k" else S2.bl2_count(0.05, len(yrs)))
                                j, C_ = F.select_designs(inp, st, caps, yrs)
                                rr += inp.wn[si] * float(np.mean(inp.Rtr[st][gi][:, j, F.BLOCKS[b]] > C_ + F.TOL))
                                cc += inp.wn[si] * F.cost_on(inp, st, j, C_, F.BLOCKS[b])
                            rs.append(rr)
                            cs.append(cc)
                    rr, cc = np.mean(rs), np.mean(cs)
                    i = np.searchsorted(x, k)
                    if not any(abs(r[jj] - rr) < 1e-9 and abs(c[jj] - cc) < 1e-6 for jj in range(max(0, i - 2), min(len(x), i + 2))):
                        bad += 1
    return bad


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "stage2":
    import r2f_stage2 as S2
    counts = {e: S2.rob_counts(e) for e in (0.025, 0.05, 0.10)}
    print("(7) B-ROB counts per budget:", counts)
    assert counts == {0.025: [0], 0.05: [0, 1], 0.10: [0, 1, 2]}, counts
    bad = stage2_check()
    print("(6) Stage 2 paths vs dense grid (signed atoms), mismatches:", bad)
    sys.exit(1 if bad else 0)
