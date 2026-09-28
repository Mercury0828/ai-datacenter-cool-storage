"""PR-R2F secondary diagnostic (declared in PR-R2F (7): "retrospective test frontiers over each knob (executable designs only; no
interpolation decides anything)"). For every procedure and every distinct value of its knob path, the outer-refit designs of all 12
folds are evaluated on the test; the cheapest point with aggregate risk <= level is reported per procedure. This uses the test
outcomes to choose knobs (hindsight) and therefore cannot change the primary verdict; it shows what each formulation could reach
with a perfect knob choice.
Writes experiments/phase1/out/r2f/frontier_summary.json. Usage: python experiments/phase1/r2f_frontier.py [--eta 0.05]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import r2f_fit as F  # noqa: E402
import r2f_paths as P  # noqa: E402
import r2f_stage2 as S2  # noqa: E402


def site_test(inp, f, si, j, C):
    st = inp.sites[si]
    wi = [inp.midx[m] for m in inp.fams[f]]
    return float(np.mean(inp.Rte[st][wi][:, j, :] > C + F.TOL)), F.registered_cost(inp, st, j, C)


def continuous_frontier(inp, cache, spec, level):
    fams = list(inp.fams)
    paths = {(f, si): cache.path(spec, (f,), None, si) for f in fams for si in range(len(inp.sites))}
    lo, hi = S2.domain(spec, cache.eta)
    T = np.unique(np.concatenate([[lo, hi]] + [p[0] for p in paths.values()]))
    x = np.unique(np.concatenate([T, 0.5 * (T[:-1] + T[1:])]))
    V = np.zeros(len(x))
    K = np.zeros(len(x))
    for (f, si), pth in paths.items():
        sols = P.path_solutions(pth)
        vt = np.array([site_test(inp, f, si, j, C)[0] for j, C in sols])
        kc = np.array([site_test(inp, f, si, j, C)[1] for j, C in sols])
        q = P.query(pth, x)
        V += inp.wn[si] * vt[q] / len(fams)
        K += inp.wn[si] * kc[q] / len(fams)
    return x, V, K


def discrete_frontier(inp, spec, knobs, eta):
    fams = list(inp.fams)
    rows = []
    for k in knobs:
        des = S2.fit_discrete(inp, spec, k, np.arange(25), None, eta)  # fold-independent (no family input)
        V = K = 0.0
        for f in fams:
            for si, (j, C) in enumerate(des):
                v, c = site_test(inp, f, si, j, C)
                V += inp.wn[si] * v / len(fams)
                K += inp.wn[si] * c / len(fams)
        rows.append((k, V, K))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=0.05)
    a = ap.parse_args()
    eta = a.eta
    inp = S2.Inputs2()
    inp.load_test()
    cache = S2.Stage2Cache(inp, eta)
    res = {}
    for name, specs in {"DR-F": [("DR-F",)], "B-SAA": [("B-SAA",)], "B-QNT": [("B-QNT",)], "B-SCAL": [("B-SCAL",)],
                        **{n: S2.PROCEDURES[n] for n in ("BL2-shift", "BL3", "BL1-shift")}}.items():
        best = None
        npts = 0
        for spec in specs:
            if spec[0].endswith("-a"):
                pts = discrete_frontier(inp, spec, S2.ABS_BL1 if spec[0] == "BL1-a" else S2.ABS_TREND, eta)
            else:
                x, V, K = continuous_frontier(inp, cache, spec, eta)
                pts = list(zip(x.tolist(), V.tolist(), K.tolist()))
            npts += len(pts)
            for k, V, K in pts:
                if V <= eta + 1e-12 and (best is None or K < best["K"]):
                    best = {"path": spec[0], "variant": spec[1] if len(spec) > 1 else "", "knob": k, "V": V, "K": K}
        res[name] = {"best_compliant_point": best, "points": npts}
        print(name, best, flush=True)
    base = {m: v["best_compliant_point"]["K"] for m, v in res.items() if m != "DR-F" and v["best_compliant_point"]}
    dr = res["DR-F"]["best_compliant_point"]
    if dr and base:
        res["DR-F_hindsight_saving_vs_best_baseline_hindsight_pct"] = 100 * (1 - dr["K"] / min(base.values()))
    (F.OUT / f"frontier_summary_{eta:g}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
