"""PR-R2F Stage 1 (amendments 2-3): calibrate and lock B-SAA, B-ROB and B-QNT on training data only (exact nested calibration),
write the locked designs with a SHA-256 before any test array is read, then evaluate them on the test (held-out family worlds x
1960-1989), solve the multi-fold pooled oracle with a certified dual bound and apply the terminating gate:
  - compliant b (aggregate V <= 0.05) with oracle lower bound L > 0.97 K_b  -> "cost WIN ruled out" (stop);
  - oracle incumbent <= 0.97 K_b* for the cheapest compliant b*           -> headroom established (Stage 2);
  - otherwise unresolved, or undefined if no baseline complies            -> Stage 2 (does not stop the reliability comparison).
Writes experiments/phase1/out/r2f/stage1_locks.json and stage1_summary.json.
Usage: python experiments/phase1/r2f_stage1.py [--part lock|evaluate|all]
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

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import r2f_fit as F  # noqa: E402
import r2f_paths as P  # noqa: E402

OUT = F.OUT
ETA = 0.05
STAGE1 = ("B-SAA", "B-ROB", "B-QNT")


def lock_all(inp):
    pc = P.PathCache(inp, ETA)
    cal = F.Calibrator(inp, ETA)
    locks = {}
    for method in STAGE1:
        locks[method] = {}
        for f in inp.fams:
            t0 = time.time()
            if method == "B-ROB":
                counts = list(range(0, int(np.floor(ETA * 20 + 1e-9)) + 1))  # {0, 1} at 5 % (as run)
                table = {c: cal.inner_metrics("B-ROB", c, f) for c in counts}
                qual = [c for c in counts if table[c][0] <= ETA + 1e-12]
                if qual:
                    k = min(qual, key=lambda c: (table[c][1], table[c][0], c))
                    sel = {"knob": k, "calibration_failure": False}
                else:
                    k = min(counts, key=lambda c: (table[c][0], table[c][1], c))
                    sel = {"knob": k, "calibration_failure": True}
                sel.update(inner_risk=table[k][0], inner_cost=table[k][1], table={str(c): list(v) for c, v in table.items()})
                designs = cal.lock("B-ROB", k, f)
            else:
                sel = pc.select(method, f)
                designs = P.lock(inp, method, f, sel["knob"], ETA)
            locks[method][f] = {**sel, "designs": [[int(j), float(C)] for j, C in designs], "seconds": round(time.time() - t0, 1)}
            print(method, f, {k: v for k, v in sel.items() if k != "table"}, flush=True)
    blob = json.dumps(locks, sort_keys=True)
    rec = {"locks": locks, "sha256": hashlib.sha256(blob.encode()).hexdigest(), "eta": ETA,
           "note": "written before any test array is read"}
    (OUT / "stage1_locks.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def evaluate_all(inp, rec):
    blob = json.dumps(rec["locks"], sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == rec["sha256"], "locks changed after writing"
    inp.load_test()
    res = {"methods": {}}
    for method, per in rec["locks"].items():
        ev = F.evaluate(inp, {f: [tuple(d) for d in v["designs"]] for f, v in per.items()})
        res["methods"][method] = {"V": ev["V"], "K": ev["K"], "V_f": ev["V_f"], "K_f": ev["K_f"], "compliant": ev["V"] <= ETA + 1e-12,
                                  "calibration_failures": [f for f, v in per.items() if v["calibration_failure"]],
                                  "detail": ev["detail"]}
        print(method, "V", round(ev["V"], 4), "K", round(ev["K"], 1), flush=True)
    comp = {m: v for m, v in res["methods"].items() if v["compliant"]}
    gate = {"compliant_baselines": sorted(comp)}
    if not comp:
        # undefined gate: proceeds independently of the oracle (which is still reported when it solves)
        gate["decision"] = "undefined (no compliant Stage 1 baseline): proceed to Stage 2"
        try:
            res["oracle_5pct"] = F.oracle(inp, ETA)
        except Exception as e:  # reporting only
            res["oracle_5pct"] = {"error": repr(e)}
    else:
        try:
            orc = F.oracle(inp, ETA)
        except Exception as e:
            orc = {"error": repr(e), "dual_bound": None, "incumbent": None}
        res["oracle_5pct"] = orc
        L, inc = orc.get("dual_bound"), orc.get("incumbent")
        gate.update(L=L, incumbent=inc)
        bstar = min(comp, key=lambda m: comp[m]["K"])
        gate["cheapest_compliant"] = {"method": bstar, "K": comp[bstar]["K"]}
        ruled = {m: v["K"] for m, v in comp.items() if L is not None and L > 0.97 * v["K"] + 1e-9}
        if inc is not None:
            gate["headroom_vs_cheapest_pct"] = 100 * (1 - inc / comp[bstar]["K"])
        if L is not None:
            gate["lower_bound_headroom_pct"] = 100 * (1 - L / comp[bstar]["K"])
        if ruled:
            gate["decision"] = f"cost WIN ruled out (certified L > 0.97 K_b for {sorted(ruled)}): stop"
        elif inc is not None and inc <= 0.97 * comp[bstar]["K"] + 1e-9:
            gate["decision"] = "headroom established: proceed to Stage 2"
        else:
            gate["decision"] = "unresolved (no certificate either way): proceed to Stage 2"
    res["gate"] = gate
    print(json.dumps(gate, indent=1), flush=True)
    (OUT / "stage1_summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=("lock", "evaluate", "all"))
    a = ap.parse_args()
    inp = F.Inputs(need_test=False)
    if a.part in ("lock", "all"):
        rec = lock_all(inp)
    else:
        rec = json.loads((OUT / "stage1_locks.json").read_text(encoding="utf-8"))
    if a.part in ("evaluate", "all"):
        evaluate_all(inp, rec)


if __name__ == "__main__":
    main()
