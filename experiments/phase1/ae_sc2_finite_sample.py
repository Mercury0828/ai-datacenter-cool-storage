"""SC-2 (MANIFEST.md, Applied Energy analyses): finite-sample calibrated stationary baseline for the drift test.

Reads the frozen per-cell records experiments/phase1/out/oos_b/<cell>.json (a.E_hist, a.E_scen) and oos_c/<cell>.json
(drift-aware cohorts), weights from data/sites/hub_sites.csv. No simulation.
Writes experiments/phase1/out/ae/sc2_finite_sample.json.
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OOSB = ROOT / "experiments/phase1/out/oos_b"
OOSC = ROOT / "experiments/phase1/out/oos_c"
OUT = ROOT / "experiments/phase1/out/ae/sc2_finite_sample.json"
TOL = 1e-9
TR = slice(0, 25)
TESTS = {"obs": ("E_hist", np.arange(25, 36)), "scen": ("E_scen", np.arange(36))}


def kth_largest(x, k):
    return float(np.sort(np.asarray(x, float))[-k])


def main():
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    recb = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in OOSB.glob("cell_*.json")}
    recc = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in OOSC.glob("cell_*.json")}
    sites = sites[sites.cell_id.isin(recb)]
    res = {"reference": {"second_of_25": 2 / 26, "largest_of_25": 1 / 26}, "n_sites": int(len(sites))}

    def pooled(rule, test, cohort):
        key, idx = TESTS[test]
        num = den = 0.0
        numu = denu = 0
        gw = gwh = 0.0
        n = 0
        for s in sites.itertuples():
            a = recb[s.cell_id]["a"]
            eh = np.array(a["E_hist"], float)
            S = kth_largest(eh[TR], 1 if rule == "largest" else 2)
            if cohort == "own":
                if S > 8 + TOL:
                    continue
            else:  # drift-aware retained cohort of the same test
                c = recc[s.cell_id]["a"][test]
                if c["h_drift"] > 8 + TOL:
                    continue
            e = np.array(a[key], float)[idx]
            v = e > S + TOL
            num += s.B_full_MW * v.sum()
            den += s.B_full_MW * len(idx)
            numu += int(v.sum())
            denu += len(idx)
            gw += s.B_full_MW / 1e3
            gwh += s.B_full_MW / 1e3 * S
            n += 1
        return {"pooled_weighted": num / den, "pooled_unweighted": numu / denu, "sites": n, "GW": gw, "stored_heat_GWh": gwh}

    for rule in ("second", "largest"):
        res[rule] = {f"{t}_{c}": pooled(rule, t, c) for t in TESTS for c in ("own", "drift_cohort")}
    # drift-aware reference values from the frozen oos_c records (same pooling)
    drift = {}
    for t in TESTS:
        num = den = 0.0
        gw = gwh = 0.0
        n = 0
        for s in sites.itertuples():
            c = recc[s.cell_id]["a"][t]
            if c["h_drift"] > 8 + TOL:
                continue
            v = np.array(c["viol"])
            num += s.B_full_MW * v.sum()
            den += s.B_full_MW * len(v)
            gw += s.B_full_MW / 1e3
            gwh += s.B_full_MW / 1e3 * c["h_drift"]
            n += 1
        drift[t] = {"pooled_weighted": num / den, "sites": n, "GW": gw, "stored_heat_GWh": gwh}
    res["drift_aware"] = drift
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
