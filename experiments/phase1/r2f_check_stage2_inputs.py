"""PR-R2F: verify every Stage 2 input array exists before locking (shift grid incl. signed atoms; fixed-q B-MON fields)."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/phase1/out/r2f"
R2_GRID = (0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.4, 3.0, 4.0, 5.0)


def main():
    g = json.loads((OUT / "shift_grid.json").read_text(encoding="utf-8"))
    s = json.loads((OUT / "deltas_2060_all.json").read_text(encoding="utf-8"))
    missing = []
    for st, v in g.items():
        need = {round(0.1 * k, 1) for k in range(1, int(round(v["grid_max_K"] * 10)) + 1)}
        ci = s["cells"].index(s["site_cell"][st])
        need |= {float(np.floor(float(np.mean(np.array(s["delta_2060"][m])[5:8, ci])) * 10 + 0.5) / 10) for m in s["valid_models"]}
        for sig in sorted(need):
            if any(abs(sig - x) < 1e-9 for x in R2_GRID):
                continue
            if not (OUT / "shift" / f"{st}__{sig:.1f}.npz").exists():
                missing.append(f"shift/{st}__{sig:.1f}")
    idx = json.loads((OUT / "bmon_index.json").read_text(encoding="utf-8"))
    for key, h in idx.items():
        st = key.split("|")[0]
        if not (OUT / "bmon" / f"{st}__{h}.npz").exists():
            missing.append(f"bmon/{st}__{h}")
    print(len(missing), "missing", missing[:10])
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
