"""Second pool for a package-2 tag that is already running: same run_item as ff_menu.py (identical outputs; atomic writes;
each item is skipped if its output already exists), but over the pending items in REVERSE order, so the two pools meet in
the middle instead of computing the same items. Records its own PID file (run_rev.pid) for the watchdog.
Usage: python experiments/phase1/ff_menu_reverse.py --weather-dir DIR --tag TAG [--workers 3]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/phase1"))
from ff_menu import CLIMATES, COMMIT, run_item  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--weather-dir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out_dir = ROOT / "experiments/phase1/out/ff" / a.tag
    (out_dir / "run_rev.pid").write_text(str(os.getpid()), encoding="utf-8")
    cells = pd.read_csv(ROOT / "data/sites/hub_cells.csv").site_id.tolist()
    items = [(c, cl, dT, a.weather_dir, str(out_dir)) for c in cells for cl, dT in CLIMATES.items()
             if not (out_dir / f"{c}__{cl}.json").exists()][::-1]
    print(f"{len(items)} items pending (reverse order); commit {COMMIT}; tag {a.tag}", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_item, items), 1):
            print(f"[{time.time() - t0:6.0f}s] {n}/{len(items)} {r}", flush=True)
            if all((out_dir / f"{c}__{cl}.json").exists() for c in cells for cl in CLIMATES):
                print("all items present; stopping the reverse pool", flush=True)
                pool.terminate()
                break
    print("complete", flush=True)


if __name__ == "__main__":
    main()
