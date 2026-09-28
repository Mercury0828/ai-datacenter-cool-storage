"""Release package S1: reproduce one hub cell of headline A from raw inputs.

Steps: (1) read the raw Copernicus CDS archive of the cell's ERA5 point (dataset reanalysis-era5-single-levels-timeseries,
1990-2025 hourly 2 m temperature, 2 m dew point, surface pressure; downloaded with experiments/phase1/cds_extract.py, which can
fetch it again with a CDS account); (2) convert it to the table format and check it equals the table of record; (3) run the
headline screen (experiments/phase1/hw_screen.py::screen_path) under the historical record, the uniform +1.2 K stress and the
SSP2-4.5 circa-2040 scenario; (4) compare h* (hours of storage needed), F_dry and F_nc with the stored rebuild of record
(experiments/phase1/out/m2/cds_rebuild/<cell>.json) and print the capacity the cell carries.
Usage: python release/reproduce_one_cell.py [--cell cell_490_321]
"""
import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/phase1"))
import cds_extract as X  # noqa: E402
from cmip_scenarios import SCEN, scenario_weather  # noqa: E402
from hw_screen import screen_path  # noqa: E402
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="cell_490_321")
    a = ap.parse_args()
    pts = pd.read_csv(ROOT / "experiments/phase1/out/m2/cds_points.csv")
    row = pts[pts.names.str.contains(f"hub_cells:{a.cell}(?:;|$)", regex=True)].iloc[0]
    raw = X.RAW / f"{row.point}__1990-2025.zip"
    print("raw archive:", raw, "exists:", raw.exists())
    with tempfile.TemporaryDirectory(dir=X.BASE, ignore_cleanup_errors=True) as d:  # scratch on D:, not C:
        tab = pathlib.Path(d) / f"{row.point}.npz"
        X.to_table(X.read_zip(raw), float(row.lat), float(row.lon), tab)
        keys = ("t2m_c100", "td_c100", "psfc_pa10")
        with np.load(tab) as mine, np.load(X.TAB / f"{row.point}.npz") as rec:
            same = all(np.array_equal(mine[k], rec[k]) for k in keys)
        print("table rebuilt from the raw archive equals the table of record:", same)
    W = E.load_weather(X.TAB / f"{row.point}.npz")  # identical content (checked above)
    P = PR.params(ch_etaII=0.5076)
    lim = PR.fws_max(P)
    dall = json.loads((ROOT / "experiments/phase1/out/cmip/deltas_all.json").read_text(encoding="utf-8"))
    exp, q, detrend = SCEN["ssp245_median"]
    Ws = {"historical": W, "stress_plus1.2K": E.shift_weather(W, 1.2),
          "ssp245_median": scenario_weather(W, np.array(dall[exp][q])[:, dall["cells"].index(a.cell)], detrend)}
    stored = json.loads((ROOT / f"experiments/phase1/out/m2/cds_rebuild/{a.cell}.json").read_text(encoding="utf-8"))
    ok = True
    for cl, Wx in Ws.items():
        r = screen_path(P, Wx, lim, D.allowed_violations(P["eta_risk"], Wx["t_db"].shape[0]))
        s = stored[cl]
        match = abs(r["h_star"] - s["h_star"]) < 1e-9 and r["F_dry"] == s["F_dry"] and r["F_nc"] == s["F_nc"]
        ok &= match
        print(f"{cl:16s} h* = {r['h_star']:8.3f} h  F_dry(8 h) = {r['h_star'] <= 8 + 1e-9}  F_nc = {r['F_nc']}  matches record: {match}")
    sites = pd.read_csv(ROOT / "data/sites/hub_sites.csv")
    mw = sites.loc[sites.cell_id == a.cell, "B_full_MW"].sum()
    print(f"capacity in this cell (B-full): {mw:.0f} MW of the 37,295 MW frame")
    print("REPRODUCED" if ok and same else "MISMATCH")


if __name__ == "__main__":
    main()
