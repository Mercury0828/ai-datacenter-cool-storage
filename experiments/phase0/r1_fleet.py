"""R1 stylized fleet run (pre-registration: ledger PR-R1, commit e69a14f).

Items = (site, climate, variant). Each item writes one JSON atomically to experiments/phase0/out/r1/items/ and the
manifest (experiments/phase0/out/r1/manifest.json) is rewritten atomically after every item. On launch the run resumes
from existing item files (skip-done); `--fresh` starts over (guide §7 checkpoint-and-resume).

Usage: python experiments/phase0/r1_fleet.py [--workers 16] [--fresh] [--only SITE]
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import subprocess
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import design as D  # noqa: E402
from model import evaluate as E  # noqa: E402
from model import params as PR  # noqa: E402

OUT = ROOT / "experiments/phase0/out/r1"
ITEMS = OUT / "items"
CLIMATES = {"current": 0.0, "2040": 1.2}  # DD-008
VARIANTS = ["headline", "hybrid", "t40", "water_constrained"]
ETA2 = 0.5076  # M1-ii calibration
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def item_path(site, climate, variant):
    return ITEMS / f"{site}__{climate}__{variant}.json"


def variant_setup(variant):
    if variant == "headline":
        return PR.params(ch_etaII=ETA2), True
    if variant == "hybrid":
        return PR.params(ch_etaII=ETA2, f_air=0.15), True
    if variant == "t40":
        return PR.params(ch_etaII=ETA2, T_TCS_max=40.0), True
    if variant == "water_constrained":
        return PR.params(ch_etaII=ETA2), False
    raise KeyError(variant)


def compact(d):
    keep = ("family", "k", "ad", "tank_h", "c_ch", "kct", "n_viol", "compliant", "capex", "cost", "evap", "dc_approach")
    out = {k: d[k] for k in keep}
    if d.get("ox"):
        out.update({f"ox_{k}": v for k, v in d["ox"].items()})
    return out


def combine(liq, air, f_air):
    """Site-level classification from the two paths (DD-007 §8)."""
    if air is None:
        return {k: liq[k] for k in ("F_dry", "F_nc", "zone")} | {"cost": liq["best"]["cost"] if liq["best"] else None}
    if liq["best"] is None or air["best"] is None:
        return {"F_dry": False, "F_nc": False, "zone": "Zempty", "cost": None}
    zones = {liq["zone"], air["zone"]}
    zone = "Z3" if "Z3" in zones else ("Z2" if "Z2" in zones else "Z1")
    return {"F_dry": liq["F_dry"] and air["F_dry"], "F_nc": liq["F_nc"] and air["F_nc"], "zone": zone,
            "cost": (1 - f_air) * liq["best"]["cost"] + f_air * air["best"]["cost"]}


def run_site(args):
    site, todo = args
    t0 = time.time()
    W0 = E.load_weather(ROOT / f"data/weather/{site}.npz")
    done = []
    for climate, dT in CLIMATES.items():
        need = [v for v in VARIANTS if (climate, v) in todo]
        if not need:
            continue
        W = E.shift_weather(W0, dT) if dT else W0
        for variant in need:
            P, water = variant_setup(variant)
            lim_liq = PR.fws_max(P)
            liq_designs, m = D.enumerate_path(P, W, lim_liq, water_allowed=water)
            liq = D.classify(liq_designs)
            rec = {"site": site, "climate": climate, "dT": dT, "variant": variant, "m_allowed": m, "commit": COMMIT,
                   "n_years": int(W["t_db"].shape[0]), "config_hash": PR.config_hash(P), "water_allowed": water,
                   "liquid": {"lim": lim_liq, "F_dry": liq["F_dry"], "F_nc": liq["F_nc"], "zone": liq["zone"],
                              "best": compact(liq["best"]) if liq["best"] else None,
                              "designs": [compact(d) for d in liq_designs]}}
            air = None
            if P["f_air"] > 0:
                lim_air = P["T_sa_max"] - P["dT_coil"]
                air_designs, _ = D.enumerate_path(P, W, lim_air, water_allowed=water)
                air = D.classify(air_designs)
                rec["air"] = {"lim": lim_air, "F_dry": air["F_dry"], "F_nc": air["F_nc"], "zone": air["zone"],
                              "best": compact(air["best"]) if air["best"] else None,
                              "designs": [compact(d) for d in air_designs]}
            rec["site_result"] = combine(liq, air, P["f_air"])
            rec["elapsed_s"] = round(time.time() - t0, 1)
            p = item_path(site, climate, variant)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, default=float), encoding="utf-8")
            os.replace(tmp, p)  # atomic
            done.append((site, climate, variant))
    return done


def write_manifest(all_items):
    state = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "done": [f"{s}__{c}__{v}" for s, c, v in all_items if item_path(s, c, v).exists()],
             "pending": [f"{s}__{c}__{v}" for s, c, v in all_items if not item_path(s, c, v).exists()]}
    tmp = OUT / "manifest.tmp"
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, OUT / "manifest.json")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    ITEMS.mkdir(parents=True, exist_ok=True)
    (OUT / "run.pid").write_text(str(os.getpid()), encoding="utf-8")  # stop only this PID tree (never by pattern)
    sites = pd.read_csv(ROOT / "data/sites/phase0_sites.csv").site_id.tolist()
    if a.only:
        sites = [s for s in sites if s == a.only]
    all_items = [(s, c, v) for s in sites for c in CLIMATES for v in VARIANTS]
    if a.fresh:
        for s, c, v in all_items:
            item_path(s, c, v).unlink(missing_ok=True)
    st = write_manifest(all_items)
    print(f"{len(st['done'])} items done, {len(st['pending'])} pending (resume)", flush=True)
    by_site = {}
    for s, c, v in all_items:
        if not item_path(s, c, v).exists():
            by_site.setdefault(s, set()).add((c, v))
    t0 = time.time()
    with mp.Pool(min(a.workers, max(1, len(by_site)))) as pool:
        for done in pool.imap_unordered(run_site, list(by_site.items())):
            st = write_manifest(all_items)
            print(f"[{time.time() - t0:6.0f}s] finished {done[0][0] if done else '-'}: {len(done)} items; "
                  f"total done {len(st['done'])}/{len(all_items)}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
