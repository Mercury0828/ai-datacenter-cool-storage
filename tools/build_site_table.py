"""Build the Phase-0 site table (DD-003, DD-004).

Reads the Epoch snapshot, computes the three capacity bases (B-full headline, B-now, B-2027) from
data_center_timelines.csv, applies the declared Phase-0 sample rule, joins geocoded coordinates and maps
each site to its nearest ERA5 0.25-degree grid point. Output: data/sites/phase0_sites.csv and
data/sites/epoch_capacity_bases.csv (all 93 hub sites, for the denominators).
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SNAP = ROOT / "data/epoch/snapshot_20260924"
TODAY = pd.Timestamp("2026-09-24")
END_2027 = pd.Timestamp("2027-12-31")

EXTRA_SAMPLE = ["OpenAI Stargate Abilene", "Microsoft Goodyear", "Nebius Mantsala"]  # DD-003
CHECK_SITES = {"Lulea (check site, non-Epoch)": "lulea"}  # unweighted (DD-003)
N_LARGEST = 12


def capacity_bases() -> pd.DataFrame:
    t = pd.read_csv(SNAP / "data_center_timelines.csv", parse_dates=["Date"])
    meta = pd.read_csv(SNAP / "data_centers.csv")
    rows = []
    for name, g in t.groupby("Data center"):
        g = g.sort_values("Date")
        now = g[g.Date <= TODAY]
        e27 = g[g.Date <= END_2027]
        rows.append({
            "site": name,
            "B_now_MW": float(now["IT power (MW)"].iloc[-1]) if len(now) else 0.0,
            "B_2027_MW": float(e27["IT power (MW)"].iloc[-1]) if len(e27) else 0.0,
            "B_full_MW": float(g["IT power (MW)"].iloc[-1]),
            "B_full_date": g.Date.iloc[-1].date().isoformat(),
            "facility_full_MW": float(g["Power (MW)"].iloc[-1]),
        })
    df = pd.DataFrame(rows).merge(meta[["Name", "Country", "Address", "Current chip types"]],
                                  left_on="site", right_on="Name", how="left").drop(columns="Name")
    return df.sort_values("B_full_MW", ascending=False).reset_index(drop=True)


def slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_").replace("__", "_")


def main():
    cap = capacity_bases()
    cap.to_csv(ROOT / "data/sites/epoch_capacity_bases.csv", index=False)
    sample = list(cap.site.iloc[:N_LARGEST]) + [s for s in EXTRA_SAMPLE if s not in set(cap.site.iloc[:N_LARGEST])]
    geo = json.loads((ROOT / "data/sites/geocode_cache.json").read_text(encoding="utf-8"))
    out = []
    for s in sample:
        r = cap[cap.site == s].iloc[0]
        g = geo[s]["result"]
        out.append({"site_id": slug(s), "site": s, "weighted": True, "rank_B_full": int(cap.index[cap.site == s][0]) + 1,
                    "B_full_MW": r.B_full_MW, "B_2027_MW": r.B_2027_MW, "B_now_MW": r.B_now_MW,
                    "B_full_date": r.B_full_date, "country": r.Country, "chips_now": r["Current chip types"],
                    "lat": float(g["lat"]), "lon": float(g["lon"]), "geocode_query": geo[s]["query"]})
    for s, sid in CHECK_SITES.items():
        g = geo[s]["result"]
        out.append({"site_id": sid, "site": "Lulea (check site)", "weighted": False, "rank_B_full": None,
                    "B_full_MW": 0.0, "B_2027_MW": 0.0, "B_now_MW": 0.0, "B_full_date": None, "country": "Sweden",
                    "chips_now": None, "lat": float(g["lat"]), "lon": float(g["lon"]), "geocode_query": geo[s]["query"]})
    df = pd.DataFrame(out)
    # DD-004 amendment: nearest ERA5 grid point that is LAND (Open-Meteo static soil_type > 0; water cells have
    # soil_type 0 and HSURF = -999). Distances are great-circle.
    st = np.load(ROOT / "data/sites/era5_static_hsurf_soil.npz")
    soil, hsurf = st["soil_type"], st["HSURF"]
    picks = []
    for _, r in df.iterrows():
        i0, j0 = int(round((r.lat + 90) / 0.25)), int(round((r.lon + 180) / 0.25)) % 1440
        best = None
        for i in range(i0 - 2, i0 + 3):
            for j in range(j0 - 2, j0 + 3):
                jj = j % 1440
                if soil[i, jj] <= 0:
                    continue
                glat, glon = i * 0.25 - 90, jj * 0.25 - 180
                dlat, dlon = np.radians(glat - r.lat), np.radians(glon - r.lon)
                a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(r.lat)) * np.cos(np.radians(glat)) * np.sin(dlon / 2) ** 2
                d = 2 * 6371.0 * np.arcsin(np.sqrt(a))
                if best is None or d < best[0]:
                    best = (d, i, jj, (i0, j0) == (i, jj))
        picks.append(best)
    df["era5_ilat"] = [b[1] for b in picks]
    df["era5_ilon"] = [b[2] for b in picks]
    df["era5_dist_km"] = [round(b[0], 1) for b in picks]
    df["era5_nearest_is_land"] = [b[3] for b in picks]
    df["era5_lat"] = df.era5_ilat * 0.25 - 90
    df["era5_lon"] = df.era5_ilon * 0.25 - 180
    df["era5_orog_m"] = [float(hsurf[i, j]) for i, j in zip(df.era5_ilat, df.era5_ilon)]
    df.to_csv(ROOT / "data/sites/phase0_sites.csv", index=False)
    w = df[df.weighted]
    print(df[["site_id", "B_full_MW", "lat", "lon", "era5_lat", "era5_lon", "era5_dist_km", "era5_nearest_is_land", "era5_orog_m"]].to_string())
    for b in ("B_full_MW", "B_2027_MW", "B_now_MW"):
        print(f"{b}: sample {w[b].sum():,.0f} MW of hub {cap[b].sum():,.0f} MW ({w[b].sum() / cap[b].sum():.1%})")


if __name__ == "__main__":
    main()
