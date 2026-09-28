"""Full-hub site table (all 93 Epoch sites): capacity bases, geocodes, nearest LAND ERA5 grid point (DD-004a), and a
grid-cell id so weather is extracted once per unique cell. Outputs data/sites/hub_sites.csv and data/sites/hub_cells.csv."""
import json, pathlib, sys
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_site_table import capacity_bases, slug  # noqa: E402

cap = capacity_bases()
geo = json.loads((ROOT / "data/sites/geocode_hub_cache.json").read_text(encoding="utf-8"))
st = np.load(ROOT / "data/sites/era5_static_hsurf_soil.npz"); soil, hs = st["soil_type"], st["HSURF"]
rows = []
for _, r in cap.iterrows():
    g = geo[r.site]["result"]; lat, lon = float(g["lat"]), float(g["lon"])
    i0, j0 = int(round((lat + 90) / 0.25)), int(round((lon + 180) / 0.25)) % 1440; best = None
    for i in range(i0 - 2, i0 + 3):
        for j in range(j0 - 2, j0 + 3):
            jj = j % 1440
            if soil[i, jj] <= 0: continue
            gl, go = i * 0.25 - 90, jj * 0.25 - 180
            a = np.sin(np.radians(gl - lat) / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(gl)) * np.sin(np.radians(go - lon) / 2) ** 2
            d = 2 * 6371 * np.arcsin(np.sqrt(a))
            if best is None or d < best[0]: best = (d, i, jj)
    rows.append({"site_id": slug(r.site), "site": r.site, "country": r.Country, "B_full_MW": r.B_full_MW,
                 "B_2027_MW": r.B_2027_MW, "B_now_MW": r.B_now_MW, "lat": lat, "lon": lon, "geocode_query": geo[r.site]["query"],
                 "era5_ilat": best[1], "era5_ilon": best[2], "era5_dist_km": round(best[0], 1),
                 "era5_lat": best[1] * 0.25 - 90, "era5_lon": best[2] * 0.25 - 180, "era5_orog_m": float(hs[best[1], best[2]]),
                 "cell_id": f"cell_{best[1]}_{best[2]}", "weighted": True})
df = pd.DataFrame(rows); df.to_csv(ROOT / "data/sites/hub_sites.csv", index=False)
cells = df.groupby("cell_id").agg(era5_ilat=("era5_ilat", "first"), era5_ilon=("era5_ilon", "first"),
                                  era5_lat=("era5_lat", "first"), era5_lon=("era5_lon", "first"),
                                  era5_orog_m=("era5_orog_m", "first"), n_sites=("site_id", "count"),
                                  B_full_MW=("B_full_MW", "sum")).reset_index().rename(columns={"cell_id": "site_id"})
cells["weighted"] = False
cells.to_csv(ROOT / "data/sites/hub_cells.csv", index=False)
print(len(df), "sites ->", len(cells), "unique ERA5 land cells; hub B-full", round(df.B_full_MW.sum()), "MW")
print(df.groupby("country").B_full_MW.sum().sort_values(ascending=False).round(0).to_string())
