"""Download NOAA ISD-lite hourly files (1990-2025) for the BV-3 stations into D:/_caches/isd/lite (resumable: skips files
already present; a missing year is recorded, not fatal). ISD-lite fields: year month day hour T(0.1C) Td(0.1C) SLP(0.1hPa) ..."""
import json, pathlib, time, urllib.request
STATIONS = {"KABI": "722660-13962", "KPHX": "722780-23183", "KELP": "722700-23044", "KMEM": "723340-13893",
            "KCMH": "724280-14821", "OMAA": "412170-99999", "ESPA": "021860-99999"}
OUT = pathlib.Path("D:/_caches/isd/lite"); OUT.mkdir(parents=True, exist_ok=True)
missing = []
for icao, sid in STATIONS.items():
    for y in range(1990, 2026):
        f = OUT / f"{sid}-{y}.gz"
        if f.exists() and f.stat().st_size > 0:
            continue
        url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{y}/{sid}-{y}.gz"
        for a in range(3):
            try:
                data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hotloop-research"}), timeout=60).read()
                tmp = f.with_suffix(".tmp"); tmp.write_bytes(data); tmp.replace(f); break
            except Exception as e:
                err = str(e); time.sleep(2 * (a + 1))
        else:
            missing.append((icao, y, err))
        time.sleep(0.2)
(OUT / "missing.json").write_text(json.dumps(missing, indent=1))
print("files:", len(list(OUT.glob("*.gz"))), "missing:", missing)
