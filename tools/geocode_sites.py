"""Geocode Phase-0 sample sites with OSM Nominatim (1 req/s, cached). Town/county precision is enough for the
0.25-degree ERA5 grid. Results are cached in data/sites/geocode_cache.json and reviewed by hand (DD-004)."""
import json, time, sys, urllib.parse, urllib.request, pathlib

CACHE = pathlib.Path("data/sites/geocode_cache.json")
QUERIES = {
    # site: query string (Epoch address when present; otherwise the announced town / county)
    "Microsoft Fairwater Wisconsin": "4800 90th St, Mount Pleasant, WI 53403",
    "Anthropic-Amazon New Carlisle": "New Carlisle, St. Joseph County, Indiana",
    "OpenAI Stargate New Mexico": "Santa Teresa, Dona Ana County, New Mexico",
    "Meta Hyperion": "Holly Ridge, Louisiana 71269",
    "Colossus 2": "5420 Tulane Rd, Memphis, TN 38109",
    "OpenAI Stargate Shackelford": "Albany, Shackelford County, Texas",
    "QTS Cedar Rapids": "Fairfax, Iowa 52228",
    "Meta Prometheus": "New Albany, Ohio 43054",
    "Goodnight": "Claude, Texas 79019",
    "OpenAI Stargate UAE": "Al Bihouth, Abu Dhabi",
    "OpenAI Stargate Michigan": "Saline Township, Washtenaw County, Michigan",
    "OpenAI Stargate Wisconsin": "Port Washington, Wisconsin",
    "OpenAI Stargate Abilene": "5502 Spinks Rd, Abilene, TX 79601",
    "Microsoft Goodyear": "14250 W Broadway Rd, Goodyear, AZ 85338",
    "Nebius Mantsala": "Mantsala, Finland",
    "Lulea (check site, non-Epoch)": "Lulea, Sweden",
}

def geocode(q):
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": q, "format": "json", "limit": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "hotloop-research/0.1 (academic research; set your own contact per the Nominatim usage policy)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
for site, q in QUERIES.items():
    if site in cache and cache[site].get("query") == q:
        continue
    try:
        res = geocode(q)
    except Exception as e:  # network failure is logged, not fatal
        print(site, "FAILED", e, file=sys.stderr); time.sleep(1.2); continue
    cache[site] = {"query": q, "result": res[0] if res else None}
    CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    time.sleep(1.2)
for site, v in cache.items():
    r = v["result"]
    print(f"{site:38s} | {v['query'][:45]:45s} | " + (f"{float(r['lat']):8.4f} {float(r['lon']):9.4f} | {r['display_name'][:70]}" if r else "NOT FOUND"))
