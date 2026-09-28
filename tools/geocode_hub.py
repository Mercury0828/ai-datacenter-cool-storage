"""Geocode all Epoch hub sites (Nominatim, 1 req/s, cached) for the full-hub screen (Phase-0 ladder data / Phase 1).
Query = Epoch address when present, else '<town/region from the site name>, <country>' (manual map below)."""
import json, pathlib, sys, time, urllib.parse, urllib.request
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/sites/geocode_hub_cache.json"
MANUAL = {  # sites without a usable Epoch address -> announced town/region
    "Google Mesa": "Mesa, Arizona", "Google Storey County": "McCarran, Storey County, Nevada",
    "DayOne Nusajaya": "Nusajaya, Johor, Malaysia", "Oracle Batam": "Batam, Indonesia",
    "Nebius Mantsala": "Mantsala, Finland", "Microsoft Narvik Norway": "Narvik, Norway",
    "Southgate Melbourne": "Melbourne, Victoria, Australia", "Anthropic Barber Lake": "Abilene, Texas",
    "Google Kansas City East": "Kansas City, Missouri", "OpenAI Stargate Michigan": "Saline Township, Michigan",
    "OpenAI Stargate Milam": "Rockdale, Milam County, Texas", "OpenAI Stargate New Mexico": "Santa Teresa, New Mexico",
    "OpenAI Stargate Wisconsin": "Port Washington, Wisconsin", "OpenAI Stargate UAE": "Al Bihouth, Abu Dhabi",
    "Anthropic-Amazon New Carlisle": "New Carlisle, Indiana", "Meta Hyperion": "Holly Ridge, Louisiana",
    "OpenAI Stargate Shackelford": "Albany, Shackelford County, Texas", "QTS Cedar Rapids": "Fairfax, Iowa",
    "Goodnight": "Claude, Texas", "VNET Bayin Ulanqab": "Ulanqab, Inner Mongolia, China",
    "Huawei Horinger": "Horinger, Hohhot, Inner Mongolia, China", "Huawei Wuhu": "Wuhu, Anhui, China",
    "Alibaba Zhangbei": "Zhangbei, Zhangjiakou, Hebei, China", "Amazon Madison Mega Site": "Canton, Mississippi",
    "CoreWeave Norway": "Ovrebo, Norway", "EcoDataCenter 2": "Borlange, Sweden", "Nscale Keflavik": "Reykjanesbaer, Iceland",
    "Start Campus Sines Data Campus": "Sines, Portugal", "Google Waltham Cross": "Cheshunt, England",
    "DayOne Kempas": "Kempas, Johor Bahru, Malaysia", "Stream Phoenix": "Goodyear, Arizona",
    "AWS New Albany": "New Albany, Ohio", "CoreWeave Lancaster Greenfield site": "Lancaster, Pennsylvania",
    "Meta Cheyenne": "Cheyenne, Wyoming",
    "Meta Prometheus": "New Albany, Ohio", "Meta Rosemount": "Rosemount, Minnesota", "Google Bristow": "Bristow, Virginia",
    "QTS Richmond 2": "Sandston, Virginia", "QTS Richmond 3": "Sandston, Virginia", "Meta Montgomery": "Montgomery, Alabama",
    "Meta Huntsville": "Toney, Alabama", "Coreweave Helios": "Afton, Dickens County, Texas",
    "Microsoft Project Osmium": "Cumming, Iowa", "Meta Eagle Mountain": "Eagle Mountain, Utah",
    "Google Arcola": "Sterling, Virginia", "Google Papillion": "Papillion, Nebraska", "Meta Gallatin": "Gallatin, Tennessee",
    "Meta Sarpy": "Springfield, Sarpy County, Nebraska", "CoreWeave Chester VA": "Chester, Virginia",
    "Meta Aiken": "Trenton, South Carolina", "Microsoft SAT40": "San Antonio, Texas",
    "Meta Bowling Green": "Bowling Green, Ohio", "OpenAI Stargate Lordstown": "Lordstown, Ohio",
}


def geocode(q):
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": q, "format": "json", "limit": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "hotloop-research/0.1 (academic research; set your own contact per the Nominatim usage policy)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


d = pd.read_csv(ROOT / "data/epoch/snapshot_20260924/data_centers.csv", encoding="utf-8", encoding_errors="replace")
cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
for _, r in d.iterrows():
    name = r["Name"]
    qs = [MANUAL[name]] if name in MANUAL else []
    if isinstance(r["Address"], str) and "\ufffd" not in r["Address"]:
        qs.append(r["Address"].replace("\n", " "))
    if not qs:
        qs = [f"{name}, {r['Country']}"]
    if name in cache and cache[name].get("result"):
        continue
    res, used = None, None
    for q in qs:
        try:
            out = geocode(q)
        except Exception as e:
            print(name, "FAILED", e, file=sys.stderr); out = []
        time.sleep(1.2)
        if out:
            res, used = out[0], q
            break
    cache[name] = {"query": used or qs[0], "result": res}
    CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
missing = [n for n, v in cache.items() if not v.get("result")]
print("geocoded", len(cache) - len(missing), "of", len(d), "; missing:", missing)
