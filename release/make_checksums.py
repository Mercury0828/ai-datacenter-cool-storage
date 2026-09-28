"""Release package S1: SHA-256 of every input file (repository inputs and the local data caches on D:), written to
release/inputs.sha256 as '<sha256>  <path>' lines. Usage: python release/make_checksums.py"""
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
GLOBS = [(ROOT, "data/sites/*"), (ROOT, "data/epoch/**/*"), (ROOT, "data/weather/*.npz"), (ROOT, "data/weather_mdhubs/*.npz"),
         (ROOT, "data/weather_nonepoch/*.npz"), (ROOT, "experiments/phase1/out/cmip/deltas_*.json"),
         (ROOT, "experiments/phase1/out/r2f/deltas_2060_*.json"), (ROOT, "lit/cost_sources/*"),
         (pathlib.Path("D:/_caches/era5/cds"), "tables/*.npz"), (pathlib.Path("D:/_caches/era5/cds"), "raw/*.zip"),
         (pathlib.Path("D:/_caches/era5/cds"), "tables_1960/*.npz"), (pathlib.Path("D:/_caches/era5/cds"), "raw_1960/*.zip"),
         (pathlib.Path("D:/_caches/cmip6"), "pangeo_cmip6.csv"), (pathlib.Path("D:/_caches/cmip6"), "cv/CMIP6_source_id.json")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lines = []
    for base, pat in GLOBS:
        for p in sorted(base.glob(pat)):
            if p.is_file() and not p.name.endswith((".tmp", ".part")) and ".tmp." not in p.name:
                shown = p.relative_to(ROOT).as_posix() if base == ROOT else p.as_posix()
                lines.append(f"{sha(p)}  {shown}")
    (ROOT / "release/inputs.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(len(lines), "files")


if __name__ == "__main__":
    main()
