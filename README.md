# Cool-water storage and dry cooling for AI data centres: model, scripts and derived data

This repository holds the thermal-chain model, the analysis scripts, the derived site and grid-cell tables and every result
file behind the figures and numbers of

> Jiachen Shen and Hui Zhong. *Cool-water storage and dry heat rejection for liquid-cooled AI data centers.* Manuscript,
> 2026.

Release 1.1 adds the analyses, result files and figure sources of the current manuscript version (`MANIFEST.md`, section
"Current manuscript version"). Release 1.0 matches the first version of the manuscript.

The study applies one 45 °C liquid-cooling reference design to the 93 AI data-centre sites tracked by Epoch AI (37.3 GW of
announced IT capacity). Each site is simulated hour by hour over ERA5 weather for 1990–2025, over the same weather with the
CMIP6 warming expected around 2040, and under a uniform +1.2 K stress. A design passes when the rack coolant stays within its
45 °C limit in at least 35 of the 36 years.

## Contents

| Path | Contents |
|---|---|
| `model/` | Thermal chain (dry cooler, plate exchanger, trim chiller, adiabatic pads, cooling towers), psychrometrics, design evaluation and the store recursion |
| `experiments/phase0/`, `experiments/phase1/` | Analysis scripts; results in `out/`, mostly one JSON file per grid cell and climate |
| `experiments/phase1/queue*.json`, `run_queue.ps1` | Exact arguments of the long runs, in the order they were run |
| `paper/figs/src/`, `paper/si/make_si.py` | Scripts that draw the figures and write the supplemental tables from the result files |
| `data/sites/`, `data/epoch/` | Site frame and capacity bases (Epoch AI snapshot of 24 September 2026), grid cells, station lists |
| `data/era5_cds_tables/` | Hourly 2 m temperature, 2 m dew point and surface pressure, 1990–2025, at the 106 ERA5 points of record |
| `data/weather*/` | Earlier ERA5 point tables (Open-Meteo copy), kept because some analyses ran on them (Note S2 of the paper) |
| `tools/` | Data-acquisition scripts (ERA5 points, NOAA ISD-Lite, geocoding, site tables) |
| `tests/` | Unit tests of the thermal chain and helpers (`python -m pytest tests`) |
| `docs/cost_basis.md` | Sources of the cost parameters |
| `MANIFEST.md` | For every result, the scripts in run order and the files that carry it |
| `release/` | Pinned environment, input checksums and the one-cell reproduction script |

## Reproducing the results

1. **Environment.** Python 3.12.10 (`release/python_version.txt`); `pip install -r release/requirements-lock.txt`. Run the
   numerics single-threaded (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS` = 1); the
   scripts set these themselves.
2. **Large inputs.** The scripts read large inputs from `D:/_caches`, the cache drive of the machine the study ran on. To run
   elsewhere, create the same layout or replace that prefix in the scripts. The ERA5 tables of record are included under
   `data/era5_cds_tables/`; copy them to `D:/_caches/era5/cds/tables/`, or rebuild them from the Copernicus Climate Data
   Store with `experiments/phase1/cds_extract.py` and your own CDS account. The CMIP6 changes used by the scenarios are
   included in `experiments/phase1/out/cmip/`; `experiments/phase1/cmip_extract.py` recomputes them from the Pangeo CMIP6
   archive. NOAA ISD-Lite station records are fetched by `tools/fetch_isd_lite.py`; US Climate Reference Network hourly
   files (NOAA NCEI, `hourly02`) are read from `D:/_caches/uscrn/hourly02`. `release/inputs.sha256` lists the SHA-256 of the
   input files.
3. **Quick check.** `python release/reproduce_one_cell.py` rebuilds the Abilene, Texas grid cell from the raw Climate Data
   Store archive and compares its storage requirement under the three climates with the archived result.
4. **Figures and tables.** `python paper/figs/src/fig2_boundary.py` (and `fig3_warming.py` to `fig6_drift.py`) draw the main
   figures from the result files; `python paper/si/make_si.py` writes the supplemental tables and figures. The figures of
   the current manuscript version are drawn by `paper_ae/figs/src/fig2_boundary.py` to `fig6_screen.py`; Figure 1 is
   `paper_ae/figs/src/fig1.tex`, compiled with `pdflatex` (TikZ). The analyses added in this version are run with
   `python experiments/phase1/ae_w1_bases.py`, `w3_risk.py --part all`, `ae_sc1_ruler1d.py`, `ae_sc2_finite_sample.py`
   and `ae_sc3_dispatch.py` (see `MANIFEST.md`).

Long runs are resumable: each writes one file per item and skips finished items. `MANIFEST.md` lists the scripts behind each
result in run order.

## Licences

Code is released under the MIT licence (`LICENSE`). Derived data and result files are released under CC BY 4.0
(`LICENSE-DATA.md`). Third-party inputs keep their own terms, listed in `LICENSE-DATA.md`.

## Citation

See `CITATION.cff`.
