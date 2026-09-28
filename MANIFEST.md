# From inputs to each result

Scripts are listed in run order. Outputs are in `experiments/phase1/out/` unless stated otherwise. Identifiers such as
`PR-FF` or `SB-3` in script headers name the analyses as they were declared before they ran.

## Environment and inputs

- Python 3.12.10 (`release/python_version.txt`); packages pinned in `release/requirements-lock.txt`. Numerics run
  single-threaded; every script sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMBA_NUM_THREADS` to 1.
- `release/inputs.sha256` lists the SHA-256 of the input files (repository inputs and the local caches);
  `release/make_checksums.py` regenerates it.

| Input | Source | Location |
|---|---|---|
| Hourly 2 m temperature, 2 m dew point and surface pressure, 1990–2025, at 106 ERA5 points (data of record) | Copernicus Climate Data Store, `reanalysis-era5-single-levels-timeseries` | `data/era5_cds_tables/` (copy to `D:/_caches/era5/cds/tables/`); rebuilt from the raw archives by `experiments/phase1/cds_extract.py` |
| Same variables, 1960–1989, at the 15 robust-sizing sites | as above | fetched by `experiments/phase1/r2f_extract_1960.py` into `D:/_caches/era5/cds/tables_1960/` |
| Earlier ERA5 point tables (Open-Meteo copy) | Open-Meteo archive of ERA5 | `data/weather*/`; the hub tables of this copy are assembled by `tools/fetch_era5_om.py` and `tools/assemble_weather.py` into `D:/_caches/era5/weather_hub/` |
| CMIP6 monthly near-surface air temperature | Pangeo CMIP6 cloud archive | changes in `experiments/phase1/out/cmip/deltas_*.json` and `out/r2f/deltas_2060_*.json`, from `cmip_extract.py` and `r2f_cmip_2060.py` |
| AI data-centre frame | Epoch AI *Frontier Data Centers* hub | `data/epoch/`; `data/sites/hub_sites.csv`, `hub_cells.csv` from `tools/build_site_table.py`, `tools/geocode_hub.py`, `tools/build_hub_table.py` |
| Station records | NOAA ISD-Lite, USCRN | `data/sites/bv3_stations.csv`, `as2_uscrn_stations.csv`; series fetched by `tools/fetch_isd_lite.py` |
| Cost sources | public reports | `docs/cost_basis.md` |

The scripts that run on the data of record read the ERA5 tables by grid cell from `D:/_caches/era5/cds/weather_hub_cds/`
(`st_store.py`, `jl_joint.py`, and `ff_menu.py --weather-dir` for the replication): one copy of each hub cell's table named
by its cell, following the point-to-cell map in `experiments/phase1/out/m2/cds_points.csv`.

## Quick check

`python release/reproduce_one_cell.py` rebuilds the Abilene, Texas cell (`cell_490_321`) from the raw archive, checks the table
against the table of record and reruns the screen for 1990–2025, the uniform +1.2 K stress and SSP2-4.5 around 2040. Expected:

```
table rebuilt from the raw archive equals the table of record: True
historical       h* =    6.526 h  F_dry(8 h) = True   F_nc = True  matches record: True
stress_plus1.2K  h* =   17.393 h  F_dry(8 h) = False  F_nc = True  matches record: True
ssp245_median    h* =   10.324 h  F_dry(8 h) = False  F_nc = True  matches record: True
capacity in this cell (B-full): 1722 MW of the 37,295 MW frame
REPRODUCED
```

## Results

| Result in the paper | Scripts, in run order | Outputs |
|---|---|---|
| Thermal-chain calibration (Note S3) | `experiments/phase0/m1_calibration.py` | `experiments/phase0/out/m1_calibration.json` |
| Water- and compressor-free boundary, 34.3 of 37.3 GW; stores by site (Figure 2; Notes S4, S5) | `experiments/phase0/hub_screen.py` → `bv1_min_storage.py` → `bv2_continuous.py`, `bv3_station_bias.py`, `bv3_lapse.py`, `bv4_crosswalk.py` → `bv_aggregate.py`; on the data of record `cds_extract.py` → `cds_rebuild.py`; `sb_storage.py` | `bv_summary.json`, `m2/cds_rebuild_summary.json`, `m2/cds_rebuild/`, `sb/sb1_storage.json` |
| Store piping, water temperatures and tank volumes (Figure S3; Note S4) | `st_store.py` | `st/`, `st_summary.json` |
| Warming around 2040, climate-model spread, robustness classes and the 1 K lever (Figure 3; Notes S7–S9) | `cmip_extract.py` → `cmip_validate.py` → `cmip_scenarios.py`; `sb_screen.py` (then `--extra`) → `sb_aggregate.py` → `sb_price1k.py`; `jl_joint.py` | `cmip/cmip_summary.json`, `sb/screen/`, `sb/sb23_robustness.json`, `sb/sb4_price1k.json`, `jl/`, `jl_summary.json` |
| Sensitivities of the boundary (Note S10) | `as_stress.py`, `as2_uscrn_nights.py` → `as2_rural_bound.py`, `m2_pressure_sensitivity.py`, `bv3_station_bias.py`, `bv3_lapse.py` | `as/`, `as_summary.json`, `as2/`, `m2/` |
| Climate ruler (Figure 4; Note S13) | `ruler.py` → `ruler_b.py`; `as2_ruler.py` | `ruler/`, `ruler_b/`, `as2/` |
| Water versus compressors, the price of a water ban, cooling electricity and peaks (Figure 5; Notes S5, S6) | `ff_menu.py` (and `ff_menu_reverse.py`) → `ff_water_frontier.py` → `ff_aggregate.py` → `ff_report.py`; `de_analysis.py`, `de3_tank_chiller.py` → `de3_aggregate.py`; `as_cost.py`; `cmip_d_summary.py`; `cost_reprice.py` → `cost_reprice_b.py` → `cost_reprice_c.py` → `cost_regimes.py`; `sb_mechanism.py` | `ff/`, `ff_water/`, `ff_summary_uncorrected.json` (fleet and coincident blocks), `ff_sites_*.csv`, `de/`, `cmip/cmip_d_summary.json`, `cost/`, `sb/sb5_mechanism.json` |
| Replication of the design menu on the data of record (Note S2) | `ff_menu.py --tag cds --years 1990-2025 --weather-dir D:/_caches/era5/cds/weather_hub_cds` → `ff_aggregate.py --tag cds --climates historical` → `ff_cds_compare.py` | `ff/cds/`, `ff_summary_cds.json`, `ff_sites_cds.csv`, `ff_cds_compare.json`, `ff_cds_site_designs.json` |
| Out-of-sample transfer and drift-aware sizing (Figure 6A; Note S11) | `oos_designs.py` → `oos_b.py`, `oos_c.py` | `oos/`, `oos_b/`, `oos_c/`, `oos_*summary.json` |
| Sizing procedures for climatic drift (Figure 6B; Note S12) | `experiments/phase0/r2_dro.py` → `r2_aggregate.py`; `r2r_precompute.py` → `r2r_analysis.py` → `r2r_b_analysis.py` → `r2r_c_analysis.py`; `r2s_structure.py`, `r2s_certificates.py`; `r2f_cmip_2060.py` → `r2f_extract_1960.py` → `r2f_precompute.py --part test`, `--part train` → `r2f_opyear.py` → `r2f_stage1.py` → `r2f_precompute.py --part shift`, `--part bmon` → `r2f_check_stage2_inputs.py` → `r2f_stage2.py` → `r2f_frontier.py` | `experiments/phase0/out/r2/`, `r2r/`, `r2r*_summary.json`, `r2s_summary.json`, `r2s_certificates.json`, `r2f/` |
| Figures and supplemental tables | `paper/figs/src/fig2_boundary.py` … `fig6_drift.py` (Figure 1 is drawn from `paper/figs/src/fig1.tex`); `paper/si/make_si.py` | figure files and `paper/si/tables/` |

## Tests

`python -m pytest tests` runs the unit tests of the thermal chain. `experiments/phase1/r2f_selftest.py` checks the
robust-sizing optimizers against linear programming and brute force.
