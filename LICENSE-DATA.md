# Data licence and third-party terms

## Derived data and result files

The derived tables and result files in this repository (`data/sites/`, `experiments/*/out/` and the other files produced by
the scripts) are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. Please cite the paper and this archive (see `CITATION.cff`).

## Third-party inputs

| Input | Source | Terms |
|---|---|---|
| ERA5 hourly single-level time series (`data/era5_cds_tables/`, and the tables derived from them) | Copernicus Climate Change Service (C3S) Climate Data Store, dataset `reanalysis-era5-single-levels-timeseries` | Licence to use Copernicus products. Contains modified Copernicus Climate Change Service information (2026). Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains. |
| Earlier ERA5 point tables (`data/weather*/`) | ERA5 via the Open-Meteo historical weather archive | CC BY 4.0 (Open-Meteo); Copernicus terms as above |
| AI data-centre frame (`data/epoch/`, and the site and capacity tables derived from it) | Epoch AI, *Frontier Data Centers* hub, snapshot of 24 September 2026 | CC BY 4.0 |
| CMIP6 monthly near-surface air temperature (changes in `experiments/phase1/out/cmip/`, `experiments/phase1/out/r2f/`) | CMIP6 via the Pangeo CMIP6 cloud archive | Each modelling centre's terms of use (CC BY 4.0 for most models). We acknowledge the World Climate Research Programme and the climate modelling groups. |
| Weather-station records (fetched by the scripts; station lists in `data/sites/`) | NOAA Integrated Surface Database (ISD-Lite) and US Climate Reference Network | US Government works, public domain |
| Site coordinates geocoded from street addresses (`data/sites/`) | Nominatim | © OpenStreetMap contributors, Open Database Licence (ODbL) 1.0 |
| Map outlines used by the figure scripts (not included) | Natural Earth | Public domain |
| Cost sources (listed in `docs/cost_basis.md`; the documents themselves are not redistributed) | Public reports (EIA, FEMP, IRENA, CEC, ARPA-E, OSTI/NETL, PNNL, LBNL) | Their publishers' terms |
