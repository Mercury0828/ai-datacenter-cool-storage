# Cost basis (2026-09-26) — sourced vs assumed inputs, common year 2024 USD

The model's central values (`model/params.py`) were declared in Phase 0; this file
anchors each one to public installed-cost evidence where it exists, states the scope and currency year, and marks what remains an
assumption. Inflation to 2024 USD uses the US CPI-U annual averages (1999 166.6; 2018 251.1; 2021 271.0; 2022 292.7; 2024 313.7) —
a general-price proxy for equipment costs, stated as such. Source files are kept in `lit/cost_sources/` (public documents).

| Input (model unit) | Model central | Sourced evidence → 2024 USD in model units | Scope / notes | Status |
|---|---|---|---|---|
| Air-cooled trim chiller (k$/MW_th installed) | 300 | EIA (2023, Guidehouse), *Updated Buildings Sector Appliance and Equipment Costs and Efficiencies*, App. B p.130: air-cooled screw chillers, total installed cost (2022$): 300-ton $820 (typical) / $920 (high eff.) per ton; 100-ton $1,250 / $1,350 → **250 / 280 (300-ton), 381 / 411 (100-ton)** | installed unit; average life 20 yr | **sourced**; central within range; test 250 and 381 |
| Chiller maintenance (per year) | not in model | EIA (2023) same table: $20/ton-yr (300-ton), $50 (100-ton) (2022$) → 6.1 k$/MW-yr ≈ 2.4 % of installed cost (300-ton) | annual maintenance | **sourced**; used in the omitted-cost check |
| Equipment life / capital recovery | 15 yr at 8 % (CRF 0.1168) | EIA (2023): chillers 20–25 yr | discount rate is a financing assumption | life **sourced** (20 yr); discount **assumed**; test (6 %, 20 yr) CRF 0.0872 and (10 %, 15 yr) CRF 0.1315 |
| Chilled-water store (k$/MWh_th installed) | 35 | FEMP (2000), *Thermal Energy Storage for Space Cooling*, Federal Technology Alert DOE/EE-0241, p.15, citing EPRI (1992): installed cost = 616·TH^0.686 (15 °F ΔT) and 498·TH^0.686 (20 °F ΔT), TH in ton-hours (validity to 9,000 / 12,000); at our 10 K (18 °F) swing and 12,000 ton-h ≈ $28.6/ton-h (1999$) → **15.3**; IRENA (2020), *Innovation Outlook: Thermal Energy Storage*, Fig. 10: tank TES 0.1–35 USD/kWh (2018$) → **≤ 44** | campus stores (≈ 10⁵–10⁶ ton-h) exceed the EPRI validity range; economies of scale make the lower end more likely | **sourced range 15–44**; central (35) on the high side (conservative for storage routes); test 15 and 44 (flip sweep 10–100 already registered) |
| Cooling tower incl. plate HX and water treatment (k$/MW at kct = 1) | 60 | FEMP (2000) p.15, citing R.S. Means (1999): tower installed cost = $982·Thr^0.64 (Thr = heat-rejection tons, 60–1,000) → at 1,000 tons $81.7/ton (1999$) → **44** (tower only) | our value adds the plate HX and treatment | **sourced (tower part)**; test 44 and 100 |
| Electricity ($/MWh) | 80 | EIA (2025), *Electric Power Annual 2024*, Table 2.10: US industrial average 8.13 ¢/kWh (2024); state industrial 5.43 (NM) – 8.99 (VA); commercial 12.75 → **81.3 (54–90 industrial)** | retail average price | **sourced**; test 54 and 90 (flip sweep 0.5–2× already registered) |
| Water, delivered ($/m³) | 2.0 | PNNL (2023), *Water and Wastewater Annual Price Escalation Rates for Selected Cities*, PNNL-34006: industrial (8-inch meter) water rate US mean $3.86/kGal (2021$) → **1.18**; regional means $2.74–5.63/kGal → **0.84–1.72**; wastewater $3.12–11.90/kGal additional (blowdown discharge) | volumetric marginal price | **sourced**; central 2.0 ≈ water plus part of sewer; the registered water-price sweep (0–20×) covers 0–40 $/m³ |
| Dry cooler, reference module (k$/MW per module; 120 at k_max) | 80 | Closest public anchor: NETL (2018), NETL-PUB-22446, Exh. 3-14/3-15, §2.9 — closed-loop air-cooled heat-exchanger bays for auxiliary cooling (vendor budgetary quotes via Black & Veatch; June 2011 $, total plant cost incl. EPC, contingency, pumps and piping; ISO 15 °C design, approach not reported): $195–201/kW_th → **≈ 270–280 $/kW_th (2024 $, full scope)**, i.e. ≈ 2–3.5× the model per rated kW. Plausibility: a whole air-cooled screw chiller costs ≈ 250 k$/MW_th (EIA 2023) | module rated 1 MW at 15 K ITD (Thermokey standard conditions) | **assumed** (no like-for-like source); flip reported: the ban's $/m³ falls from $12–20 to $1.3–4.5 above ≈ $300–360 per kW of design heat (PR-COST-b) |
| Adiabatic pads (fraction of dry-cooler capex) | 0.25 | none found | — | **assumed**; test 0.10 and 0.50 |
| Differential O&M (dry coolers, pads, tanks, towers) | not in model | none found beyond chillers | — | **assumed** 2 %/yr of capex in the omitted-cost check |
| Pumping power differences between routes | not in model (equal flows) | none | — | **assumed**; bounded in PR-COST-c: ±5 kW per MW of design heat charged to towers vs dry coolers moves the tower switch between 2× and 4× the central dry-cooler price and leaves the central-price result at $17.7–18.0/m³ |

### Additions from PR-SB SB-6 (2026-09-27)

- **Plate heat exchanger of the dry route.** It is not in the model, and is used only to compare with SB-4's break-even premium.
  - *Derived scenario.* Halving the LMTD from 2 to 1 K doubles the area. At U = 4 kW m⁻² K⁻¹ in 1 MW modules, the area goes from
    125 to 250 m² per MW.
  - *Correlation.* Towler & Sinnott, *Chemical Engineering Design*, 2nd ed. (2013), Table 7.2, plate-and-frame:
    C = 1,600 + 210·A^0.95 (A in m², 1–500 m², 304 SS, purchased, January 2010 USD, CEPCI 532.9).
  - *Result.* +$19.2/kW_th (2010 USD), or **+$28.9/kW_th in May 2024 USD** (CEPCI 800.2, *Chemical Engineering*, September
    2024).
  - *Sensitivity.* U = 3–5 kW m⁻² K⁻¹ gives $23–38/kW_th. The support for U: a SWEP glycol–water selection of 3.63 kW m⁻² K⁻¹;
    Alfa Laval's water–water guidance of 6–7.5.
  - *Scope.* Purchased equipment; installation and hydraulic redesign are additional.
  - Status: **derived scenario** (a standard cost-engineering correlation plus stated assumptions).
- **Dry cooler.** No defensible installed price at HOTLOOP's condition (≈ 35 °C air / 39 °C leaving glycol) was found in public
  sources.
  - *Nearest published evidence.*
    - D'Antoni, Romeli & Fedrizzi, EuroSun 2014: a catalogue survey at €49–107 per kW rejected, at 25 °C air and 40 → 35 °C
      water (10 K approach). Equipment plus selected accessories; the upper trend includes hybrid units.
    - NETL 2018 (OSTI 1529314): ≈ $195–201/kW_th in June 2011 USD, 15 °C design air, a component-cooling account (see the table
      above).
  - *Status.* These do not form a range at our condition. The central value **stays assumed**, and the paper presents the ban's
    economics as a cost-condition map (the regimes against the dry-cooler price multiplier, 0.5–10×, in PR-COST-b / -c).
  - *No re-rating.* An earlier sentence re-rated the catalogue cost to our approach ("≈ 2–2.5×"). It is withdrawn: it is not a
    matched engineering estimate (internal review strengthen-boundary R1). The dry-cooler price stays assumed, and C4's regimes show the
    consequences across 0.5–10×.

Currency and scope of all model results: 2024 USD, installed equipment of the heat-rejection plant (dry cooler, pads, store, trim
chiller or tower with its HX); CDU, piping and electrical infrastructure common to all routes are excluded (they do not differ
between the compared routes at equal flow).
