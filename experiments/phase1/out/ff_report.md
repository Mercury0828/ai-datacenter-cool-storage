### uncorrected (items 160, complete True)
- B_full_MW historical: water allowed Z1/Z2/Z3/Z∅ = 0.038/0.962/0.000/0.000; banned Z1/Z3/Z∅ = 0.561/0.439/0.000; ban split GW_th installed 2.94 = physical 0.74 + economic 2.19; water-allowed installed 0.00 (physical 0.00)
- B_full_MW stress_plus1.2K: water allowed Z1/Z2/Z3/Z∅ = 0.026/0.974/0.000/0.000; banned Z1/Z3/Z∅ = 0.405/0.595/0.000; ban split GW_th installed 4.54 = physical 1.57 + economic 2.96; water-allowed installed 0.00 (physical 0.00)
- B_2027_MW historical: water allowed Z1/Z2/Z3/Z∅ = 0.059/0.941/0.000/0.000; banned Z1/Z3/Z∅ = 0.616/0.384/0.000; ban split GW_th installed 1.46 = physical 0.27 + economic 1.19; water-allowed installed 0.00 (physical 0.00)
- B_2027_MW stress_plus1.2K: water allowed Z1/Z2/Z3/Z∅ = 0.041/0.959/0.000/0.000; banned Z1/Z3/Z∅ = 0.423/0.577/0.000; ban split GW_th installed 2.41 = physical 0.66 + economic 1.75; water-allowed installed 0.00 (physical 0.00)
- B_now_MW historical: water allowed Z1/Z2/Z3/Z∅ = 0.055/0.945/0.000/0.000; banned Z1/Z3/Z∅ = 0.649/0.351/0.000; ban split GW_th installed 0.64 = physical 0.17 + economic 0.48; water-allowed installed 0.00 (physical 0.00)
- B_now_MW stress_plus1.2K: water allowed Z1/Z2/Z3/Z∅ = 0.047/0.953/0.000/0.000; banned Z1/Z3/Z∅ = 0.390/0.610/0.000; ban split GW_th installed 1.20 = physical 0.33 + economic 0.88; water-allowed installed 0.00 (physical 0.00)
- B-full historical scale (GW_th chiller / GW_e peak / TWh/yr / GL/yr / GWh_th store / B$/yr):
  - least cost, water: 0.00 / 0.51 / 0.69 / 21.12 / 1.3 / 0.503 (covered 37295 MW; unmet in violating years 2.80 GWh)
  - least cost, water ban: 2.94 / 1.17 / 0.66 / 0.00 / 51.3 / 0.877 (covered 37295 MW; unmet in violating years 21.53 GWh)
  - compressor-free, water: 0.00 / 0.51 / 0.69 / 21.12 / 1.3 / 0.503 (covered 37295 MW; unmet in violating years 2.80 GWh)
  - compressor-free, water-free (feasible sites): 0.00 / 0.63 / 0.39 / 0.00 / 74.2 / 0.804 (covered 34290 MW; unmet in violating years 34.60 GWh)
  - water-free: compressor-free where feasible + C_phys elsewhere: 0.74 / 0.82 / 0.63 / 0.00 / 98.2 / 0.990 (covered 37295 MW; unmet in violating years 44.19 GWh)
  - tower counterfactual: 0.03 / 0.11 / 0.52 / 470.91 / 0.0 / 1.115 (covered 37295 MW; unmet in violating years 0.01 GWh)
  - eliminate compressors with water: 0 MW, Δcost 0.0 M$/yr, Δwater 0.00 GL/yr, Δelec 0.000 TWh/yr, Δpeak 0.000 GW_e
  - eliminate compressors under a ban: feasible 13384 MW (infeasible 3005), Δcost 56.1 M$/yr, Δstore 28.9 GWh_th, Δelec -0.026 TWh/yr
  - C_phys GW_th (ban / ≤1% / ≤10% / ≤100% tower water / peak ≤50% / free) at h = 0, 2, 4, 8: h0: 11.70/6.04/1.37/0.00/11.70/0.00; h2: 2.94/0.90/0.00/0.00/2.94/0.00; h4: 1.40/0.81/0.00/0.00/1.40/0.00; h8: 0.74/0.69/0.00/0.00/0.74/0.00
  - mixed mode: {'MW_feasible_C<=0.1': 35689.631374, 'MW_feasible_C<=0.2': 35689.631374, 'MW_mm01_cheaper_than_tower': 27154.631374}; near-ties MW: {'2%': 2442.0, '5%': 12329.691664}
- B-full stress_plus1.2K scale (GW_th chiller / GW_e peak / TWh/yr / GL/yr / GWh_th store / B$/yr):
  - least cost, water: 0.00 / 0.54 / 0.73 / 25.14 / 1.9 / 0.541 (covered 37295 MW; unmet in violating years 2.57 GWh)
  - least cost, water ban: 4.54 / 1.49 / 0.87 / 0.00 / 61.3 / 0.994 (covered 37295 MW; unmet in violating years 24.35 GWh)
  - compressor-free, water: 0.00 / 0.54 / 0.73 / 25.14 / 1.9 / 0.541 (covered 37295 MW; unmet in violating years 2.57 GWh)
  - compressor-free, water-free (feasible sites): 0.00 / 0.55 / 0.31 / 0.00 / 58.0 / 0.650 (covered 28195 MW; unmet in violating years 28.68 GWh)
  - water-free: compressor-free where feasible + C_phys elsewhere: 1.57 / 1.01 / 0.83 / 0.00 / 130.8 / 1.172 (covered 37295 MW; unmet in violating years 64.37 GWh)
  - tower counterfactual: 0.00 / 0.12 / 0.54 / 473.43 / 0.0 / 1.124 (covered 37295 MW; unmet in violating years 0.01 GWh)
  - eliminate compressors with water: 0 MW, Δcost 0.0 M$/yr, Δwater 0.00 GL/yr, Δelec 0.000 TWh/yr, Δpeak 0.000 GW_e
  - eliminate compressors under a ban: feasible 13096 MW (infeasible 9100), Δcost 14.9 M$/yr, Δstore 14.9 GWh_th, Δelec -0.014 TWh/yr
  - C_phys GW_th (ban / ≤1% / ≤10% / ≤100% tower water / peak ≤50% / free) at h = 0, 2, 4, 8: h0: 14.35/8.81/1.49/0.00/14.35/0.00; h2: 4.55/1.05/0.66/0.00/4.55/0.00; h4: 2.47/0.97/0.62/0.00/2.47/0.00; h8: 1.57/0.88/0.00/0.00/1.57/0.00
  - mixed mode: {'MW_feasible_C<=0.1': 33597.631374, 'MW_feasible_C<=0.2': 35689.631374, 'MW_mm01_cheaper_than_tower': 26745.631374}; near-ties MW: {'2%': 1487.0, '5%': 10337.0}
- coincident historical|lc: 0.291 GW_e (max 0.314) vs sum of site peaks 0.510; ratio 0.571
- coincident historical|ban: 0.744 GW_e (max 1.006) vs sum of site peaks 1.172; ratio 0.633
- coincident historical|cf: 0.291 GW_e (max 0.314) vs sum of site peaks 0.510; ratio 0.571
- coincident historical|cfdry: 0.436 GW_e (max 0.627) vs sum of site peaks 0.627; ratio 0.691
- coincident historical|tower: 0.096 GW_e (max 0.103) vs sum of site peaks 0.113; ratio 0.854
- coincident stress_plus1.2K|lc: 0.305 GW_e (max 0.340) vs sum of site peaks 0.542; ratio 0.564
- coincident stress_plus1.2K|ban: 0.985 GW_e (max 1.280) vs sum of site peaks 1.491; ratio 0.659
- coincident stress_plus1.2K|cf: 0.305 GW_e (max 0.340) vs sum of site peaks 0.542; ratio 0.564
- coincident stress_plus1.2K|cfdry: 0.383 GW_e (max 0.534) vs sum of site peaks 0.548; ratio 0.695
- coincident stress_plus1.2K|tower: 0.102 GW_e (max 0.110) vs sum of site peaks 0.121; ratio 0.845
- flips tank_kUSD_per_MWh (plausible bounds): historical|water_allowed: 436 MW; historical|water_banned: 19024 MW; stress_plus1.2K|water_allowed: 1925 MW; stress_plus1.2K|water_banned: 24175 MW
- flips water_mult (plausible bounds): historical|water_allowed: 1967 MW; historical|water_banned: 0 MW; stress_plus1.2K|water_allowed: 1967 MW; stress_plus1.2K|water_banned: 0 MW
- flips elec_mult (plausible bounds): historical|water_allowed: 436 MW; historical|water_banned: 0 MW; stress_plus1.2K|water_allowed: 0 MW; stress_plus1.2K|water_banned: 0 MW
### corrected (items 160, complete True)
- B_full_MW historical: water allowed Z1/Z2/Z3/Z∅ = 0.026/0.974/0.000/0.000; banned Z1/Z3/Z∅ = 0.440/0.560/0.000; ban split GW_th installed 3.80 = physical 0.84 + economic 2.96; water-allowed installed 0.00 (physical 0.00)
- B_full_MW stress_plus1.2K: water allowed Z1/Z2/Z3/Z∅ = 0.014/0.986/0.000/0.000; banned Z1/Z3/Z∅ = 0.343/0.657/0.000; ban split GW_th installed 5.63 = physical 1.82 + economic 3.81; water-allowed installed 0.00 (physical 0.00)
- B_2027_MW historical: water allowed Z1/Z2/Z3/Z∅ = 0.041/0.959/0.000/0.000; banned Z1/Z3/Z∅ = 0.463/0.537/0.000; ban split GW_th installed 1.94 = physical 0.31 + economic 1.63; water-allowed installed 0.00 (physical 0.00)
- B_2027_MW stress_plus1.2K: water allowed Z1/Z2/Z3/Z∅ = 0.021/0.979/0.000/0.000; banned Z1/Z3/Z∅ = 0.328/0.672/0.000; ban split GW_th installed 3.10 = physical 0.74 + economic 2.36; water-allowed installed 0.00 (physical 0.00)
- B_now_MW historical: water allowed Z1/Z2/Z3/Z∅ = 0.047/0.953/0.000/0.000; banned Z1/Z3/Z∅ = 0.423/0.577/0.000; ban split GW_th installed 0.94 = physical 0.18 + economic 0.76; water-allowed installed 0.00 (physical 0.00)
- B_now_MW stress_plus1.2K: water allowed Z1/Z2/Z3/Z∅ = 0.030/0.970/0.000/0.000; banned Z1/Z3/Z∅ = 0.359/0.641/0.000; ban split GW_th installed 1.60 = physical 0.37 + economic 1.22; water-allowed installed 0.00 (physical 0.00)
- B-full historical scale (GW_th chiller / GW_e peak / TWh/yr / GL/yr / GWh_th store / B$/yr):
  - least cost, water: 0.00 / 0.52 / 0.67 / 21.31 / 1.1 / 0.510 (covered 37295 MW; unmet in violating years 2.07 GWh)
  - least cost, water ban: 3.80 / 1.34 / 0.68 / 0.00 / 58.6 / 0.941 (covered 37295 MW; unmet in violating years 23.44 GWh)
  - compressor-free, water: 0.00 / 0.52 / 0.67 / 21.31 / 1.1 / 0.510 (covered 37295 MW; unmet in violating years 2.07 GWh)
  - compressor-free, water-free (feasible sites): 0.00 / 0.60 / 0.33 / 0.00 / 72.1 / 0.760 (covered 31848 MW; unmet in violating years 28.10 GWh)
  - water-free: compressor-free where feasible + C_phys elsewhere: 0.84 / 0.87 / 0.64 / 0.00 / 115.7 / 1.068 (covered 37295 MW; unmet in violating years 54.01 GWh)
  - tower counterfactual: 0.04 / 0.11 / 0.52 / 470.93 / 0.0 / 1.115 (covered 37295 MW; unmet in violating years 0.01 GWh)
  - eliminate compressors with water: 0 MW, Δcost 0.0 M$/yr, Δwater 0.00 GL/yr, Δelec 0.000 TWh/yr, Δpeak 0.000 GW_e
  - eliminate compressors under a ban: feasible 15443 MW (infeasible 5447), Δcost 33.4 M$/yr, Δstore 24.4 GWh_th, Δelec -0.024 TWh/yr
  - C_phys GW_th (ban / ≤1% / ≤10% / ≤100% tower water / peak ≤50% / free) at h = 0, 2, 4, 8: h0: 13.66/6.54/1.40/0.00/13.66/0.00; h2: 3.79/0.93/0.00/0.00/3.79/0.00; h4: 1.75/0.83/0.00/0.00/1.75/0.00; h8: 0.84/0.71/0.00/0.00/0.84/0.00
  - mixed mode: {'MW_feasible_C<=0.1': 35689.631374, 'MW_feasible_C<=0.2': 35689.631374, 'MW_mm01_cheaper_than_tower': 26745.631374}; near-ties MW: {'2%': 5517.0, '5%': 13397.021374}
- B-full stress_plus1.2K scale (GW_th chiller / GW_e peak / TWh/yr / GL/yr / GWh_th store / B$/yr):
  - least cost, water: 0.00 / 0.55 / 0.73 / 25.51 / 2.2 / 0.549 (covered 37295 MW; unmet in violating years 3.45 GWh)
  - least cost, water ban: 5.63 / 1.68 / 0.90 / 0.00 / 67.6 / 1.062 (covered 37295 MW; unmet in violating years 25.19 GWh)
  - compressor-free, water: 0.00 / 0.55 / 0.73 / 25.51 / 2.2 / 0.549 (covered 37295 MW; unmet in violating years 3.45 GWh)
  - compressor-free, water-free (feasible sites): 0.00 / 0.57 / 0.32 / 0.00 / 73.6 / 0.718 (covered 28195 MW; unmet in violating years 37.12 GWh)
  - water-free: compressor-free where feasible + C_phys elsewhere: 1.82 / 1.08 / 0.85 / 0.00 / 146.4 / 1.249 (covered 37295 MW; unmet in violating years 73.04 GWh)
  - tower counterfactual: 0.00 / 0.12 / 0.54 / 473.44 / 0.0 / 1.124 (covered 37295 MW; unmet in violating years 0.01 GWh)
  - eliminate compressors with water: 0 MW, Δcost 0.0 M$/yr, Δwater 0.00 GL/yr, Δelec 0.000 TWh/yr, Δpeak 0.000 GW_e
  - eliminate compressors under a ban: feasible 15396 MW (infeasible 9100), Δcost 27.2 M$/yr, Δstore 24.1 GWh_th, Δelec -0.027 TWh/yr
  - C_phys GW_th (ban / ≤1% / ≤10% / ≤100% tower water / peak ≤50% / free) at h = 0, 2, 4, 8: h0: 16.34/9.61/1.52/0.00/16.34/0.00; h2: 5.62/1.08/0.69/0.00/5.62/0.00; h4: 2.88/1.00/0.64/0.00/2.88/0.00; h8: 1.82/0.90/0.00/0.00/1.82/0.00
  - mixed mode: {'MW_feasible_C<=0.1': 31847.631374, 'MW_feasible_C<=0.2': 34289.631374000004, 'MW_mm01_cheaper_than_tower': 23633.631374}; near-ties MW: {'2%': 4263.0, '5%': 12343.0}
- coincident historical|lc: 0.289 GW_e (max 0.314) vs sum of site peaks 0.520; ratio 0.556
- coincident historical|ban: 0.838 GW_e (max 1.133) vs sum of site peaks 1.337; ratio 0.625
- coincident historical|cf: 0.289 GW_e (max 0.314) vs sum of site peaks 0.520; ratio 0.556
- coincident historical|cfdry: 0.420 GW_e (max 0.604) vs sum of site peaks 0.603; ratio 0.691
- coincident historical|tower: 0.097 GW_e (max 0.104) vs sum of site peaks 0.115; ratio 0.847
- coincident stress_plus1.2K|lc: 0.307 GW_e (max 0.347) vs sum of site peaks 0.551; ratio 0.558
- coincident stress_plus1.2K|ban: 1.099 GW_e (max 1.429) vs sum of site peaks 1.684; ratio 0.651
- coincident stress_plus1.2K|cf: 0.307 GW_e (max 0.347) vs sum of site peaks 0.551; ratio 0.558
- coincident stress_plus1.2K|cfdry: 0.412 GW_e (max 0.537) vs sum of site peaks 0.570; ratio 0.718
- coincident stress_plus1.2K|tower: 0.103 GW_e (max 0.111) vs sum of site peaks 0.123; ratio 0.840
- flips tank_kUSD_per_MWh (plausible bounds): historical|water_allowed: 473 MW; historical|water_banned: 25472 MW; stress_plus1.2K|water_allowed: 473 MW; stress_plus1.2K|water_banned: 24314 MW
- flips water_mult (plausible bounds): historical|water_allowed: 1967 MW; historical|water_banned: 0 MW; stress_plus1.2K|water_allowed: 1424 MW; stress_plus1.2K|water_banned: 0 MW
- flips elec_mult (plausible bounds): historical|water_allowed: 473 MW; historical|water_banned: 0 MW; stress_plus1.2K|water_allowed: 473 MW; stress_plus1.2K|water_banned: 0 MW
### Decision-rule criteria (PR-FF)
```
{
 "1_material_B_full_hist_GW_th": 2.193354611317893,
 "1_met": true,
 "2_robust_values_GW_th": {
  "uncorrected|B_full_MW|historical": 2.193354611317893,
  "uncorrected|B_full_MW|stress_plus1.2K": 2.9647878640951224,
  "uncorrected|B_2027_MW|historical": 1.1898570722612223,
  "uncorrected|B_2027_MW|stress_plus1.2K": 1.752193945576884,
  "corrected|B_full_MW|historical": 2.960185684306797,
  "corrected|B_full_MW|stress_plus1.2K": 3.8134376220327786,
  "corrected|B_2027_MW|historical": 1.6289765166216275,
  "corrected|B_2027_MW|stress_plus1.2K": 2.355019856427252
 },
 "2_met": true,
 "3_moves_MW": {
  "uncorrected|tank_kUSD_per_MWh|historical|water_allowed": 436.0,
  "uncorrected|tank_kUSD_per_MWh|historical|water_banned": 19024.329709999998,
  "uncorrected|tank_kUSD_per_MWh|stress_plus1.2K|water_allowed": 1925.0,
  "uncorrected|tank_kUSD_per_MWh|stress_plus1.2K|water_banned": 24174.72971,
  "uncorrected|water_mult|historical|water_allowed": 1966.691664,
  "uncorrected|water_mult|historical|water_banned": 0.0,
  "uncorrected|water_mult|stress_plus1.2K|water_allowed": 1966.691664,
  "uncorrected|water_mult|stress_plus1.2K|water_banned": 0.0,
  "uncorrected|elec_mult|historical|water_allowed": 436.0,
  "uncorrected|elec_mult|historical|water_banned": 0.0,
  "uncorrected|elec_mult|stress_plus1.2K|water_allowed": 0.0,
  "uncorrected|elec_mult|stress_plus1.2K|water_banned": 0.0,
  "corrected|tank_kUSD_per_MWh|historical|water_allowed": 473.0,
  "corrected|tank_kUSD_per_MWh|historical|water_banned": 25471.72971,
  "corrected|tank_kUSD_per_MWh|stress_plus1.2K|water_allowed": 473.0,
  "corrected|tank_kUSD_per_MWh|stress_plus1.2K|water_banned": 24313.72971,
  "corrected|water_mult|historical|water_allowed": 1966.691664,
  "corrected|water_mult|historical|water_banned": 0.0,
  "corrected|water_mult|stress_plus1.2K|water_allowed": 1423.691664,
  "corrected|water_mult|stress_plus1.2K|water_banned": 0.0,
  "corrected|elec_mult|historical|water_allowed": 473.0,
  "corrected|elec_mult|historical|water_banned": 0.0,
  "corrected|elec_mult|stress_plus1.2K|water_allowed": 473.0,
  "corrected|elec_mult|stress_plus1.2K|water_banned": 0.0
 },
 "3_met": true,
 "ALL_MET": true
}
```