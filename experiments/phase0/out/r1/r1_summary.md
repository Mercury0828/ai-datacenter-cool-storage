# R1 results (items 128/128)

Capacity shares, basis B-full (MC 10-90 % in brackets):

| climate | variant | Z1 | Z2 | Z3 | Z∅ | F_nc | F_dry |
|---|---|---|---|---|---|---|---|
| current | headline | 0.00 [0.00–0.00] | 0.88 [0.85–0.90] | 0.12 [0.10–0.15] | 0.00 [0.00–0.00] | 1.00 [1.00–1.00] | 0.85 [0.82–0.88] |
| current | hybrid | 0.00 [0.00–0.00] | 0.06 [0.04–0.08] | 0.94 [0.92–0.96] | 0.00 [0.00–0.00] | 0.16 [0.13–0.20] | 0.00 [0.00–0.00] |
| current | t40 | 0.00 [0.00–0.00] | 0.11 [0.09–0.14] | 0.89 [0.86–0.91] | 0.00 [0.00–0.00] | 1.00 [1.00–1.00] | 0.24 [0.20–0.27] |
| current | water_constrained | 0.07 [0.05–0.09] | 0.00 [0.00–0.00] | 0.93 [0.91–0.95] | 0.00 [0.00–0.00] | 0.85 [0.82–0.88] | 0.85 [0.82–0.88] |
| 2040 | headline | 0.00 [0.00–0.00] | 0.58 [0.53–0.63] | 0.42 [0.37–0.47] | 0.00 [0.00–0.00] | 1.00 [1.00–1.00] | 0.71 [0.66–0.75] |
| 2040 | hybrid | 0.00 [0.00–0.00] | 0.10 [0.07–0.13] | 0.90 [0.87–0.93] | 0.00 [0.00–0.00] | 0.16 [0.13–0.20] | 0.00 [0.00–0.00] |
| 2040 | t40 | 0.00 [0.00–0.00] | 0.11 [0.08–0.14] | 0.89 [0.86–0.92] | 0.00 [0.00–0.00] | 1.00 [1.00–1.00] | 0.06 [0.04–0.08] |
| 2040 | water_constrained | 0.12 [0.09–0.15] | 0.00 [0.00–0.00] | 0.88 [0.85–0.91] | 0.00 [0.00–0.00] | 0.71 [0.66–0.75] | 0.71 [0.66–0.75] |

**Test A-phys:** {'current': {'F_nc_feasible': np.float64(1.0), 'F_nc_infeasible': np.float64(0.0), 'F_dry': np.float64(0.8533423514823582), 'pass': False}, '2040': {'F_nc_feasible': np.float64(1.0), 'F_nc_infeasible': np.float64(0.0), 'F_dry': np.float64(0.707191973847368), 'pass': False}}

**Test A-econ:** {'current': {'Z1+Z2': np.float64(0.878536805320708), 'Z3+Zempty': np.float64(0.12146319467929208), 'pass': True}, '2040': {'Z1+Z2': np.float64(0.5785142599481456), 'Z3+Zempty': np.float64(0.4214857400518543), 'pass': True}}

**Test B:** {"current|headline": {"1-max_z": 0.12146319467929201, "argmax": "Z2", "pass": true}, "current|hybrid": {"1-max_z": 0.056701611994138035, "argmax": "Z3", "pass": false}, "current|t40": {"1-max_z": 0.1132904971254649, "argmax": "Z3", "pass": true}, "current|water_constrained": {"1-max_z": 0.07203246533648966, "argmax": "Z3", "pass": false}, "2040|headline": {"1-max_z": 0.42148574005185435, "argmax": "Z2", "pass": true}, "2040|hybrid": {"1-max_z": 0.09863600495998182, "argmax": "Z3", "pass": false}, "2040|t40": {"1-max_z": 0.10680870251380892, "argmax": "Z3", "pass": true}, "2040|water_constrained": {"1-max_z": 0.11667230300980713, "argmax": "Z3", "pass": true}}

**Outcome:** (ii-a) A-phys fails, A-econ or Test B passes -> alternative contribution; ladder; owner at gate (a)

**F_dry vs storage bound (headline):** {'current|tank<=0.0h': 0.008172697553827078, 'current|tank<=2.0h': 0.47221282831698796, 'current|tank<=4.0h': 0.6504903618532296, 'current|tank<=8.0h': 0.8533423514823583, '2040|tank<=0.0h': 0.008172697553827078, '2040|tank<=2.0h': 0.47221282831698796, '2040|tank<=4.0h': 0.6504903618532296, '2040|tank<=8.0h': 0.7071919738473679}

**Checks:** {'lulea_current_headline': {'F_dry': True, 'F_nc': True, 'zone': 'Z2', 'cost': 7797.449083607208}, 'goodyear_current_water_constrained': {'F_dry': False, 'F_nc': False, 'zone': 'Z3', 'cost': 49376.16880807427}, 'goodyear_2040_water_constrained': {'F_dry': False, 'F_nc': False, 'zone': 'Z3', 'cost': 55845.30476669996}}

**Scale (15-site sample, B-full):** {
 "current|headline": {
  "sample_GW": 17.742,
  "water_GL_per_yr": 11.41387940045032,
  "cooling_TWh_per_yr": 0.31085413081720403,
  "peak_cooling_MW": 254.7738800976347,
  "trim_chiller_MW": 18.285389555668075
 },
 "current|water_constrained": {
  "sample_GW": 17.742,
  "water_GL_per_yr": 0.0,
  "cooling_TWh_per_yr": 0.4054618451500878,
  "peak_cooling_MW": 644.5215271753697,
  "trim_chiller_MW": 2350.9427118791855
 },
 "2040|headline": {
  "sample_GW": 17.742,
  "water_GL_per_yr": 13.227109312713196,
  "cooling_TWh_per_yr": 0.32779478984832927,
  "peak_cooling_MW": 272.8395926315059,
  "trim_chiller_MW": 153.4686463726345
 },
 "2040|water_constrained": {
  "sample_GW": 17.742,
  "water_GL_per_yr": 0.0,
  "cooling_TWh_per_yr": 0.522399896385698,
  "peak_cooling_MW": 803.705784437501,
  "trim_chiller_MW": 3243.9568399367836
 }
}