# cc-plugin-aicc

AWI ICON Coordinate Checker (AICC) — compliance_checker plugin for
[CMIP7](https://wcrp-cmip.org/cmip7/) coordinate verification on AWI and ICON
model output that uses unstructured horizontal grids.

## Overview

The plugin resolves the `branded_variable` global attribute against the CMIP7
CMOR tables, identifies which coordinates are required, and runs four
targeted checks:

| Check | What is verified |
|---|---|
| `check_grid` | Unstructured lat/lon auxiliary coordinates, vertices bounds |
| `check_vertical` | Generic vertical levels (alevel/alevhalf/olevel/olevhalf), formula_terms |
| `check_time` | Time axis, units, calendar, bounds / CF climatology |
| `check_coord` | All other coordinates: scalar, character scalar, multi-value numeric/character |
| `check_dimensions` | C-order dimension ordering of the data variable |

AWI model output uses `alternate_hybrid_sigma` / `alternate_hybrid_sigma_half`
for atmospheric levels; ICON uses `modified_sleve_model_level` /
`modified_sleve_half_level`. Both use `depth_coord` / `depth_coord_half` for
ocean levels. Detection is automatic via the `source_id` global attribute.

## Requirements

* Python ≥ 3.10
* [compliance-checker](https://github.com/ioos/compliance-checker) ≥ 5.1.2
* CMIP7 CMOR tables (JSON files)

## Installation

```bash
pip install -e .
```

## Usage

```bash
# point to CMIP7 tables via option
compliance-checker -t aicc --options tables:/path/to/cmip7-cmor-tables/tables myfile.nc

# or set the environment variable
export CMIP7_TABLES_PATH=/path/to/cmip7-cmor-tables/tables
compliance-checker -t aicc myfile.nc
```

The `tables` option (or `CMIP7_TABLES_PATH` environment variable) must point
to the directory containing the CMIP7 JSON tables
(`CMIP7_coordinate.json`, `CMIP7_grids.json`, `CMIP7_formula_terms.json`,
and the variable tables such as `CMIP7_atmos.json`).
