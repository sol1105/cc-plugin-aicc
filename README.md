# cc-plugin-aicc

AWI-ESM and ICON-XPP Coordinate Checker (AICC) — compliance_checker plugin for
[CMIP7](https://wcrp-cmip.org/cmip7/) coordinate verification on AWI-ESM and
ICON-XPP model output using configured horizontal grids and vertical coordinate
systems.

This plugin has been developed with heavy AI support as an intermediate solution
until `cc-plugin-wcrp` provides general support for coordinate checks.

## Overview

The plugin resolves the `branded_variable` global attribute against the CMIP7
CMOR tables, identifies which coordinates are required, and runs several
targeted checks:

| Check | What is verified |
|---|---|
| `check_grid` | Rectilinear or unstructured latitude/longitude coordinates and bounds |
| `check_vertical` | Generic vertical levels (alevel/alevhalf/olevel/olevhalf), formula_terms |
| `check_time` | Time axis, units, calendar, bounds / CF climatology |
| `check_coord` | All other coordinates: scalar, character scalar, multi-value numeric/character |
| `check_dimensions` | C-order dimension ordering of the data variable |
| `check_coordinates_attribute` | No unexpected entries in the data variable's `coordinates` attribute |
| `check_quantization` | CF-1.12 lossy quantization metadata and precision parameters |

Further models can be configured through the `model_config` checker option or by
extending the vertical defaults in `config.py`.

Currently, AWI-ESM is configured to verify `alternate_hybrid_sigma` /
`alternate_hybrid_sigma_half` for atmospheric levels; ICON-XPP is configured to
verify `modified_sleve_model_level` / `modified_sleve_half_level`. Both use
`depth_coord` / `depth_coord_half` for ocean levels. Detection is automatic via
the `source_id` global attribute.

CMIP7 `grid_label` values are registered globally rather than per model. All
currently registered labels from `g100` through `g236` are classified as
`"rectilinear"`, `"unstructured"`, or `"curvilinear"`. Coordinate validation is
currently implemented for rectilinear and unstructured grids; curvilinear labels
are recognized and reported as not yet implemented. The registry can be replaced
through the `grid_config` checker option or extended in `config.py`.

## Requirements

* Python ≥ 3.10
* [compliance-checker](https://github.com/ioos/compliance-checker) ≥ 6.1.0
* CMIP7 CMOR tables (JSON files)

## Installation

```bash
pip install -e .
```

## Usage

```bash
# point to CMIP7 tables via option
compliance-checker -t aicc -c strict --options tables:/path/to/cmip7-cmor-tables/tables myfile.nc

# or set the environment variable
export CMIP7_TABLES_PATH=/path/to/cmip7-cmor-tables/tables
compliance-checker -t aicc -c strict myfile.nc
```

The `tables` option (or `CMIP7_TABLES_PATH` environment variable) must point
to the directory containing the CMIP7 JSON tables
(`CMIP7_coordinate.json`, `CMIP7_grids.json`, `CMIP7_formula_terms.json`,
and the variable tables such as `CMIP7_atmos.json`).
