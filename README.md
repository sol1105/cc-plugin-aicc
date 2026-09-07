# cc-plugin-aicc

AWI-ESM and ICON-XPP Coordinate Checker (AICC) — compliance_checker plugin for
[CMIP7](https://wcrp-cmip.org/cmip7/) coordinate verification on AWI-ESM and
ICON-XPP model output using configured horizontal grids and vertical coordinate
systems. Other modeling systems could be easily configured.

This plugin has been developed with heavy AI support as an intermediate solution
until `cc-plugin-wcrp` provides general support for coordinate checks.

## Overview

The plugin resolves the `branded_variable` global attribute, identifies which
coordinates are required, and runs several targeted checks. By default the
metadata comes from the CMIP7 CMOR JSON tables. Passing `-O aicc:esgvoc`
instead uses ESGVoc's `known_branded_variable`, `data_coordinate`,
`model_level_coordinate`, `formula_term`, `grid_variable`, and `grid_axis`
descriptors.

| Check | What is verified |
|---|---|
| `check_branded_variable` | Identification of the requested variable in the selected CMIP7 metadata source |
| `check_grid` | Rectilinear, unstructured, or curvilinear latitude/longitude coordinates, vertices, and explicit grid axes |
| `check_vertical` | Generic vertical levels (alevel/alevhalf/olevel/olevhalf), formula_terms |
| `check_vertical_direction` | Stored and formula-derived direction of generic vertical levels |
| `check_time` | Time axis, units, calendar, bounds / CF climatology |
| `check_coord` | All other coordinates: scalar, character scalar, multi-value numeric/character |
| `check_coordinate_direction` | Strict monotonicity of numeric 1-D coordinates, prescribed stored direction, and physical direction of pressure, height, and depth |
| `check_dimensions` | C-order dimension ordering of the data variable |
| `check_coordinates_attribute` | No unexpected entries in the data variable's `coordinates` attribute |
| `check_quantization` | CF-1.12 lossy quantization metadata and precision parameters |

Further models can be configured through the `vertical_config` checker option or by
extending the vertical defaults in `config.py`.

Currently, AWI-ESM is configured to verify `alternate_hybrid_sigma` /
`alternate_hybrid_sigma_half` for atmospheric levels; ICON-XPP is configured to
verify `modified_sleve_model_level` / `modified_sleve_half_level`. Both use
`depth_coord` / `depth_coord_half` for ocean levels. Detection is automatic via
the `source_id` global attribute.

CMIP7 `grid_label` values are registered globally rather than per model. All
currently registered labels from `g100` through `g236` are classified as
`"rectilinear"`, `"unstructured"`, or `"curvilinear"`. Coordinate validation is
implemented for all three topologies. Curvilinear grids may use rotated
`rlat`/`rlon`, projected metre or angular `x`/`y`, explicit index axes, or bare
implicit-index dimensions. The registry can be replaced through the `grid_config`
checker option or extended in `config.py`.

## Requirements

* Python ≥ 3.10
* [compliance-checker](https://github.com/ioos/compliance-checker) ≥ 6.1.0
* CMIP7 CMOR tables (JSON files), or ESGVoc >= 5.1.0 with the latest stable
  `universe` and `cmip7` vocabulary database snapshots when using
  `-O aicc:esgvoc`

## Installation

- Directly from GitHub:

```bash
python -m pip install "cc-plugin-aicc @ git+https://github.com/sol1105/cc-plugin-aicc.git@main"
```

- From a local source checkout:

```bash
python -m pip install -e .
```

ESGVoc is not installed as an AICC dependency. To use the ESGVoc metadata path,
install ESGVoc separately at version 5.1.0 or newer. ESGVoc 5.1.0 is the first
release containing the required coordinate descriptor Pydantic models. The
ESGVoc package and its versioned vocabulary database snapshots are installed
and updated separately. Install and activate the latest stable snapshots for
both Universe and CMIP7:

```bash
esgvoc use universe@latest
esgvoc use cmip7@latest
```

For an existing installation, check for and activate newer stable snapshots
with `esgvoc update --check` and `esgvoc update`.

## Usage

### Basic invocation

Use Compliance Checker for direct checks of individual files. Use
[ESGF-QA](https://github.com/ESGF/esgf-qa), invoked as `esgqa`, for multi-file
and simulation-level checking. All AICC checker options shown below apply to
both tools; the examples use Compliance Checker unless an `esgqa` command is
shown explicitly.

```bash
# check one file directly with Compliance Checker
compliance-checker -t aicc -c strict \
  --option aicc:tables:/path/to/cmip7-cmor-tables/tables \
  myfile.nc

# check a simulation directory with ESGF-QA
esgqa -t aicc \
  -O aicc:tables:/path/to/cmip7-cmor-tables/tables \
  /path/to/simulation

# or set the environment variable
export CMIP7_TABLES_PATH=/path/to/cmip7-cmor-tables/tables
compliance-checker -t aicc -c strict myfile.nc
```

The `tables` option (or `CMIP7_TABLES_PATH` environment variable) must point
to the directory containing the CMIP7 JSON tables
(`CMIP7_coordinate.json`, `CMIP7_grids.json`, `CMIP7_formula_terms.json`,
and the variable tables such as `CMIP7_atmos.json`).

### ESGVoc invocation

Use `-O aicc:esgvoc` to resolve the branded variable and all coordinate
definitions from the active ESGVoc database instead of reading CMOR JSON files:

```bash
compliance-checker -t aicc -c strict -O aicc:esgvoc myfile.nc
```

ESGVoc must be version 5.1.0 or newer, and the latest stable `universe` and
`cmip7` vocabulary database snapshots must both be installed and active. AICC
checks the installed package version and verifies that the required coordinate
collections can be read, but it does not currently determine whether the active
database snapshots are the latest releases. The `tables` checker option and
`CMIP7_TABLES_PATH` are ignored in this mode. Grid topology and the choice of a
concrete model-level coordinate remain model/file configuration, so
`grid_config` and `vertical_config` work identically with both metadata sources.

Bounds-like variables are checked structurally using the variable named by the
parent coordinate attribute. The trailing dimension names `bnds` for bounds and
`vertices` for horizontal vertices are recommendations reported at medium
severity. The recommended climatology bounds variable name is
`climatology_bnds`, also at medium severity.

For example, with both optional configuration files:

```bash
compliance-checker -t aicc -c strict \
  -O aicc:esgvoc \
  -O aicc:grid_config:/path/to/grid_config.json \
  -O aicc:vertical_config:/path/to/vertical_config.json \
  myfile.nc
```

### Grid configuration

The optional `grid_config` file maps each permitted `grid_label` global
attribute to the horizontal grid topology that AICC should validate. Supported
values are `"rectilinear"`, `"unstructured"`, and `"curvilinear"`.

Example `grid_config.json`:

```json
{
  "g122": "curvilinear",
  "g456": "rectilinear",
  "g567": "unstructured"
}
```

A custom file replaces the built-in registry; it does not extend it. It must
therefore contain every `grid_label` that should be accepted during the run.

### Vertical configuration

The optional vertical model configuration maps generic CMIP7 level IDs to
concrete model-level coordinate IDs. These are entries in
`CMIP7_coordinate.json` on the default path and `model_level_coordinate` terms
on the ESGVoc path. Its top-level keys are matched as substrings of the file's
`source_id` global attribute. If several keys match, the longest and therefore
most specific key is selected.

Example `vertical_config.json`:

```json
{
  "MY-MODEL": {
    "vertical": {
      "alevel": "alternate_hybrid_sigma",
      "alevhalf": "alternate_hybrid_sigma_half",
      "olevel": "depth_coord",
      "olevhalf": "depth_coord_half"
    }
  },
  "MY-MODEL-SPECIAL": {
    "vertical": {
      "alevel": "modified_sleve_model_level",
      "alevhalf": "modified_sleve_half_level",
      "olevel": "depth_coord",
      "olevhalf": "depth_coord_half"
    }
  }
}
```

For example, `source_id="MY-MODEL-SPECIAL-1"` selects the
`MY-MODEL-SPECIAL` entry. The inner values must be coordinate IDs present in
the selected metadata source. A bare mapping containing only `alevel`,
`alevhalf`, `olevel`, and `olevhalf` is not sufficient because it provides no
`source_id` match.

### Invocation with all configuration files

```bash
compliance-checker -t aicc -c strict \
  --option aicc:tables:/path/to/cmip7-cmor-tables/tables \
  --option aicc:grid_config:/path/to/grid_config.json \
  --option aicc:vertical_config:/path/to/vertical_config.json \
  myfile.nc
```

The short `-O` form is equivalent:

```bash
compliance-checker -t aicc -c strict \
  -O aicc:tables:/path/to/cmip7-cmor-tables/tables \
  -O aicc:grid_config:/path/to/grid_config.json \
  -O aicc:vertical_config:/path/to/vertical_config.json \
  myfile.nc
```

To run AICC together with another installed checker, repeat `--test`. For
example:

```bash
compliance-checker -t cmip7 -t aicc -c strict \
  -O aicc:tables:/path/to/cmip7-cmor-tables/tables \
  -O aicc:grid_config:/path/to/grid_config.json \
  -O aicc:vertical_config:/path/to/vertical_config.json \
  myfile.nc
```
