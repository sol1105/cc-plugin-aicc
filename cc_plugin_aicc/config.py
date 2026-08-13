"""
config.py — Model configuration and resolution helpers for AICC.

DEFAULT_CONFIG and DEFAULT_GRID_CONFIG are the public entry points. Extend or
replace them (or pass a custom dict / JSON path through the checker options) to
support additional modelling systems and grid labels.
"""

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths and domain constants
# ---------------------------------------------------------------------------

# Override via CMIP7_TABLES_PATH env var or the 'tables' checker option.
DEFAULT_TABLES_PATH = os.environ.get(
    "CMIP7_TABLES_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "cmip7-cmor-tables" / "tables"),
)

VERTICAL_GENERIC_IDS = frozenset({"alevel", "alevhalf", "olevel", "olevhalf"})
HORIZONTAL_DIM_IDS = frozenset({"latitude", "longitude"})

# First word of the realm global attribute → CMIP7 variable-table name fragment
REALM_TO_TABLE = {
    "atmos": "atmos",
    "land": "land",
    "ocean": "ocean",
    "seaIce": "seaIce",
    "landIce": "landIce",
    "aerosol": "aerosol",
    "atmosChem": "atmosChem",
    "ocnBgchem": "ocnBgchem",
}

# ---------------------------------------------------------------------------
# Global grid-label configuration
# ---------------------------------------------------------------------------
# Grid labels are registered globally by CMIP7 and do not depend on source_id.
# Labels are reduced to the topology understood by the coordinate checker.

RECTILINEAR_GRID_LABELS = frozenset(
    {
        "g100", "g101", "g105", "g106", "g108", "g109", "g110", "g111",
        "g113", "g114", "g115", "g120", "g121", "g123", "g129", "g131",
        "g134", "g137", "g138", "g139", "g140", "g150", "g151", "g152",
        "g158", "g159", "g163", "g167", "g179", "g180", "g183", "g189",
        "g190", "g198", "g200", "g201", "g207", "g208", "g209", "g210",
        "g214", "g215", "g222", "g223", "g224", "g225", "g229", "g230",
    }
)

UNSTRUCTURED_GRID_LABELS = frozenset(
    {
        "g117", "g122", "g130", "g132", "g136", "g141", "g142", "g176",
        "g177", "g178", "g187", "g188", "g235", "g236",
    }
)

CURVILINEAR_GRID_LABELS = frozenset(
    f"g{number}"
    for number in range(100, 237)
    if f"g{number}" not in RECTILINEAR_GRID_LABELS | UNSTRUCTURED_GRID_LABELS
)

DEFAULT_GRID_CONFIG = {
    **{label: "rectilinear" for label in RECTILINEAR_GRID_LABELS},
    **{label: "unstructured" for label in UNSTRUCTURED_GRID_LABELS},
    **{label: "curvilinear" for label in CURVILINEAR_GRID_LABELS},
}

# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------
# Structure: source_id_substring -> {"vertical": {...}}
#
# vertical:   generic_level_id  -> CMIP7_coordinate.json axis_entry key
#
# Longer (more-specific) source_id keys take precedence over shorter ones,
# so "AWI-ESM" beats "AWI" for source_id "AWI-ESM-2-3-Veg".
#
# Pass a custom dict or path to a JSON file via the 'model_config' option.

DEFAULT_CONFIG = {
    "AWI-ESM": {
        "vertical": {
            "alevel": "alternate_hybrid_sigma",
            "alevhalf": "alternate_hybrid_sigma_half",
            "olevel": "depth_coord",
            "olevhalf": "depth_coord_half",
        },
    },
    "ICON-XPP": {
        "vertical": {
            "alevel": "modified_sleve_model_level",
            "alevhalf": "modified_sleve_half_level",
            "olevel": "depth_coord",
            "olevhalf": "depth_coord_half",
        },
    },
}


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_model_config(source_id: str, config: dict) -> tuple:
    """Return (matched_key, model_dict) for the most-specific config entry.

    All config keys that are substrings of source_id are candidates;
    the longest key wins (most specific). Returns (None, None) if no match.
    """
    matches = [(k, v) for k, v in config.items() if k in source_id]
    if not matches:
        return None, None
    return max(matches, key=lambda x: len(x[0]))


def resolve_grid_type(grid_label: str, grid_config: dict) -> tuple:
    """Return ``(grid_type, is_known)`` from the global grid-label registry."""
    if grid_label and grid_label in grid_config:
        return grid_config[grid_label], True
    return None, False


def load_model_config(option_value) -> dict:
    """Load a model config from a dict, a JSON file path, or return DEFAULT_CONFIG."""
    if option_value is None:
        return DEFAULT_CONFIG
    if isinstance(option_value, dict):
        return option_value
    with open(option_value) as fh:
        return json.load(fh)


def load_grid_config(option_value) -> dict:
    """Load a global grid config from a dict, JSON path, or use the defaults."""
    if option_value is None:
        return DEFAULT_GRID_CONFIG
    if isinstance(option_value, dict):
        return option_value
    with open(option_value) as fh:
        return json.load(fh)
