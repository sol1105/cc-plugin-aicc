"""
config.py — Model configuration and resolution helpers for AICC.

DEFAULT_CONFIG is the public entry point: extend or replace it (or pass
a custom dict / path to a JSON file via the 'model_config' checker option)
to support additional modelling systems.
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
# Per-model configuration
# ---------------------------------------------------------------------------
# Structure: source_id_substring -> {"vertical": {...}, "horizontal": {...}}
#
# vertical:   generic_level_id  -> CMIP7_coordinate.json axis_entry key
# horizontal: grid_label        -> registered grid type
#                                (currently "unstructured" or "rectilinear")
#             "default"         -> fallback when no exact grid_label match
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
        "horizontal": {
            "default": "unstructured",
            "g132": "unstructured",
            "g130": "unstructured",
            "g129": "rectilinear",
            "g122": "unstructured",
            "g113": "rectilinear",
        },
    },
    "ICON-XPP": {
        "vertical": {
            "alevel": "modified_sleve_model_level",
            "alevhalf": "modified_sleve_half_level",
            "olevel": "depth_coord",
            "olevhalf": "depth_coord_half",
        },
        "horizontal": {
            "default": "unstructured",
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


def resolve_grid_type(horizontal_config: dict, grid_label: str) -> tuple:
    """Return (grid_type, is_known) for the given grid_label.

    is_known is False only when horizontal_config is non-empty, the
    grid_label is absent, and there is no 'default' key — the caller
    should then report an unknown grid_label issue.
    """
    if not horizontal_config:
        return "unstructured", True          # no config → silent default
    if grid_label and grid_label in horizontal_config:
        return horizontal_config[grid_label], True
    if "default" in horizontal_config:
        return horizontal_config["default"], True
    return "unstructured", False             # config present but label unknown


def load_model_config(option_value) -> dict:
    """Load a model config from a dict, a JSON file path, or return DEFAULT_CONFIG."""
    if option_value is None:
        return DEFAULT_CONFIG
    if isinstance(option_value, dict):
        return option_value
    with open(option_value) as fh:
        return json.load(fh)
