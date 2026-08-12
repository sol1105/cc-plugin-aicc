"""
utils.py — Pure utility functions for AICC coordinate checks.

All functions here are side-effect-free and have no dependency on
compliance_checker's Result/TestCtx machinery.
"""

import re

import numpy as np
from compliance_checker.cf import util as cfutil


# Adapted from swarnaleem's attr() — github.com/swarnaleem/cc-plugin-wcrp
# feature/coordinate-standard db0791d plugins/coordinate_standard/classify.py
def _ncattr(var_or_ds, name: str, default=""):
    """Safe attribute read for both netCDF4 variables and Datasets."""
    return getattr(var_or_ds, name, default) or default


# Adapted from swarnaleem's neutral_dtype() — github.com/swarnaleem/cc-plugin-wcrp
# feature/coordinate-standard db0791d plugins/coordinate_standard/classify.py
def _neutral_dtype(var) -> str:
    """Return 'character', 'integer', or 'double' for a netCDF4 variable."""
    kind = getattr(getattr(var, "dtype", None), "kind", "")
    if kind in ("S", "U"):
        return "character"
    if kind in ("i", "u"):
        return "integer"
    return "double"


# Adapted from swarnaleem's _compare_units() — github.com/swarnaleem/cc-plugin-wcrp
# feature/coordinate-standard db0791d plugins/coordinate_standard/matching.py
def _compare_units(candidate_units: str, entry_units: str) -> tuple:
    """Tiered units comparison backed by udunits.

    Returns (level, message): 'ok', 'warn' (convertible but not identical),
    or 'fail'. CMOR time templates like 'days since ?' accept any date.
    """
    if not entry_units:
        return "ok", ""
    if not candidate_units:
        return "warn", f"units missing; table expects '{entry_units}'"
    if candidate_units == entry_units:
        return "ok", ""
    if "?" in entry_units:
        pattern = re.escape(entry_units).replace(r"\?", ".+")
        if re.fullmatch(pattern, candidate_units):
            return "ok", ""
        if " since " in entry_units and " since " in candidate_units:
            entry_base = entry_units.split(" since ")[0]
            cand_base = candidate_units.split(" since ")[0]
            if cfutil.units_convertible(cand_base, entry_base):
                return "warn", (f"units '{candidate_units}' use base unit "
                                f"'{cand_base}'; table expects '{entry_base}'")
        return "fail", (f"units '{candidate_units}' do not match the table "
                        f"template '{entry_units}'")
    if cfutil.units_convertible(candidate_units, entry_units):
        return "warn", (f"units '{candidate_units}' convertible to but not "
                        f"identical to table units '{entry_units}'")
    return "fail", f"units '{candidate_units}' not convertible to '{entry_units}'"


def _is_time_dim(dim_id: str) -> bool:
    return dim_id.startswith("time")


def _as_list(val) -> list:
    """Normalise a CMIP7 table 'requested' / 'requested_bounds' field to a list."""
    if not val:
        return []
    if isinstance(val, list):
        return [v for v in val if v != ""]
    return [val] if isinstance(val, str) and val else list(val)


def _is_scalar_coord(ce: dict) -> bool:
    """True if the coordinate entry represents a scalar (no multi-value list)."""
    return not bool(_as_list(ce.get("requested", [])))


def _decode_char_var(var) -> list:
    """Decode a netCDF4 char(n, strlen) or char(strlen) variable to a list of strings."""
    raw = np.asarray(var[:])
    if raw.ndim == 1:
        return [raw.tobytes().decode("utf-8", errors="replace").rstrip("\x00").strip()]
    return [
        raw[i].tobytes().decode("utf-8", errors="replace").rstrip("\x00").strip()
        for i in range(raw.shape[0])
    ]


def _decode_char_scalar(var) -> str:
    """Decode a netCDF4 char(strlen) variable to a plain string."""
    return _decode_char_var(var)[0]


def _parse_formula_terms(ft_str: str) -> dict:
    """Parse 'ap: ap b: b ps: ps' style formula_terms string."""
    return {m.group(1): m.group(2) for m in re.finditer(r"(\w+)\s*:\s*(\w+)", ft_str)}


def _find_formula_entry(formula_entries: dict, var_name: str, generic_id: str) -> dict:
    """Return the formula_terms table entry whose out_name and dimension match."""
    for entry in formula_entries.values():
        if (entry.get("out_name") == var_name
                and generic_id in entry.get("dimensions", "")):
            return entry
    return {}
