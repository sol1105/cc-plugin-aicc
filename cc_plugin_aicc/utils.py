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
def _format_attribute(value) -> str:
    """Format an attribute safely for a one-line finding message."""
    if not isinstance(value, str):
        return repr(value)

    has_control_whitespace = any(char in value for char in "\t\r\n")
    displayed = " ".join(value.split()) if has_control_whitespace else value
    formatted = repr(displayed)
    if has_control_whitespace:
        formatted += (
            " (Note: tab or newline characters were found and removed "
            "from this message.)"
        )
    return formatted


def _ncattr(var_or_ds, name: str, default=""):
    """Read an attribute safely without altering its value."""
    return getattr(var_or_ds, name, default) or default


# Adapted from swarnaleem's neutral_dtype() — github.com/swarnaleem/cc-plugin-wcrp
# feature/coordinate-standard db0791d plugins/coordinate_standard/classify.py
def _neutral_dtype(var) -> str:
    """Return the CMOR type represented by a netCDF4 variable's dtype."""
    dtype = getattr(var, "dtype", None)
    kind = getattr(dtype, "kind", "")
    if kind in ("S", "U") or dtype is str:
        return "character"
    if kind in ("i", "u"):
        return "integer"
    if kind == "f":
        if getattr(dtype, "itemsize", 0) == 4:
            return "real"
        if getattr(dtype, "itemsize", 0) == 8:
            return "double"
    return ""


# Adapted from swarnaleem's _compare_units() — github.com/swarnaleem/cc-plugin-wcrp
# feature/coordinate-standard db0791d plugins/coordinate_standard/matching.py
def _compare_units(candidate_units: str, entry_units: str) -> tuple:
    """Exact units comparison with udunits-backed conversion information.

    Returns ``("ok", "")`` only for an exact table match. Missing, convertible,
    and incompatible units all return ``("fail", message)``. CMOR time
    templates such as ``days since ?`` accept any non-empty reference date.
    """
    if not entry_units:
        return "ok", ""
    if not candidate_units:
        return "fail", (
            f"units missing; table requires {_format_attribute(entry_units)}"
        )
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
                return "fail", (
                    f"units {_format_attribute(candidate_units)} do not match the "
                    f"table template {_format_attribute(entry_units)}; base unit "
                    f"{_format_attribute(cand_base)} is convertible to required "
                    f"base unit {_format_attribute(entry_base)}"
                )
        return "fail", (
            f"units {_format_attribute(candidate_units)} do not match the table "
            f"template {_format_attribute(entry_units)}"
        )
    if cfutil.units_convertible(candidate_units, entry_units):
        return "fail", (
            f"units {_format_attribute(candidate_units)} are convertible to "
            f"required table units {_format_attribute(entry_units)} but are not "
            f"identical"
        )
    return "fail", (
        f"units {_format_attribute(candidate_units)} not convertible to "
        f"{_format_attribute(entry_units)}"
    )


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
    """True if the coordinate table entry prescribes a scalar value or bounds."""
    return ce.get("coordinate_type") == "scalar" or bool(
        ce.get("value") or ce.get("bounds_values")
    )


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


def _dimension_ids(entry: dict) -> list[str]:
    """Return a descriptor's ordered coordinate IDs in either source format."""
    dimensions = entry.get("dimensions") or []
    if isinstance(dimensions, str):
        return dimensions.split()
    result = []
    for dimension in dimensions:
        if isinstance(dimension, str):
            result.append(dimension)
        elif isinstance(dimension, dict) and dimension.get("id"):
            result.append(str(dimension["id"]))
        elif getattr(dimension, "id", None):
            result.append(str(dimension.id))
    return result


def _find_formula_entry(
    formula_entries: dict,
    var_name: str,
    generic_id: str,
    requested_dims=(),
) -> dict:
    """Select the formula-term entry appropriate to this branded variable."""
    candidates = [
        entry for entry in formula_entries.values() if entry.get("out_name") == var_name
    ]
    if not candidates:
        return {}
    generic_matches = [
        entry for entry in candidates if generic_id in _dimension_ids(entry)
    ]
    if generic_matches:
        return generic_matches[0]
    requested = set(requested_dims or [])
    compatible = [
        entry
        for entry in candidates
        if all(identifier in requested for identifier in _dimension_ids(entry))
    ]
    return compatible[0] if compatible else candidates[0]


def _cmor_tol_val(i: int, req_vals: list, bound_pairs: list, tolerance: float) -> float:
    """Compute the CMOR tolerance for one requested coordinate value.

    ``bound_pairs`` is retained for compatibility with existing callers, but
    coordinate-value tolerances do not depend on requested bounds.  The first
    value uses the following spacing and subsequent values use the preceding
    spacing, exactly as specified by the CMIP7 coordinate QC recipe.
    """
    tol = 0.001 * tolerance * abs(req_vals[i])
    if i == 0 and len(req_vals) > 1:
        tol = min(tol, tolerance * abs(req_vals[1] - req_vals[0]))
    elif i > 0:
        tol = min(tol, tolerance * abs(req_vals[i] - req_vals[i - 1]))
    return tol


def _cmor_bound_tol_vals(
    i: int, bound_pairs: list, tolerance: float
) -> tuple[float, float]:
    """Return the CMOR tolerances for a requested lower/upper bound pair."""
    lo, hi = bound_pairs[i]
    cell_tol = tolerance * abs(hi - lo)
    return (
        min(cell_tol, 0.001 * tolerance * abs(lo)),
        min(cell_tol, 0.001 * tolerance * abs(hi)),
    )
