"""Shared coordinate attribute and numeric-profile validators."""

import numpy as np
from compliance_checker.base import TestCtx

from cc_plugin_aicc.utils import (
    _compare_units,
    _format_attribute,
    _ncattr,
    _neutral_dtype,
)

_DIRECT_VERTICAL_STANDARD_NAMES = frozenset({"air_pressure", "height", "depth"})


def _implied_positive(standard_name: str):
    """Return the physical positive direction implied by a standard_name."""
    if standard_name in {"air_pressure", "depth"}:
        return "down"
    if standard_name == "height":
        return "up"
    if standard_name.startswith("atmosphere_"):
        if "pressure_coordinate" in standard_name:
            return "down"
        if "height_coordinate" in standard_name or "sleve_coordinate" in standard_name:
            return "up"
    if standard_name.startswith("ocean_") and standard_name.endswith("_coordinate"):
        # CF ocean parametric formulas calculate z as height, positive upwards.
        return "up"
    return None


def _numeric_profile(values, label: str):
    """Return a finite, unmasked one-dimensional numeric profile."""
    masked = np.ma.asarray(values, dtype="float64").reshape(-1)
    if np.any(np.ma.getmaskarray(masked)):
        raise ValueError(f"{label} contains masked values")
    profile = np.asarray(masked, dtype="float64")
    if not np.all(np.isfinite(profile)):
        raise ValueError(f"{label} contains non-finite values")
    return profile


def _strict_direction(values, label: str) -> str | None:
    """Return the direction of a finite strict numeric profile, if any."""
    profile = _numeric_profile(values, label)
    if profile.size < 2:
        return None
    differences = np.diff(profile)
    if np.all(differences > 0):
        return "increasing"
    if np.all(differences < 0):
        return "decreasing"
    return None


def _check_strict_monotonicity(
    ctx: TestCtx,
    values,
    label: str,
    *,
    source_ndim=None,
) -> str | None:
    """Require strict monotonicity without prescribing its direction."""
    if source_ndim is not None and source_ndim != 1:
        ctx.add_failure(
            f"{label} must be one-dimensional to check strict monotonicity; "
            f"found {source_ndim} dimensions."
        )
        return None
    try:
        direction = _strict_direction(values, label)
    except (TypeError, ValueError) as exc:
        ctx.add_failure(f"Could not check strict monotonicity: {exc}.")
        return None
    if direction is None:
        ctx.add_failure(
            f"{label} is not strictly monotonic (increasing or decreasing)."
        )
        return None
    ctx.add_pass()
    return direction


def _check_profile_direction(
    ctx: TestCtx,
    values,
    direction: str,
    label: str,
    source_ndim=None,
):
    """Add a strict monotonicity result for a stored or calculated profile."""
    if source_ndim is not None and source_ndim != 1:
        ctx.add_failure(
            f"{label} must be one-dimensional to check stored_direction; "
            f"found {source_ndim} dimensions."
        )
        return

    try:
        profile = _numeric_profile(values, label)
    except (TypeError, ValueError) as exc:
        ctx.add_failure(f"Could not check {direction} direction: {exc}.")
        return

    if profile.size < 2:
        ctx.add_failure(
            f"{label} has fewer than two values, so its {direction} direction "
            f"cannot be verified."
        )
        return

    differences = np.diff(profile)
    if direction == "increasing":
        valid = bool(np.all(differences > 0))
    elif direction == "decreasing":
        valid = bool(np.all(differences < 0))
    else:
        ctx.add_failure(f"Unsupported stored_direction '{direction}' for {label}.")
        return

    if valid:
        ctx.add_pass()
    else:
        bad = np.flatnonzero(
            differences <= 0 if direction == "increasing" else differences >= 0
        )
        ctx.add_failure(
            f"{label} is not strictly {direction} as required by "
            f"stored_direction; first mismatch is between positions "
            f"{int(bad[0])} and {int(bad[0]) + 1}."
        )


def _check_positive_attribute(
    ctx: TestCtx,
    coord_var,
    var_name: str,
    standard_name: str,
    implied_positive,
):
    """Check positive against the physical direction implied by standard_name."""
    if implied_positive is None:
        return
    actual = str(_ncattr(coord_var, "positive")).lower()
    if actual != implied_positive:
        displayed = _format_attribute(_ncattr(coord_var, "positive"))
        ctx.add_failure(
            f"'{var_name}' positive={displayed} "
            f"is inconsistent with standard_name='{standard_name}', which implies "
            f"positive='{implied_positive}'."
        )
    else:
        ctx.add_pass()


def _check_direct_vertical_values(
    ctx: TestCtx,
    values,
    var_name: str,
    standard_name: str,
):
    """Check the sign domain of direct pressure, height, and depth values."""
    try:
        profile = _numeric_profile(values, f"Coordinate '{var_name}'")
    except (TypeError, ValueError) as exc:
        ctx.add_failure(f"Could not check physical values: {exc}.")
        return

    if standard_name == "air_pressure":
        valid = bool(np.all(profile > 0))
        expected = "strictly positive"
    else:
        valid = bool(np.all(profile >= 0))
        expected = "non-negative"

    if valid:
        ctx.add_pass()
    else:
        ctx.add_failure(
            f"Coordinate '{var_name}' with standard_name='{standard_name}' must "
            f"contain {expected} values; found minimum {profile.min()}."
        )


def _check_coord_attrs(
    ctx: TestCtx,
    low_ctx: TestCtx,
    var,
    var_name: str,
    ce: dict,
    missing_ok: bool = False,
    medium_ctx: TestCtx | None = None,
    enforce_long_name_fallback: bool = True,
):
    """Check naming attributes with the severities prescribed by CMIP7."""
    expected_sn = ce.get("standard_name", "")
    if expected_sn:
        actual_sn = _ncattr(var, "standard_name")
        if missing_ok and not actual_sn:
            ctx.add_pass()
        elif actual_sn != expected_sn:
            ctx.add_failure(
                f"'{var_name}' standard_name={_format_attribute(actual_sn)}; "
                f"expected '{expected_sn}'."
            )
        else:
            ctx.add_pass()
    elif _ncattr(var, "standard_name") and not missing_ok:
        low_ctx.add_failure(
            f"'{var_name}' standard_name="
            f"{_format_attribute(_ncattr(var, 'standard_name'))}, but the CMOR "
            f"entry does not prescribe a standard_name. Verify that this "
            f"model-dependent metadata is appropriate."
        )

    expected_ln = ce.get("long_name", "")
    actual_ln = _ncattr(var, "long_name")
    if not expected_sn and not missing_ok and enforce_long_name_fallback:
        if not actual_ln:
            ctx.add_failure(
                f"'{var_name}' must have a long_name because the CMOR entry "
                f"does not define a standard_name."
            )
        else:
            ctx.add_pass()
            if expected_ln and actual_ln != expected_ln:
                target_ctx = medium_ctx or low_ctx
                target_ctx.add_failure(
                    f"'{var_name}' long_name={_format_attribute(actual_ln)}; "
                    f"expected '{expected_ln}'."
                )
    elif expected_ln:
        if actual_ln and actual_ln != expected_ln:
            low_ctx.add_failure(
                f"'{var_name}' long_name={_format_attribute(actual_ln)}; "
                f"expected '{expected_ln}'."
            )
        elif not actual_ln and not missing_ok:
            low_ctx.add_failure(
                f"'{var_name}' long_name={_format_attribute(actual_ln)}; "
                f"expected '{expected_ln}'."
            )


def _check_coord_axis(ctx: TestCtx, low_ctx: TestCtx, var, var_name: str, ce: dict):
    """Check a prescribed axis and warn about an unprescribed one."""
    expected = ce.get("axis", "")
    actual = _ncattr(var, "axis")
    if expected:
        if actual != expected:
            ctx.add_failure(
                f"'{var_name}' axis={_format_attribute(actual)}; expected '{expected}'."
            )
        else:
            ctx.add_pass()
    elif actual:
        low_ctx.add_failure(
            f"'{var_name}' axis={_format_attribute(actual)}, but the CMOR entry "
            f"does not prescribe an axis. Verify that the extra attribute is valid."
        )


def _check_coord_units(ctx: TestCtx, low_ctx: TestCtx, var, var_name: str, ce: dict):
    """Check prescribed units and warn about units absent from the table entry."""
    expected = ce.get("units", "")
    actual = _ncattr(var, "units")
    if expected:
        level, msg = _compare_units(actual, expected)
        if level != "ok":
            ctx.add_failure(f"'{var_name}' units: {msg}")
        else:
            ctx.add_pass()
    elif actual:
        low_ctx.add_failure(
            f"'{var_name}' units={_format_attribute(actual)}, but the CMOR entry "
            f"does not prescribe units. Verify that the extra attribute is valid."
        )


def _check_coord_positive_with_warning(
    ctx: TestCtx, low_ctx: TestCtx, var, var_name: str, ce: dict
):
    """Check prescribed positive metadata and warn when it is unprescribed."""
    expected = ce.get("positive", "")
    actual = _ncattr(var, "positive")
    if expected:
        _check_coord_table_positive(ctx, var, var_name, ce)
    elif actual:
        low_ctx.add_failure(
            f"'{var_name}' positive={_format_attribute(actual)}, but the CMOR entry "
            f"does not prescribe positive. Verify that the extra attribute is valid."
        )


def _check_valid_range(ctx: TestCtx, var, var_name: str, ce: dict):
    """Check all numeric coordinate values against independent table limits."""
    valid_min = ce.get("valid_min", "")
    valid_max = ce.get("valid_max", "")
    if valid_min in ("", None) and valid_max in ("", None):
        return
    try:
        values = np.ma.asarray(var[...], dtype="float64")
        values = np.asarray(values.compressed(), dtype="float64")
    except (TypeError, ValueError) as exc:
        ctx.add_failure(f"Could not check valid range of '{var_name}': {exc}.")
        return
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ctx.add_failure(
            f"Could not check valid range of '{var_name}': no finite values."
        )
        return
    if valid_min not in ("", None):
        lower = float(valid_min)
        slack = 1.0e-6 * abs(lower)
        bad = finite < lower - slack
        if np.any(bad):
            ctx.add_failure(
                f"'{var_name}' contains value {finite[bad].min()} below valid_min="
                f"{lower}."
            )
        else:
            ctx.add_pass()
    if valid_max not in ("", None):
        upper = float(valid_max)
        slack = 1.0e-6 * abs(upper)
        bad = finite > upper + slack
        if np.any(bad):
            ctx.add_failure(
                f"'{var_name}' contains value {finite[bad].max()} above valid_max="
                f"{upper}."
            )
        else:
            ctx.add_pass()


def _check_coord_type(
    ctx: TestCtx,
    var,
    var_name: str,
    ce: dict,
):
    """Check a coordinate variable's storage type against its CMOR entry."""
    expected_type = ce.get("type", "")
    if not expected_type:
        return

    actual_type = _neutral_dtype(var)
    if actual_type != expected_type:
        ctx.add_failure(
            f"'{var_name}' has data type '{var.dtype}'"
            + (f" (CMOR type '{actual_type}')" if actual_type else "")
            + f"; expected CMOR type '{expected_type}'."
        )
    else:
        ctx.add_pass()


def _check_coord_table_positive(
    ctx: TestCtx,
    var,
    var_name: str,
    ce: dict,
):
    """Check positive against the value declared by the CMOR table."""
    expected_positive = ce.get("positive", "")
    if not expected_positive:
        return

    actual_positive = _ncattr(var, "positive")
    if actual_positive != expected_positive:
        ctx.add_failure(
            f"'{var_name}' positive={_format_attribute(actual_positive)}; "
            f"expected '{expected_positive}' from the CMOR table."
        )
    else:
        ctx.add_pass()
