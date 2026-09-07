"""Shared coordinate-bounds validators."""

import numpy as np
from compliance_checker.base import TestCtx

from cc_plugin_aicc.utils import _format_attribute, _ncattr
from cc_plugin_aicc.validation.attributes import _check_coord_type, _strict_direction


def _check_bounds_direction(
    ctx: TestCtx,
    coord_var,
    coord_name: str,
    bnds_var,
    bnds_name: str,
    stored_direction: str,
    *,
    scalar: bool = False,
):
    """Check bounds within and between cells against their parent direction."""
    direction = (
        stored_direction if stored_direction in {"increasing", "decreasing"} else ""
    )
    direction_source = f"stored_direction={stored_direction!r}"
    if not direction and not scalar and coord_var.ndim == 1:
        try:
            direction = (
                _strict_direction(coord_var[:], f"Parent coordinate '{coord_name}'")
                or ""
            )
        except (TypeError, ValueError):
            direction = ""
        direction_source = f"the strictly monotonic parent coordinate '{coord_name}'"
    if not direction:
        return
    try:
        bounds = np.ma.asarray(bnds_var[:], dtype="float64")
        if bounds.shape[-1] != 2:
            return  # shape is owned by the structural check
        pairs = np.asarray(bounds.filled(np.nan), dtype="float64").reshape(-1, 2)
        differences = pairs[:, 1] - pairs[:, 0]
        valid = (
            np.all(differences > 0)
            if direction == "increasing"
            else np.all(differences < 0)
        )
    except (TypeError, ValueError):
        return
    if valid:
        ctx.add_pass()
    else:
        expected = "lower-to-upper" if direction == "increasing" else "upper-to-lower"
        ctx.add_failure(
            f"Bounds variable '{bnds_name}' is not ordered {expected} consistently "
            f"with {direction_source}."
        )

    if not scalar and pairs.shape[0] > 1:
        overall_differences = np.diff(pairs, axis=0)
        overall_valid = (
            np.all(overall_differences > 0)
            if direction == "increasing"
            else np.all(overall_differences < 0)
        )
        if overall_valid:
            ctx.add_pass()
        else:
            ctx.add_failure(
                f"Bounds variable '{bnds_name}' is not strictly {direction} "
                f"between successive cells, consistently with {direction_source}."
            )


def _check_trailing_dimension(
    medium_ctx: TestCtx,
    var,
    var_name: str,
    recommended_name: str,
    *,
    description: str,
):
    """Check the recommended name of a bounds-like trailing dimension."""
    if var.ndim and var.dimensions[-1] != recommended_name:
        medium_ctx.add_failure(
            f"{description} '{var_name}' uses trailing dimension "
            f"'{var.dimensions[-1]}'; recommended name is '{recommended_name}'."
        )


def _check_bounds_reference(
    ctx: TestCtx,
    medium_ctx: TestCtx,
    ds,
    coord_var,
    coord_name: str,
    recommended_coord_name: str | None = None,
    *,
    recommended_name: str | None = None,
):
    """Follow a bounds attribute and separately check its recommended name.

    The attribute value is authoritative for locating the bounds variable.  The
    CMOR-derived name is only a naming recommendation and is never used to infer
    that a bounds variable is absent.
    """
    recommended_name = recommended_name or (
        f"{recommended_coord_name or coord_name}_bnds"
    )
    declared = _ncattr(coord_var, "bounds")
    if not declared:
        ctx.add_failure(
            f"'{coord_name}' must have a 'bounds' attribute naming its bounds variable."
        )
        return None, None
    ctx.add_pass()
    if declared != recommended_name:
        medium_ctx.add_failure(
            f"'{coord_name}' bounds={_format_attribute(declared)}; recommended name "
            f"is '{recommended_name}'."
        )
    if declared not in ds.variables:
        ctx.add_failure(
            f"Bounds variable {_format_attribute(declared)} for '{coord_name}' not found."
        )
        return declared, None
    ctx.add_pass()
    return declared, ds.variables[declared]


def _check_bounds_structure(
    ctx: TestCtx,
    medium_ctx: TestCtx,
    bnds_var,
    bnds_name: str,
    coord_var,
    ce: dict,
    *,
    scalar: bool = False,
):
    """Check bounds dimensions, size-two axis, storage type, and direction."""
    if scalar:
        valid_shape = bnds_var.ndim == 1 and bnds_var.shape == (2,)
        expected = "one size-2 dimension"
    else:
        valid_shape = (
            coord_var.ndim == 1
            and bnds_var.ndim == 2
            and bnds_var.dimensions[0] == coord_var.dimensions[0]
            and bnds_var.shape[0] == coord_var.shape[0]
            and bnds_var.shape[1] == 2
        )
        expected = f"dimensions ('{coord_var.dimensions[0]}', <size-2>)"
    if not valid_shape:
        ctx.add_failure(
            f"Bounds variable '{bnds_name}' has dimensions "
            f"{list(bnds_var.dimensions)} and shape {bnds_var.shape}; expected {expected}."
        )
    else:
        ctx.add_pass()
    _check_coord_type(ctx, bnds_var, bnds_name, ce)
    _check_trailing_dimension(
        medium_ctx,
        bnds_var,
        bnds_name,
        "bnds",
        description="Bounds variable",
    )
    _check_bounds_direction(
        ctx,
        coord_var,
        getattr(coord_var, "name", "coordinate"),
        bnds_var,
        bnds_name,
        ce.get("stored_direction", ""),
        scalar=scalar,
    )
