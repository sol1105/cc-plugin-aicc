"""Validation of scalar, multi-value, text, and site coordinates."""

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx

from cc_plugin_aicc.utils import (
    _cmor_bound_tol_vals,
    _cmor_tol_val,
    _compare_units,
    _decode_char_scalar,
    _decode_char_var,
    _format_attribute,
    _ncattr,
)
from cc_plugin_aicc.validation.attributes import (
    _check_coord_attrs,
    _check_coord_axis,
    _check_coord_positive_with_warning,
    _check_coord_type,
    _check_coord_units,
    _check_valid_range,
)
from cc_plugin_aicc.validation.bounds import (
    _check_bounds_reference,
    _check_bounds_structure,
)
from cc_plugin_aicc.validation.results import append_context_results


def _check_scalar_coord(
    ctx: TestCtx,
    ds,
    dim_id: str,
    out_name: str,
    ce: dict,
    value: str,
    is_character: bool,
    data_out_name: str,
    dim_coord_names: set,
    aux_coord_names: set,
    must_have_bounds: bool = False,
    bounds_values: str = "",
) -> list:
    """Check a scalar coordinate. Returns list[Result] for LOW advisories."""
    low_ctx = TestCtx(BaseCheck.LOW, f"[AICC005] Coordinate '{out_name}' (advisory)")
    medium_ctx = TestCtx(
        BaseCheck.MEDIUM, f"[AICC005] Coordinate '{out_name}' (recommended)"
    )

    # Locate by exact out_name, fall back to standard_name search
    coord_var_name = out_name if out_name in ds.variables else None
    if coord_var_name is None:
        expected_sn = ce.get("standard_name", "")
        if expected_sn:
            matches = ds.get_variables_by_attributes(standard_name=expected_sn)
            if matches:
                coord_var_name = matches[0].name

    if coord_var_name is None:
        ctx.add_failure(
            f"Scalar coordinate '{out_name}' (dim_id='{dim_id}') not found in file."
        )
        return []
    ctx.add_pass()
    if coord_var_name != out_name:
        ctx.add_failure(
            f"Scalar coordinate '{coord_var_name}' was identified by metadata, but "
            f"CMOR requires the variable name '{out_name}'."
        )

    coord_var = ds.variables[coord_var_name]
    _check_coord_attrs(
        ctx, low_ctx, coord_var, coord_var_name, ce, medium_ctx=medium_ctx
    )
    _check_coord_type(ctx, coord_var, coord_var_name, ce)
    _check_coord_axis(ctx, low_ctx, coord_var, coord_var_name, ce)
    _check_coord_units(ctx, low_ctx, coord_var, coord_var_name, ce)
    _check_coord_positive_with_warning(ctx, low_ctx, coord_var, coord_var_name, ce)
    if not is_character:
        _check_valid_range(ctx, coord_var, coord_var_name, ce)

    if is_character:
        dims = list(coord_var.dimensions)
        if dims != ["strlen"]:
            ctx.add_failure(
                f"Character scalar coordinate '{coord_var_name}' must have only "
                f"dimension 'strlen'; found {dims}."
            )
        else:
            ctx.add_pass()
        if value:
            actual_str = _decode_char_scalar(coord_var)
            if actual_str != value:
                ctx.add_failure(
                    f"'{coord_var_name}' value={_format_attribute(actual_str)}; "
                    f"expected {_format_attribute(value)}."
                )
            else:
                ctx.add_pass()
    else:
        if coord_var.ndim != 0:
            ctx.add_failure(
                f"Scalar coordinate '{coord_var_name}' (dim_id='{dim_id}') must be "
                f"dimensionless; found dims={list(coord_var.dimensions)}."
            )
        else:
            ctx.add_pass()
            if value:
                try:
                    actual = float(np.asarray(coord_var[...]).flat[0])
                    expected_val = float(value)
                    if actual == expected_val:
                        ctx.add_pass()
                    else:
                        valid_min = ce.get("valid_min", "")
                        valid_max = ce.get("valid_max", "")
                        within_limits = bool(valid_min or valid_max)
                        if valid_min:
                            within_limits = within_limits and actual >= float(valid_min)
                        if valid_max:
                            within_limits = within_limits and actual <= float(valid_max)
                        if within_limits:
                            ctx.add_pass()
                            low_ctx.add_failure(
                                f"'{coord_var_name}' value={actual} differs from "
                                f"expected {expected_val} but is within the prescribed "
                                f"valid range."
                            )
                        else:
                            ctx.add_failure(
                                f"'{coord_var_name}' value={actual}; expected "
                                f"{expected_val}."
                            )
                except (TypeError, ValueError):
                    ctx.add_pass()

    # Data variable must list the scalar coord in its 'coordinates' attribute
    if data_out_name and data_out_name in ds.variables:
        data_var = ds.variables[data_out_name]
        coord_attr = _ncattr(data_var, "coordinates")
        if out_name not in (coord_attr.split() if coord_attr else []):
            ctx.add_failure(
                f"'{data_out_name}' 'coordinates' attribute must include scalar "
                f"coordinate '{out_name}' (dim_id='{dim_id}'); "
                f"current value: {_format_attribute(coord_attr)}."
            )
        else:
            ctx.add_pass()

    # Scalar bounds
    if not is_character and (must_have_bounds or bool(_ncattr(coord_var, "bounds"))):
        bnds_name, bnds_var = _check_bounds_reference(
            ctx, medium_ctx, ds, coord_var, coord_var_name, out_name
        )
        if bnds_var is not None:
            _check_bounds_structure(
                ctx,
                medium_ctx,
                bnds_var,
                bnds_name,
                coord_var,
                ce,
                scalar=True,
            )
            if bounds_values and value:
                try:
                    parts = bounds_values.split()
                    exp_lo, exp_hi = float(parts[0]), float(parts[1])
                    file_bnds = np.asarray(bnds_var[:]).flatten()
                    act_lo, act_hi = float(file_bnds[0]), float(file_bnds[1])

                    if act_lo != exp_lo or act_hi != exp_hi:
                        ctx.add_failure(
                            f"'{bnds_name}' bounds=[{act_lo}, {act_hi}]; expected "
                            f"exact scalar bounds [{exp_lo}, {exp_hi}]."
                        )
                    else:
                        ctx.add_pass()
                except Exception as exc:
                    ctx.add_failure(
                        f"Could not check bounds_values of '{bnds_name}': {exc}"
                    )

    results = []
    append_context_results(results, medium_ctx, low_ctx)
    return results


def _check_multi_value_coord(
    ctx: TestCtx,
    ds,
    out_name: str,
    ce: dict,
    requested: list,
    requested_bounds: list,
    must_have_bounds: bool,
    is_character: bool,
    data_out_name: str = "",
) -> list:
    """Check a multi-value coordinate. Returns list[Result] for LOW advisories."""
    low_ctx = TestCtx(BaseCheck.LOW, f"[AICC005] Coordinate '{out_name}' (advisory)")
    medium_ctx = TestCtx(
        BaseCheck.MEDIUM, f"[AICC005] Coordinate '{out_name}' (recommended)"
    )

    expected_var_name = "sector" if is_character else out_name
    coord_var_name = expected_var_name if expected_var_name in ds.variables else None
    if coord_var_name is None:
        standard_name = ce.get("standard_name", "")
        matches = (
            ds.get_variables_by_attributes(standard_name=standard_name)
            if standard_name
            else []
        )
        if not matches:
            ctx.add_failure(
                f"Coordinate variable '{expected_var_name}' not found in file."
            )
            return []
        coord_var_name = matches[0].name
        ctx.add_failure(
            f"Coordinate variable '{coord_var_name}' was identified by "
            f"standard_name='{standard_name}', but must be named "
            f"'{expected_var_name}'."
        )
    else:
        ctx.add_pass()

    coord_var = ds.variables[coord_var_name]

    # Verify standard_name and long_name against the table
    _check_coord_attrs(
        ctx, low_ctx, coord_var, coord_var_name, ce, medium_ctx=medium_ctx
    )
    _check_coord_type(ctx, coord_var, coord_var_name, ce)
    _check_coord_axis(ctx, low_ctx, coord_var, coord_var_name, ce)
    _check_coord_positive_with_warning(ctx, low_ctx, coord_var, coord_var_name, ce)
    _check_coord_units(ctx, low_ctx, coord_var, coord_var_name, ce)

    if is_character:
        # Dims must be (out_name, strlen)
        dims = list(coord_var.dimensions)
        expected_dims = [out_name, "strlen"]
        if dims != expected_dims:
            ctx.add_failure(
                f"Character coordinate '{coord_var_name}' must have dims "
                f"{expected_dims}; found {dims}."
            )
        else:
            ctx.add_pass()

        if data_out_name and data_out_name in ds.variables:
            coordinates = _ncattr(ds.variables[data_out_name], "coordinates")
            listed = coordinates.split() if coordinates else []
            if coord_var_name not in listed:
                ctx.add_failure(
                    f"'{data_out_name}' 'coordinates' attribute must include text "
                    f"auxiliary coordinate '{coord_var_name}'."
                )
            else:
                ctx.add_pass()

        if requested:
            try:
                vals = _decode_char_var(coord_var)
                missing = [r for r in requested if r not in vals]
                if missing:
                    ctx.add_failure(
                        f"Character coordinate '{coord_var_name}' missing requested "
                        f"value(s) {missing}; found {vals}."
                    )
                else:
                    ctx.add_pass()
            except Exception as exc:
                ctx.add_failure(
                    f"Could not decode character coordinate '{coord_var_name}': {exc}"
                )
    else:
        tol_str = ce.get("tolerance", "")
        tol_factor = float(tol_str) if tol_str else None

        if coord_var_name != out_name or coord_var.dimensions != (out_name,):
            ctx.add_failure(
                f"Numeric 1-D coordinate must be '{out_name}({out_name})'; found "
                f"'{coord_var_name}({', '.join(coord_var.dimensions)})'."
            )
        else:
            ctx.add_pass()

        _check_valid_range(ctx, coord_var, coord_var_name, ce)

        # Build requested bound pairs aligned with requested values (for tol calc)
        req_floats = [float(r) for r in requested] if requested else []
        req_pairs: list = []
        if requested_bounds:
            req_pairs = list(
                zip(
                    [
                        float(requested_bounds[i])
                        for i in range(0, len(requested_bounds), 2)
                    ],
                    [
                        float(requested_bounds[i])
                        for i in range(1, len(requested_bounds), 2)
                    ],
                )
            )

        if requested:
            try:
                file_vals = np.asarray(coord_var[:]).flatten()
                outside_tol, within_tol_not_exact = [], []
                for i, rv in enumerate(req_floats):
                    if tol_factor is not None:
                        tol = _cmor_tol_val(i, req_floats, req_pairs, tol_factor)
                        matches_tol = [fv for fv in file_vals if abs(rv - fv) <= tol]
                        if not matches_tol:
                            outside_tol.append(rv)
                        elif not any(fv == rv for fv in file_vals):
                            within_tol_not_exact.append(rv)
                    else:
                        if not any(fv == rv for fv in file_vals):
                            outside_tol.append(rv)
                if outside_tol:
                    ctx.add_failure(
                        f"'{coord_var_name}' missing requested value(s) {outside_tol}"
                        + (
                            f" (outside tolerance, factor={tol_factor})."
                            if tol_factor
                            else "."
                        )
                    )
                else:
                    ctx.add_pass()
                if within_tol_not_exact:
                    low_ctx.add_failure(
                        f"'{coord_var_name}' value(s) {within_tol_not_exact} match "
                        f"within tolerance (factor={tol_factor}) but not exactly."
                    )
            except Exception as exc:
                ctx.add_failure(f"Could not check values of '{coord_var_name}': {exc}")

        if must_have_bounds or bool(_ncattr(coord_var, "bounds")):
            bnds_name, bnds_var = _check_bounds_reference(
                ctx, medium_ctx, ds, coord_var, coord_var_name, out_name
            )
            if bnds_var is not None:
                _check_bounds_structure(
                    ctx, medium_ctx, bnds_var, bnds_name, coord_var, ce
                )
                if req_pairs:
                    try:
                        file_bnds = np.asarray(bnds_var[:]).reshape(-1, 2)
                        outside_tol_bnds, within_tol_bnds = [], []
                        for i, (exp_lo, exp_hi) in enumerate(req_pairs):
                            if tol_factor is not None:
                                lo_tol, hi_tol = _cmor_bound_tol_vals(
                                    i, req_pairs, tol_factor
                                )
                                matches = [
                                    fb
                                    for fb in file_bnds
                                    if abs(fb[0] - exp_lo) <= lo_tol
                                    and abs(fb[1] - exp_hi) <= hi_tol
                                ]
                                if not matches:
                                    outside_tol_bnds.append((exp_lo, exp_hi))
                                elif not any(
                                    fb[0] == exp_lo and fb[1] == exp_hi
                                    for fb in file_bnds
                                ):
                                    within_tol_bnds.append((exp_lo, exp_hi))
                            else:
                                if not any(
                                    fb[0] == exp_lo and fb[1] == exp_hi
                                    for fb in file_bnds
                                ):
                                    outside_tol_bnds.append((exp_lo, exp_hi))
                        if outside_tol_bnds:
                            ctx.add_failure(
                                f"'{bnds_name}' missing requested bound pair(s) {outside_tol_bnds}"
                                + (
                                    f" (outside tolerance, factor={tol_factor})."
                                    if tol_factor
                                    else "."
                                )
                            )
                        else:
                            ctx.add_pass()
                        if within_tol_bnds:
                            low_ctx.add_failure(
                                f"'{bnds_name}' bound pair(s) {within_tol_bnds} match "
                                f"within tolerance (factor={tol_factor}) but not exactly."
                            )
                    except Exception as exc:
                        ctx.add_failure(
                            f"Could not check bounds of '{coord_var_name}': {exc}"
                        )

    results = []
    append_context_results(results, medium_ctx, low_ctx)
    return results


def _check_site_coordinate(ctx: TestCtx, ds, site_dimension: str, data_out_name: str):
    """Check the latitude/longitude auxiliary coordinates required for sites."""
    if not data_out_name or data_out_name not in ds.variables:
        return  # another checker owns the missing data-variable finding

    data_var = ds.variables[data_out_name]
    coordinates = _ncattr(data_var, "coordinates")
    listed = coordinates.split() if coordinates else []
    if not listed:
        ctx.add_failure(
            f"Site variable '{data_out_name}' must have a 'coordinates' attribute "
            f"listing latitude and longitude auxiliary coordinates."
        )
        return

    for standard_name, expected_units in (
        ("latitude", "degrees_north"),
        ("longitude", "degrees_east"),
    ):
        matches = [
            name
            for name in listed
            if name in ds.variables
            and _ncattr(ds.variables[name], "standard_name") == standard_name
        ]
        if not matches:
            ctx.add_failure(
                f"Site variable '{data_out_name}' coordinates={listed} must include "
                f"an existing auxiliary coordinate with standard_name="
                f"'{standard_name}'."
            )
            continue
        ctx.add_pass()
        name = matches[0]
        var = ds.variables[name]
        if var.dimensions != (site_dimension,):
            ctx.add_failure(
                f"Site {standard_name} coordinate '{name}' must be a function of "
                f"dimension '{site_dimension}'; found {list(var.dimensions)}."
            )
        else:
            ctx.add_pass()
        level, message = _compare_units(_ncattr(var, "units"), expected_units)
        if level != "ok":
            ctx.add_failure(
                f"Site {standard_name} coordinate '{name}' units: {message}"
            )
        else:
            ctx.add_pass()
