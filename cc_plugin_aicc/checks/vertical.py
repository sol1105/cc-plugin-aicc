"""Generic vertical-coordinate checks for AICC."""

from compliance_checker.base import BaseCheck, TestCtx
from compliance_checker.cf import util as cfutil

from cc_plugin_aicc.config import VERTICAL_GENERIC_IDS
from cc_plugin_aicc.utils import (
    _compare_units,
    _dimension_ids,
    _find_formula_entry,
    _format_attribute,
    _ncattr,
    _parse_formula_terms,
)
from cc_plugin_aicc.validation.attributes import (
    _DIRECT_VERTICAL_STANDARD_NAMES,
    _check_coord_attrs,
    _check_coord_type,
    _check_direct_vertical_values,
    _check_positive_attribute,
    _check_profile_direction,
    _check_strict_monotonicity,
    _check_valid_range,
    _implied_positive,
)
from cc_plugin_aicc.validation.bounds import (
    _check_bounds_direction,
    _check_bounds_reference,
    _check_bounds_structure,
    _check_trailing_dimension,
)
from cc_plugin_aicc.validation.formulas import _formula_vertical_profile
from cc_plugin_aicc.validation.results import (
    append_context_results,
    vertical_config_context,
)


class VerticalChecks:
    """Generic vertical coordinate and formula validation."""

    def check_vertical(self, ds):
        """Verify vertical coordinate(s) against the CMIP7 coordinate table."""
        vert_dims = [d for d in self.requested_dims if d in VERTICAL_GENERIC_IDS]
        if not vert_dims:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC003] Vertical coordinates")
            ctx.add_pass()
            return [ctx.to_result()]

        if self._vert_mapping is None:
            ctx = vertical_config_context(ds, "[AICC003] Vertical coordinates")
            return [ctx.to_result()]

        results = []
        axis_entries = self.CTcoords.get("axis_entry", {})
        formula_entries = self.CTformulas.get("formula_entry", {})

        # Use cfutil for CF-aware Z variable discovery
        z_var_names = set(cfutil.get_z_variables(ds))

        for generic_id in vert_dims:
            coord_key = self._vert_mapping.get(generic_id)
            if not coord_key:
                ctx = TestCtx(BaseCheck.HIGH, f"[AICC003] {generic_id}")
                ctx.add_failure(
                    f"No vertical coordinate mapping for '{generic_id}' "
                    f"in config entry '{self._conf_key}'."
                )
                results.append(ctx.to_result())
                continue

            ce = axis_entries.get(coord_key)
            if ce is None:
                ctx = TestCtx(BaseCheck.HIGH, f"[AICC003] {generic_id}")
                ctx.add_failure(
                    f"Configured vertical coordinate entry '{coord_key}' for "
                    f"'{generic_id}' was not found in CMIP7_coordinate.json."
                )
                results.append(ctx.to_result())
                continue

            out_name = ce.get("out_name", "lev")
            expected_sn = ce.get("standard_name", "")
            expected_units = ce.get("units", "")
            expected_positive = ce.get("positive", "")
            expected_formula = ce.get("formula", "")
            must_have_bounds = ce.get("must_have_bounds", "no") == "yes"
            z_factors_str = ce.get("z_factors", "")
            z_bounds_factors_str = ce.get("z_bounds_factors", "")

            ctx = TestCtx(
                BaseCheck.HIGH,
                f"[AICC003] Vertical coordinate '{out_name}' ({generic_id})",
            )
            low_ctx = TestCtx(
                BaseCheck.LOW,
                f"[AICC003] Vertical coordinate '{out_name}' ({generic_id}, advisory)",
            )
            medium_ctx = TestCtx(
                BaseCheck.MEDIUM,
                f"[AICC003] Vertical coordinate '{out_name}' "
                f"({generic_id}, recommended)",
            )

            table_generic_id = ce.get("generic_level_name", "")
            if table_generic_id != generic_id:
                ctx.add_failure(
                    f"Configured CMOR coordinate entry '{coord_key}' has "
                    f"generic_level_name={_format_attribute(table_generic_id)}; "
                    f"expected '{generic_id}'."
                )
            else:
                ctx.add_pass()

            # Locate lev variable: exact out_name, then cfutil Z vars by standard_name
            if out_name in ds.variables:
                lev_var_name = out_name
            else:
                lev_var_name = next(
                    (
                        v
                        for v in z_var_names
                        if expected_sn
                        and _ncattr(ds.variables[v], "standard_name") == expected_sn
                    ),
                    next(iter(z_var_names), None) if z_var_names else None,
                )

            if lev_var_name is None:
                ctx.add_failure(
                    f"Vertical coordinate variable not found for '{generic_id}' "
                    f"(expected out_name='{out_name}', standard_name='{expected_sn}')."
                )
                results.append(ctx.to_result())
                continue
            ctx.add_pass()

            lev_var = ds.variables[lev_var_name]
            if lev_var_name != out_name or lev_var.dimensions != (out_name,):
                ctx.add_failure(
                    f"Generic vertical coordinate must be '{out_name}({out_name})'; "
                    f"found '{lev_var_name}({', '.join(lev_var.dimensions)})'."
                )
            else:
                ctx.add_pass()

            # axis=Z
            if _ncattr(lev_var, "axis") != "Z":
                ctx.add_failure(f"'{lev_var_name}' must have attribute axis='Z'.")
            else:
                ctx.add_pass()

            # Attributes from CMIP7_coordinate.json
            _check_coord_attrs(
                ctx, low_ctx, lev_var, lev_var_name, ce, medium_ctx=medium_ctx
            )
            _check_coord_type(ctx, lev_var, lev_var_name, ce)
            _check_valid_range(ctx, lev_var, lev_var_name, ce)

            actual_formula = _ncattr(lev_var, "formula")
            if expected_formula:
                if actual_formula != expected_formula:
                    ctx.add_failure(
                        f"'{lev_var_name}' formula="
                        f"{_format_attribute(actual_formula)}; expected "
                        f"{_format_attribute(expected_formula)} from the CMOR table."
                    )
                else:
                    ctx.add_pass()
            elif actual_formula:
                low_ctx.add_failure(
                    f"'{lev_var_name}' formula="
                    f"{_format_attribute(actual_formula)}, but the CMOR entry "
                    f"does not prescribe a formula. Verify that the extra "
                    f"attribute is valid."
                )

            # units (via udunits-backed comparison)
            if expected_units:
                level, msg = _compare_units(_ncattr(lev_var, "units"), expected_units)
                if level != "ok":
                    ctx.add_failure(f"'{lev_var_name}' units: {msg}")
                else:
                    ctx.add_pass()

            # positive
            if expected_positive:
                actual_pos = _ncattr(lev_var, "positive")
                if actual_pos != expected_positive:
                    ctx.add_failure(
                        f"'{lev_var_name}' positive="
                        f"{_format_attribute(actual_pos)}; "
                        f"expected '{expected_positive}'."
                    )
                else:
                    ctx.add_pass()

            # bounds
            has_bounds = bool(_ncattr(lev_var, "bounds"))
            bnds_name = None
            bnds_var = None
            if must_have_bounds or has_bounds:
                bnds_name, bnds_var = _check_bounds_reference(
                    ctx, medium_ctx, ds, lev_var, lev_var_name, out_name
                )
                if bnds_var is not None:
                    _check_bounds_structure(
                        ctx, medium_ctx, bnds_var, bnds_name, lev_var, ce
                    )
                    lev_attrs = set(lev_var.ncattrs())
                    allowed_bnds_attrs = {
                        attr
                        for attr in ("formula", "formula_terms")
                        if attr in lev_attrs
                    }
                    unexpected_bnds_attrs = [
                        attr
                        for attr in bnds_var.ncattrs()
                        if attr not in allowed_bnds_attrs
                    ]
                    if unexpected_bnds_attrs:
                        ctx.add_failure(
                            f"'{bnds_name}' has unexpected attributes: "
                            f"{unexpected_bnds_attrs}. Only 'formula' and "
                            f"'formula_terms' are allowed, and only when also "
                            f"present on '{lev_var_name}'."
                        )
                    else:
                        ctx.add_pass()

                    if "formula" in allowed_bnds_attrs:
                        bounds_formula = _ncattr(bnds_var, "formula")
                        if bounds_formula != actual_formula:
                            ctx.add_failure(
                                f"'{bnds_name}' formula="
                                f"{_format_attribute(bounds_formula)}; expected the "
                                f"same formula as '{lev_var_name}': "
                                f"{_format_attribute(actual_formula)}."
                            )
                        else:
                            ctx.add_pass()

            # formula_terms
            if z_factors_str:
                ft_attr = _ncattr(lev_var, "formula_terms")
                if not ft_attr:
                    ctx.add_failure(
                        f"'{lev_var_name}' must have 'formula_terms' attribute "
                        f"(formula: '{ce.get('formula', '')}')."
                    )
                else:
                    expected_ft_map = _parse_formula_terms(z_factors_str)
                    ft_map = _parse_formula_terms(ft_attr)
                    if ft_map != expected_ft_map:
                        ctx.add_failure(
                            f"'{lev_var_name}' formula_terms="
                            f"{_format_attribute(ft_attr)}; expected "
                            f"{_format_attribute(z_factors_str)} from the CMOR "
                            f"table."
                        )
                    else:
                        ctx.add_pass()

                    # Validate the CMOR-prescribed variables rather than allowing
                    # divergent file metadata to select different inputs.
                    for term, var_name in expected_ft_map.items():
                        if var_name not in ds.variables:
                            ctx.add_failure(
                                f"formula_terms: term '{term}' references '{var_name}' "
                                f"which does not exist in the file."
                            )
                        else:
                            ctx.add_pass()
                            ft_entry = _find_formula_entry(
                                formula_entries,
                                var_name,
                                generic_id,
                                self.requested_dims,
                            )
                            if ft_entry:
                                _check_formula_var_attrs(
                                    ctx,
                                    low_ctx,
                                    ds.variables[var_name],
                                    var_name,
                                    ft_entry,
                                    expected_dimensions=self._expected_dimensions(
                                        ds, _dimension_ids(ft_entry)
                                    ),
                                )

            # bounds formula terms
            if (must_have_bounds or has_bounds) and z_bounds_factors_str:
                zbf_map = _parse_formula_terms(z_bounds_factors_str)
                if bnds_var is not None:
                    actual_zbf_map = _parse_formula_terms(
                        _ncattr(bnds_var, "formula_terms")
                    )
                    if actual_zbf_map != zbf_map:
                        ctx.add_failure(
                            f"'{bnds_name}' formula_terms="
                            f"{_format_attribute(_ncattr(bnds_var, 'formula_terms'))}; "
                            f"expected {_format_attribute(z_bounds_factors_str)} "
                            "from the CMOR table."
                        )
                    else:
                        ctx.add_pass()
                for term, var_name in zbf_map.items():
                    if var_name not in ds.variables:
                        ctx.add_failure(
                            f"bounds formula_terms: term '{term}' references "
                            f"'{var_name}' which does not exist."
                        )
                    else:
                        ctx.add_pass()
                        ft_entry = _find_formula_entry(
                            formula_entries,
                            var_name,
                            generic_id,
                            self.requested_dims,
                        )
                        if ft_entry:
                            _check_formula_var_attrs(
                                ctx,
                                low_ctx,
                                ds.variables[var_name],
                                var_name,
                                ft_entry,
                                expected_dimensions=self._expected_dimensions(
                                    ds, _dimension_ids(ft_entry)
                                ),
                                bounds=True,
                                generic_out_name=self._generic_vertical_out_name(
                                    generic_id
                                ),
                                medium_ctx=medium_ctx,
                            )

            results.append(ctx.to_result())
            append_context_results(results, medium_ctx, low_ctx)

        return results

    # ------------------------------------------------------------------

    def check_vertical_direction(self, ds):
        """Verify stored and physical direction of generic vertical coordinates."""
        vert_dims = [d for d in self.requested_dims if d in VERTICAL_GENERIC_IDS]
        if not vert_dims:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC003b] Vertical direction")
            ctx.add_pass()
            return [ctx.to_result()]

        if self._vert_mapping is None:
            ctx = vertical_config_context(ds, "[AICC003b] Vertical direction")
            return [ctx.to_result()]

        results = []
        axis_entries = self.CTcoords.get("axis_entry", {})
        z_var_names = set(cfutil.get_z_variables(ds))

        for generic_id in vert_dims:
            ctx = TestCtx(
                BaseCheck.HIGH,
                f"[AICC003b] Vertical direction ({generic_id})",
            )
            coord_key = self._vert_mapping.get(generic_id)
            if not coord_key:
                ctx.add_failure(
                    f"No vertical coordinate mapping for '{generic_id}' in "
                    f"config entry '{self._conf_key}'."
                )
                results.append(ctx.to_result())
                continue

            ce = axis_entries.get(coord_key)
            if ce is None:
                ctx.add_failure(
                    f"Configured vertical coordinate entry '{coord_key}' for "
                    f"'{generic_id}' is absent from CMIP7_coordinate.json."
                )
                results.append(ctx.to_result())
                continue

            out_name = ce.get("out_name", "lev")
            # Direction semantics come from the selected CMOR table entry.  The
            # file's standard_name is checked separately by AICC003 and may be
            # absent or incorrect, so it must not control this check.
            table_standard_name = ce.get("standard_name", "")
            if out_name in ds.variables:
                var_name = out_name
            else:
                var_name = next(
                    (
                        name
                        for name in z_var_names
                        if table_standard_name
                        and _ncattr(ds.variables[name], "standard_name")
                        == table_standard_name
                    ),
                    None,
                )

            if var_name is None:
                ctx.add_failure(
                    f"Cannot check vertical direction for '{generic_id}': "
                    f"coordinate '{out_name}' was not found."
                )
                results.append(ctx.to_result())
                continue

            coord_var = ds.variables[var_name]
            stored_direction = ce.get("stored_direction", "")
            if stored_direction:
                _check_profile_direction(
                    ctx,
                    coord_var[:],
                    stored_direction,
                    f"Stored coordinate '{var_name}'",
                    source_ndim=coord_var.ndim,
                )
                declared_bounds = _ncattr(coord_var, "bounds")
                if declared_bounds in ds.variables:
                    _check_bounds_direction(
                        ctx,
                        coord_var,
                        var_name,
                        ds.variables[declared_bounds],
                        declared_bounds,
                        stored_direction,
                    )
            else:
                _check_strict_monotonicity(
                    ctx,
                    coord_var[:],
                    f"Stored coordinate '{var_name}'",
                    source_ndim=coord_var.ndim,
                )
                declared_bounds = _ncattr(coord_var, "bounds")
                if declared_bounds in ds.variables:
                    _check_bounds_direction(
                        ctx,
                        coord_var,
                        var_name,
                        ds.variables[declared_bounds],
                        declared_bounds,
                        "",
                    )

            implied_positive = _implied_positive(table_standard_name)
            if implied_positive:
                _check_positive_attribute(
                    ctx,
                    coord_var,
                    var_name,
                    table_standard_name,
                    implied_positive,
                )

            formula = ce.get("formula", "")
            if formula:
                expected_formula_terms = _parse_formula_terms(ce.get("z_factors", ""))
                file_formula_terms = _parse_formula_terms(
                    _ncattr(coord_var, "formula_terms")
                )
                metadata_matches = (
                    bool(expected_formula_terms)
                    and file_formula_terms == expected_formula_terms
                )
                terms_available = metadata_matches and all(
                    term_var in ds.variables
                    for term_var in expected_formula_terms.values()
                )
                if terms_available:
                    try:
                        profile, sample = _formula_vertical_profile(
                            ds,
                            coord_var,
                            ce,
                            expected_formula_terms,
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        ctx.add_failure(
                            f"Could not evaluate vertical formula for "
                            f"'{var_name}': {exc}."
                        )
                    else:
                        if stored_direction:
                            _check_profile_direction(
                                ctx,
                                profile,
                                stored_direction,
                                f"Formula-derived profile for '{var_name}' at {sample}",
                            )
            elif table_standard_name in _DIRECT_VERTICAL_STANDARD_NAMES:
                _check_direct_vertical_values(
                    ctx, coord_var[:], var_name, table_standard_name
                )

            results.append(ctx.to_result())

        return results

    # ------------------------------------------------------------------


def _check_formula_var_attrs(
    ctx: TestCtx,
    low_ctx: TestCtx,
    var,
    var_name: str,
    ft_entry: dict,
    *,
    expected_dimensions=None,
    bounds: bool = False,
    generic_out_name: str = "",
    medium_ctx: TestCtx | None = None,
):
    """Check a formula-term variable's table-defined attributes."""
    _check_coord_type(ctx, var, var_name, ft_entry)

    expected_units = ft_entry.get("units", "")
    if expected_units:
        level, msg = _compare_units(_ncattr(var, "units"), expected_units)
        if level != "ok":
            ctx.add_failure(f"Formula term '{var_name}' units: {msg}")
        else:
            ctx.add_pass()

    _check_coord_attrs(
        ctx,
        low_ctx,
        var,
        var_name,
        ft_entry,
        enforce_long_name_fallback=False,
    )

    if expected_dimensions is None:
        return
    actual_dimensions = list(var.dimensions)
    expected = list(expected_dimensions)

    def dimensions_match(actual, prescribed):
        return len(actual) == len(prescribed) and all(
            expected_name == "<ncells>" or actual_name == expected_name
            for actual_name, expected_name in zip(actual, prescribed)
        )

    if bounds and generic_out_name in expected:
        valid = (
            var.ndim == len(expected) + 1
            and dimensions_match(actual_dimensions[:-1], expected)
            and var.shape[-1] == 2
        )
        expected_description = f"{expected} followed by a size-2 bounds dimension"
        if medium_ctx is not None:
            _check_trailing_dimension(
                medium_ctx,
                var,
                var_name,
                "bnds",
                description="Bounds formula-term variable",
            )
    else:
        valid = dimensions_match(actual_dimensions, expected)
        expected_description = str(expected)
    if valid:
        ctx.add_pass()
    else:
        term_label = "Bounds formula term" if bounds else "Formula term"
        ctx.add_failure(
            f"{term_label} '{var_name}' has dimensions {actual_dimensions}; "
            f"expected {expected_description} from its formula-term table entry."
        )
