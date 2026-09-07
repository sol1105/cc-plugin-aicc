"""Ordinary coordinate, dimension, and association checks for AICC."""

from compliance_checker.base import BaseCheck, TestCtx
from compliance_checker.cf import util as cfutil

from cc_plugin_aicc.config import HORIZONTAL_DIM_IDS, VERTICAL_GENERIC_IDS
from cc_plugin_aicc.utils import (
    _as_list,
    _is_scalar_coord,
    _is_time_dim,
    _ncattr,
)
from cc_plugin_aicc.validation.attributes import (
    _DIRECT_VERTICAL_STANDARD_NAMES,
    _check_direct_vertical_values,
    _check_positive_attribute,
    _check_profile_direction,
    _check_strict_monotonicity,
    _implied_positive,
)
from cc_plugin_aicc.validation.coordinates import (
    _check_multi_value_coord,
    _check_scalar_coord,
    _check_site_coordinate,
)
from cc_plugin_aicc.validation.results import vertical_config_context


class CoordinateChecks:
    """Ordinary, scalar, text, site, and dimension validation."""

    def check_coord(self, ds):
        """Verify non-grid, non-vertical, non-time coordinate dimensions."""
        other_dims = [
            d
            for d in self.requested_dims
            if d not in HORIZONTAL_DIM_IDS
            and d not in VERTICAL_GENERIC_IDS
            and not _is_time_dim(d)
        ]
        if not other_dims:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC005] Other coordinates")
            ctx.add_pass()
            return [ctx.to_result()]

        results = []
        axis_entries = self.CTcoords.get("axis_entry", {})
        data_out_name = self.var_entry.get("out_name", "") if self.var_entry else ""

        # Use cfutil to classify what is in the file
        aux_coord_names = set(cfutil.get_auxiliary_coordinate_variables(ds))
        dim_coord_names = set(cfutil.get_coordinate_variables(ds))

        for dim_id in other_dims:
            ce = axis_entries.get(dim_id)
            ctx = TestCtx(BaseCheck.HIGH, f"[AICC005] Coordinate '{dim_id}'")

            if ce is None:
                ctx.add_failure(
                    f"dim_id '{dim_id}' not found in CMIP7 coordinate table."
                )
                results.append(ctx.to_result())
                continue

            out_name = ce.get("out_name", dim_id)
            coord_type = ce.get("type", "")
            value = ce.get("value", "")
            requested = _as_list(ce.get("requested", []))
            requested_bounds = _as_list(ce.get("requested_bounds", []))
            must_have_bounds = ce.get("must_have_bounds", "no") == "yes"
            is_character = coord_type == "character"
            is_scalar = _is_scalar_coord(ce)

            if is_scalar:
                bounds_values = ce.get("bounds_values", "")
                low = _check_scalar_coord(
                    ctx,
                    ds,
                    dim_id,
                    out_name,
                    ce,
                    value,
                    is_character,
                    data_out_name,
                    dim_coord_names,
                    aux_coord_names,
                    must_have_bounds,
                    bounds_values,
                )
            else:
                low = _check_multi_value_coord(
                    ctx,
                    ds,
                    out_name,
                    ce,
                    requested,
                    requested_bounds,
                    must_have_bounds,
                    is_character,
                    data_out_name,
                )

            if dim_id == "site":
                _check_site_coordinate(ctx, ds, out_name, data_out_name)

            results.append(ctx.to_result())
            results.extend(low)

        return results

    # ------------------------------------------------------------------

    def check_coordinate_direction(self, ds):
        """Verify stored direction and physical direction where interpretable."""
        axis_entries = self.CTcoords.get("axis_entry", {})
        applicable = []
        for dim_id in self.requested_dims:
            if (
                dim_id in HORIZONTAL_DIM_IDS
                or dim_id in VERTICAL_GENERIC_IDS
                or _is_time_dim(dim_id)
            ):
                continue
            ce = axis_entries.get(dim_id)
            if ce and not _is_scalar_coord(ce) and ce.get("type", "") != "character":
                applicable.append((dim_id, ce))

        if not applicable:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC005b] Coordinate direction")
            ctx.add_pass()
            return [ctx.to_result()]

        results = []
        for dim_id, ce in applicable:
            expected_name = ce.get("out_name", dim_id)
            standard_name = ce.get("standard_name", "")
            ctx = TestCtx(
                BaseCheck.HIGH,
                f"[AICC005b] Coordinate direction '{expected_name}'",
            )

            if expected_name in ds.variables:
                var_name = expected_name
            else:
                matches = ds.get_variables_by_attributes(standard_name=standard_name)
                var_name = matches[0].name if matches else None

            if var_name is None:
                ctx.add_failure(
                    f"Cannot check direction: coordinate '{expected_name}' "
                    f"(standard_name='{standard_name}') was not found."
                )
                results.append(ctx.to_result())
                continue

            coord_var = ds.variables[var_name]
            if coord_var.ndim != 1:
                ctx.add_failure(
                    f"Coordinate '{var_name}' must be one-dimensional for the "
                    f"AICC005b direction check; found dimensions "
                    f"{list(coord_var.dimensions)}."
                )
                results.append(ctx.to_result())
                continue

            stored_direction = ce.get("stored_direction", "")
            if stored_direction:
                _check_profile_direction(
                    ctx,
                    coord_var[:],
                    stored_direction,
                    f"Stored coordinate '{var_name}'",
                )
            else:
                _check_strict_monotonicity(
                    ctx,
                    coord_var[:],
                    f"Coordinate '{var_name}'",
                    source_ndim=coord_var.ndim,
                )

            if standard_name in _DIRECT_VERTICAL_STANDARD_NAMES and not _ncattr(
                coord_var, "formula_terms"
            ):
                implied_positive = _implied_positive(standard_name)
                _check_positive_attribute(
                    ctx, coord_var, var_name, standard_name, implied_positive
                )
                _check_direct_vertical_values(
                    ctx, coord_var[:], var_name, standard_name
                )
            results.append(ctx.to_result())

        return results

    # ------------------------------------------------------------------

    def check_dimensions(self, ds):
        """Verify that the data variable's dimensions are in the expected C order."""
        ctx = TestCtx(BaseCheck.HIGH, "[AICC006] Variable dimension ordering")

        if self.var_entry is None:
            ctx.add_failure("Cannot check dimensions: CMOR table entry not resolved.")
            return [ctx.to_result()]

        if (
            any(dim_id in VERTICAL_GENERIC_IDS for dim_id in self.requested_dims)
            and self._vert_mapping is None
        ):
            ctx = vertical_config_context(ds, "[AICC006] Variable dimension ordering")
            return [ctx.to_result()]

        data_out_name = self.var_entry.get("out_name", "")
        if not data_out_name or data_out_name not in ds.variables:
            ctx.add_pass()
            return [ctx.to_result()]

        expected = self._expected_dimensions(ds, self.requested_dims)
        if expected is None:
            # AICC002 already reports that the horizontal topology cannot be
            # resolved, so avoid a derivative dimension-order finding.
            ctx.add_pass()
            return [ctx.to_result()]

        actual = list(ds.variables[data_out_name].dimensions)
        dimension_summary = (
            f"'{data_out_name}' dimensions in file (C order): {actual}. "
            f"Expected dimensions from CMOR (C order): {expected}."
        )

        if len(actual) != len(expected):
            ctx.add_failure(
                f"{dimension_summary} Found {len(actual)} dimensions; "
                f"expected {len(expected)}."
            )
        else:
            mismatches = [
                f"position {i}: '{act}' (expected '{exp}')"
                for i, (exp, act) in enumerate(zip(expected, actual))
                if exp != "<ncells>" and exp != act
            ]
            if mismatches:
                ctx.add_failure(
                    f"{dimension_summary} Dimension mismatches: "
                    + "; ".join(mismatches)
                    + "."
                )
            else:
                ctx.add_pass()

        return [ctx.to_result()]

    def _expected_dimensions(self, ds, dimension_ids):
        """Resolve CMOR/ESGVoc coordinate IDs to expected netCDF/C dimensions."""
        dimension_ids = _as_list(dimension_ids)
        axis_entries = self.CTcoords.get("axis_entry", {})
        horizontal_dim_ids = [
            dim_id for dim_id in dimension_ids if dim_id in HORIZONTAL_DIM_IDS
        ]
        expected_horizontal = []
        if horizontal_dim_ids:
            handler = self._grid_type_handlers.get(self._grid_type)
            if handler is None:
                return None
            horizontal_entries = self._grid_table_entries(handler)
            expected_horizontal = getattr(self, handler["dimensions"])(
                ds, horizontal_dim_ids, horizontal_entries
            )

        expected = []
        horizontal_added = False
        for dim_id in reversed(dimension_ids):
            if dim_id in HORIZONTAL_DIM_IDS:
                if not horizontal_added:
                    expected.extend(expected_horizontal)
                    horizontal_added = True
                continue
            ce = axis_entries.get(dim_id, {})
            if dim_id in VERTICAL_GENERIC_IDS:
                expected.append(self._generic_vertical_out_name(dim_id))
                continue
            if _is_time_dim(dim_id):
                expected.append(ce.get("out_name", "time"))
                continue
            if ce and _is_scalar_coord(ce):
                continue
            expected.append(ce.get("out_name", dim_id) if ce else dim_id)
        return expected

    def _generic_vertical_out_name(self, generic_id):
        """Return the configured generic coordinate's table-defined output name."""
        axis_entries = self.CTcoords.get("axis_entry", {})
        out_name = axis_entries.get(generic_id, {}).get("out_name", "")
        if not out_name and self._vert_mapping:
            configured_id = self._vert_mapping.get(generic_id)
            out_name = axis_entries.get(configured_id, {}).get("out_name", "")
        return out_name or generic_id

    # ------------------------------------------------------------------

    def check_coordinates_attribute(self, ds):
        """Allow only requested auxiliary/scalar coordinates on the data variable."""
        ctx = TestCtx(
            BaseCheck.LOW,
            "[AICC007] Data variable coordinates attribute",
        )

        data_out_name = self.var_entry.get("out_name", "") if self.var_entry else ""
        if not data_out_name or data_out_name not in ds.variables:
            ctx.add_pass()
            return [ctx.to_result()]

        data_var = ds.variables[data_out_name]
        coordinates = _ncattr(data_var, "coordinates")
        listed_coordinates = coordinates.split() if coordinates else []
        if not listed_coordinates:
            ctx.add_pass()
            return [ctx.to_result()]

        allowed_coordinates = set()
        axis_entries = self.CTcoords.get("axis_entry", {})

        grid_handler = self._grid_type_handlers.get(self._grid_type)
        if grid_handler is None:
            # AICC002 owns the unknown/unsupported grid-type finding. Until the
            # topology is known, latitude and longitude cannot reliably be
            # classified as dimension or auxiliary coordinates. Conservatively
            # allow CF-detected horizontal coordinates here while still checking
            # every other entry in the coordinates attribute.
            for dim_id, candidates in (
                ("latitude", cfutil.get_true_latitude_variables(ds)),
                ("longitude", cfutil.get_true_longitude_variables(ds)),
            ):
                if dim_id in self.requested_dims:
                    allowed_coordinates.update(candidates)

        # Unstructured and curvilinear horizontal coordinates are auxiliary.
        elif self._grid_type in {"unstructured", "curvilinear"}:
            dim_coord_names = set(cfutil.get_coordinate_variables(ds))
            for dim_id, candidates in (
                ("latitude", cfutil.get_true_latitude_variables(ds)),
                ("longitude", cfutil.get_true_longitude_variables(ds)),
            ):
                if dim_id not in self.requested_dims:
                    continue
                allowed_coordinates.update(
                    name
                    for name in candidates
                    if name not in dim_coord_names
                    and ds.variables[name].ndim
                    == (1 if self._grid_type == "unstructured" else 2)
                )

        for dim_id in self.requested_dims:
            if (
                dim_id in HORIZONTAL_DIM_IDS
                or dim_id in VERTICAL_GENERIC_IDS
                or _is_time_dim(dim_id)
            ):
                continue

            ce = axis_entries.get(dim_id, {})
            if not ce:
                continue

            out_name = ce.get("out_name", dim_id)
            if dim_id == "site":
                allowed_coordinates.update(
                    name
                    for name in listed_coordinates
                    if name in ds.variables
                    and _ncattr(ds.variables[name], "standard_name")
                    in {"latitude", "longitude"}
                )
            elif _is_scalar_coord(ce):
                if out_name in ds.variables:
                    allowed_coordinates.add(out_name)
                else:
                    standard_name = ce.get("standard_name", "")
                    if standard_name:
                        allowed_coordinates.update(
                            var.name
                            for var in ds.get_variables_by_attributes(
                                standard_name=standard_name
                            )
                        )
            elif ce.get("type", "") == "character":
                # Non-scalar character labels use sector(out_name, strlen).
                if "sector" in ds.variables:
                    allowed_coordinates.add("sector")
                else:
                    standard_name = ce.get("standard_name", "")
                    if standard_name:
                        allowed_coordinates.update(
                            var.name
                            for var in ds.get_variables_by_attributes(
                                standard_name=standard_name
                            )
                        )

        unexpected = [
            name for name in listed_coordinates if name not in allowed_coordinates
        ]
        if unexpected:
            ctx.add_failure(
                f"'{data_out_name}' coordinates attribute contains entries that "
                f"are not requested auxiliary or scalar coordinates: {unexpected}. "
                f"Allowed entries: {sorted(allowed_coordinates)}."
            )
        else:
            ctx.add_pass()

        return [ctx.to_result()]

    # ------------------------------------------------------------------
