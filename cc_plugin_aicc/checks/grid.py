"""Horizontal grid checks for AICC."""

from compliance_checker.base import BaseCheck, TestCtx
from compliance_checker.cf import util as cfutil

from cc_plugin_aicc.utils import _compare_units, _format_attribute, _ncattr
from cc_plugin_aicc.validation.attributes import (
    _check_coord_attrs,
    _check_coord_axis,
    _check_coord_type,
    _check_coord_units,
    _check_valid_range,
    _check_profile_direction,
    _check_strict_monotonicity,
)
from cc_plugin_aicc.validation.bounds import (
    _check_bounds_direction,
    _check_bounds_reference,
    _check_bounds_structure,
    _check_trailing_dimension,
)
from cc_plugin_aicc.validation.results import append_context_results


class GridChecks:
    """Rectilinear, curvilinear, and unstructured grid validation."""

    def check_grid(self, ds):
        """Verify horizontal coordinates for a registered grid topology."""
        has_lat = "latitude" in self.requested_dims
        has_lon = "longitude" in self.requested_dims
        if not (has_lat or has_lon):
            ctx = TestCtx(BaseCheck.HIGH, "[AICC002] Horizontal grid coordinates")
            ctx.add_pass()
            return [ctx.to_result()]

        lat_vars = cfutil.get_true_latitude_variables(ds)
        lon_vars = cfutil.get_true_longitude_variables(ds)

        if not self._grid_type_known:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC002] Horizontal grid type")
            ctx.add_failure(
                f"grid_label={_format_attribute(self._grid_label)} is not "
                f"registered in the global grid configuration. Add it to "
                f"DEFAULT_GRID_CONFIG or pass a custom registry through the "
                f"'grid_config' option."
            )
            return [ctx.to_result()]

        handler = self._grid_type_handlers.get(self._grid_type)
        if handler is None:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC002] Horizontal grid type")
            ctx.add_failure(
                f"Grid type '{self._grid_type}' is configured but not implemented. "
                f"Supported grid types: {sorted(self._grid_type_handlers)}."
            )
            return [ctx.to_result()]

        grid_entries = self._grid_table_entries(handler)
        return getattr(self, handler["check"])(ds, lat_vars, lon_vars, grid_entries)

    def _grid_table_entries(self, handler):
        """Return entries from the table assigned to a grid-type handler.

        Rectilinear grids use CMIP7_coordinate.json. All other grid types use
        CMIP7_grids.json by default, including future registered grid types.
        """
        if handler.get("table", "grids") == "coordinate":
            return self.CTcoords.get("axis_entry", {})
        return self.CTgrids.get("variable_entry", {})

    def _check_unstructured_grid(self, ds, lat_vars, lon_vars, grid_entries):
        """Auxiliary lat/lon (1-D cell dim) + vertex bounds."""
        results = []
        aux_coord_names = set(cfutil.get_auxiliary_coordinate_variables(ds))
        horizontal_coord_dims = {}

        for dim_id, cf_vars in (("latitude", lat_vars), ("longitude", lon_vars)):
            if dim_id not in self.requested_dims:
                continue

            ctx = TestCtx(BaseCheck.HIGH, f"[AICC002] {dim_id} auxiliary coordinate")
            low_ctx = TestCtx(
                BaseCheck.LOW,
                f"[AICC002] {dim_id} auxiliary coordinate (advisory)",
            )
            medium_ctx = TestCtx(
                BaseCheck.MEDIUM,
                f"[AICC002] {dim_id} auxiliary coordinate (recommended)",
            )
            grid_entry = grid_entries.get(dim_id, {})
            expected_units = grid_entry.get("units", "")
            vtx_key = f"vertices_{dim_id}"
            vtx_entry = grid_entries.get(vtx_key, {})
            vtx_out_name = vtx_entry.get("out_name", vtx_key)
            vtx_expected_units = vtx_entry.get("units", "")

            # Prefer variables that cfutil classified as auxiliary
            aux_matches = [v for v in cf_vars if v in aux_coord_names]
            if not aux_matches:
                aux_matches = cf_vars  # fall through; 1-D check will catch it

            if not aux_matches:
                ctx.add_failure(
                    f"No {dim_id} auxiliary coordinate variable found in the file "
                    f"(expected standard_name='{dim_id}' or matching units)."
                )
                results.append(ctx.to_result())
                continue
            ctx.add_pass()

            aux_var_name = aux_matches[0]
            aux_var = ds.variables[aux_var_name]

            # Attributes from CMIP7_grids.json
            _check_coord_attrs(
                ctx,
                low_ctx,
                aux_var,
                aux_var_name,
                grid_entry,
                medium_ctx=medium_ctx,
            )
            _check_coord_type(ctx, aux_var, aux_var_name, grid_entry)
            _check_valid_range(ctx, aux_var, aux_var_name, grid_entry)

            # Must be 1-D (single unstructured cell dimension)
            if aux_var.ndim != 1:
                ctx.add_failure(
                    f"'{aux_var_name}' must have exactly one dimension (cell/ncells) "
                    f"for an unstructured grid; found ndim={aux_var.ndim}."
                )
            else:
                ctx.add_pass()
                horizontal_coord_dims[dim_id] = aux_var.dimensions[0]

            # Must be auxiliary (not a dimension coordinate)
            if aux_var_name not in aux_coord_names:
                ctx.add_failure(
                    f"'{aux_var_name}' is a dimension coordinate; for an unstructured "
                    f"grid it must be an auxiliary coordinate referenced via "
                    f"the 'coordinates' attribute."
                )
            else:
                ctx.add_pass()

            # Units via cfutil-backed comparison
            level, msg = _compare_units(_ncattr(aux_var, "units"), expected_units)
            if level != "ok":
                ctx.add_failure(f"'{aux_var_name}' units: {msg}")
            else:
                ctx.add_pass()  # pass even for convertible (advisory only)

            results.append(ctx.to_result())
            append_context_results(results, medium_ctx, low_ctx)

            # --- Vertex bounds ---
            vtx_ctx = TestCtx(BaseCheck.HIGH, f"[AICC002] {vtx_key}")
            vtx_medium_ctx = TestCtx(
                BaseCheck.MEDIUM, f"[AICC002] {vtx_key} (recommended)"
            )
            vtx_low_ctx = TestCtx(BaseCheck.LOW, f"[AICC002] {vtx_key} (advisory)")
            vtx_var_name, vtx_var = _check_bounds_reference(
                vtx_ctx,
                vtx_medium_ctx,
                ds,
                aux_var,
                aux_var_name,
                recommended_name=vtx_out_name,
            )
            if vtx_var is None:
                results.append(vtx_ctx.to_result())
                append_context_results(results, vtx_medium_ctx, vtx_low_ctx)
                continue

            # Attributes from CMIP7_grids.json
            _check_coord_attrs(
                vtx_ctx,
                vtx_low_ctx,
                vtx_var,
                vtx_var_name,
                vtx_entry,
                missing_ok=True,
            )
            _check_coord_type(vtx_ctx, vtx_var, vtx_var_name, vtx_entry)

            valid_vertex_dims = (
                aux_var.ndim == 1
                and vtx_var.ndim == 2
                and vtx_var.dimensions[0] == aux_var.dimensions[0]
                and vtx_var.shape[0] == aux_var.shape[0]
                and vtx_var.shape[1] >= 3
            )
            if not valid_vertex_dims:
                vtx_ctx.add_failure(
                    f"'{vtx_var_name}' must use the auxiliary coordinate dimension "
                    f"{list(aux_var.dimensions)} followed by a vertex dimension of "
                    f"size at least 3; found dimensions "
                    f"{list(vtx_var.dimensions)} and shape {vtx_var.shape}."
                )
            else:
                vtx_ctx.add_pass()
            _check_trailing_dimension(
                vtx_medium_ctx,
                vtx_var,
                vtx_var_name,
                "vertices",
                description="Vertex variable",
            )

            vtx_units = _ncattr(vtx_var, "units")
            if not vtx_units:
                # CF bounds variables normally inherit coordinate metadata and
                # should not duplicate it. CMOR-listed vertex attributes are
                # therefore optional, but must be correct when supplied.
                vtx_ctx.add_pass()
            else:
                level, msg = _compare_units(vtx_units, vtx_expected_units)
                if level != "ok":
                    vtx_ctx.add_failure(f"'{vtx_var_name}' units: {msg}")
                else:
                    vtx_ctx.add_pass()

            results.append(vtx_ctx.to_result())
            append_context_results(results, vtx_medium_ctx, vtx_low_ctx)

        cell_dims = set(horizontal_coord_dims.values())
        shared_cell_dim = next(iter(cell_dims)) if len(cell_dims) == 1 else None
        if shared_cell_dim is not None:
            results.extend(self._check_unstructured_axis(ds, shared_cell_dim))

        # Data variable must list lat/lon in its 'coordinates' attribute
        data_out_name = self.var_entry.get("out_name", "") if self.var_entry else ""
        if data_out_name and data_out_name in ds.variables:
            data_var = ds.variables[data_out_name]
            coords_attr = _ncattr(data_var, "coordinates")
            coords_listed = coords_attr.split() if coords_attr else []

            if horizontal_coord_dims:
                ctx = TestCtx(
                    BaseCheck.HIGH,
                    f"[AICC002] '{data_out_name}' unstructured cell dimension",
                )
                if len(cell_dims) != 1:
                    details = ", ".join(
                        f"{dim_id}='{dim_name}'"
                        for dim_id, dim_name in horizontal_coord_dims.items()
                    )
                    ctx.add_failure(
                        f"Unstructured horizontal coordinates must share one cell "
                        f"dimension; found {details}."
                    )
                else:
                    if shared_cell_dim not in data_var.dimensions:
                        ctx.add_failure(
                            f"'{data_out_name}' must use the unstructured cell "
                            f"dimension '{shared_cell_dim}' used by its latitude/longitude "
                            f"coordinates; found dimensions "
                            f"{list(data_var.dimensions)}."
                        )
                    else:
                        ctx.add_pass()
                results.append(ctx.to_result())

            for dim_id, cf_vars in (("latitude", lat_vars), ("longitude", lon_vars)):
                if dim_id not in self.requested_dims:
                    continue
                ctx = TestCtx(
                    BaseCheck.HIGH,
                    f"[AICC002] '{data_out_name}' coordinates attribute ({dim_id})",
                )
                found = any(
                    _ncattr(ds.variables.get(v), "standard_name") == dim_id
                    for v in coords_listed
                )
                if not found:
                    ctx.add_failure(
                        f"'{data_out_name}' 'coordinates' attribute must include the "
                        f"{dim_id} auxiliary coordinate (standard_name='{dim_id}')."
                    )
                else:
                    ctx.add_pass()
                results.append(ctx.to_result())

        return results

    def _check_unstructured_axis(self, ds, cell_dimension):
        """Validate an optional explicit unstructured cell-index coordinate."""
        if cell_dimension not in ds.variables:
            return []  # a bare dimension is a valid implicit index

        ctx = TestCtx(BaseCheck.HIGH, "[AICC002] unstructured grid axis")
        medium_ctx = TestCtx(
            BaseCheck.MEDIUM,
            "[AICC002] unstructured grid axis (recommended)",
        )
        low_ctx = TestCtx(
            BaseCheck.LOW,
            "[AICC002] unstructured grid axis (advisory)",
        )
        var = ds.variables[cell_dimension]
        entries = self.CTgrids.get("axis_entry", {})
        candidates = [
            entry
            for identifier, entry in entries.items()
            if entry.get("out_name", identifier) == cell_dimension
        ]
        if not candidates:
            ctx.add_failure(
                f"Explicit unstructured dimension coordinate '{cell_dimension}' "
                "does not match any grid-axis table entry."
            )
            return [ctx.to_result()]

        def mismatch_score(entry):
            return sum(
                (
                    bool(entry.get("axis"))
                    and _ncattr(var, "axis") != entry.get("axis"),
                    bool(entry.get("standard_name"))
                    and _ncattr(var, "standard_name") != entry.get("standard_name"),
                    bool(entry.get("units"))
                    and _ncattr(var, "units") != entry.get("units"),
                )
            )

        entry = min(candidates, key=mismatch_score)
        if var.dimensions != (cell_dimension,):
            ctx.add_failure(
                f"Unstructured grid axis '{cell_dimension}' must be a 1-D "
                f"coordinate variable; found {list(var.dimensions)}."
            )
        else:
            ctx.add_pass()
        _check_coord_attrs(
            ctx, low_ctx, var, cell_dimension, entry, medium_ctx=medium_ctx
        )
        _check_coord_type(ctx, var, cell_dimension, entry)
        stored_direction = entry.get("stored_direction", "")
        if stored_direction:
            _check_profile_direction(
                ctx,
                var[:],
                stored_direction,
                f"Grid axis '{cell_dimension}'",
                source_ndim=var.ndim,
            )
        else:
            _check_strict_monotonicity(
                ctx,
                var[:],
                f"Grid axis '{cell_dimension}'",
                source_ndim=var.ndim,
            )

        results = [ctx.to_result()]
        append_context_results(results, medium_ctx, low_ctx)
        return results

    def _check_curvilinear_grid(self, ds, lat_vars, lon_vars, grid_entries):
        """Validate 2-D auxiliary lon/lat, vertices, and optional grid axes."""
        results = []
        coordinate_dims = {}

        for dim_id, cf_vars in (("latitude", lat_vars), ("longitude", lon_vars)):
            if dim_id not in self.requested_dims:
                continue
            entry = grid_entries.get(dim_id, {})
            out_name = entry.get("out_name", dim_id)
            ctx = TestCtx(BaseCheck.HIGH, f"[AICC002] curvilinear {dim_id}")
            medium_ctx = TestCtx(
                BaseCheck.MEDIUM,
                f"[AICC002] curvilinear {dim_id} (recommended)",
            )
            low_ctx = TestCtx(
                BaseCheck.LOW, f"[AICC002] curvilinear {dim_id} (advisory)"
            )

            name = out_name if out_name in ds.variables else next(iter(cf_vars), None)
            if name is None:
                ctx.add_failure(
                    f"Curvilinear {dim_id} variable '{out_name}' not found."
                )
                results.append(ctx.to_result())
                continue
            if name != out_name:
                ctx.add_failure(
                    f"Curvilinear {dim_id} variable '{name}' was identified by CF "
                    f"metadata, but CMIP7_grids.json requires '{out_name}'."
                )
            else:
                ctx.add_pass()

            var = ds.variables[name]
            if var.ndim != 2:
                ctx.add_failure(
                    f"Curvilinear {dim_id} coordinate '{name}' must be 2-D; found "
                    f"dimensions {list(var.dimensions)}."
                )
            else:
                ctx.add_pass()
                coordinate_dims[dim_id] = var.dimensions

            _check_coord_attrs(ctx, low_ctx, var, name, entry, medium_ctx=medium_ctx)
            _check_coord_type(ctx, var, name, entry)
            _check_coord_units(ctx, low_ctx, var, name, entry)
            _check_valid_range(ctx, var, name, entry)

            vertex_key = f"vertices_{dim_id}"
            vertex_entry = grid_entries.get(vertex_key, {})
            recommended_vertex_name = vertex_entry.get("out_name", vertex_key)
            vertex_name, vertex = _check_bounds_reference(
                ctx,
                medium_ctx,
                ds,
                var,
                name,
                recommended_name=recommended_vertex_name,
            )
            if vertex is not None:
                _check_coord_attrs(
                    ctx,
                    low_ctx,
                    vertex,
                    vertex_name,
                    vertex_entry,
                    missing_ok=True,
                    medium_ctx=medium_ctx,
                )
                _check_coord_type(ctx, vertex, vertex_name, vertex_entry)
                if _ncattr(vertex, "units"):
                    _check_coord_units(ctx, low_ctx, vertex, vertex_name, vertex_entry)
                else:
                    # Bounds inherit units from their coordinate under CF.
                    ctx.add_pass()
                _check_valid_range(ctx, vertex, vertex_name, vertex_entry)
                valid_vertex_dims = (
                    var.ndim == 2
                    and vertex.ndim == 3
                    and vertex.dimensions[:2] == var.dimensions
                    and vertex.shape[:2] == var.shape
                    and vertex.shape[2] >= 3
                )
                if not valid_vertex_dims:
                    ctx.add_failure(
                        f"Curvilinear vertex variable '{vertex_name}' must use the "
                        f"coordinate's two dimensions followed by a vertex dimension "
                        f"of size at least 3; found dimensions "
                        f"{list(vertex.dimensions)} and shape {vertex.shape}."
                    )
                else:
                    ctx.add_pass()
                _check_trailing_dimension(
                    medium_ctx,
                    vertex,
                    vertex_name,
                    "vertices",
                    description="Vertex variable",
                )

            results.append(ctx.to_result())
            append_context_results(results, medium_ctx, low_ctx)

        if len(coordinate_dims) == 2:
            shared_ctx = TestCtx(
                BaseCheck.HIGH, "[AICC002] curvilinear horizontal dimensions"
            )
            if coordinate_dims["latitude"] != coordinate_dims["longitude"]:
                shared_ctx.add_failure(
                    "Curvilinear latitude and longitude must share the same two "
                    f"dimensions; found latitude{coordinate_dims['latitude']} and "
                    f"longitude{coordinate_dims['longitude']}."
                )
            else:
                shared_ctx.add_pass()
                results.extend(
                    self._check_curvilinear_axes(ds, coordinate_dims["latitude"])
                )
            results.append(shared_ctx.to_result())

        data_out_name = self.var_entry.get("out_name", "") if self.var_entry else ""
        if data_out_name and data_out_name in ds.variables:
            ctx = TestCtx(
                BaseCheck.HIGH,
                f"[AICC002] '{data_out_name}' curvilinear coordinates attribute",
            )
            listed = _ncattr(ds.variables[data_out_name], "coordinates")
            listed = listed.split() if listed else []
            for dim_id in ("latitude", "longitude"):
                if dim_id not in self.requested_dims:
                    continue
                expected = grid_entries.get(dim_id, {}).get("out_name", dim_id)
                if expected not in listed:
                    ctx.add_failure(
                        f"'{data_out_name}' coordinates attribute must include "
                        f"curvilinear coordinate '{expected}'."
                    )
                else:
                    ctx.add_pass()
            results.append(ctx.to_result())

        return results

    def _check_curvilinear_axes(self, ds, horizontal_dims):
        """Validate one supported optional curvilinear dimension-axis scheme."""
        ctx = TestCtx(BaseCheck.HIGH, "[AICC002] curvilinear grid axes")
        medium_ctx = TestCtx(
            BaseCheck.MEDIUM, "[AICC002] curvilinear grid axes (recommended)"
        )
        low_ctx = TestCtx(BaseCheck.LOW, "[AICC002] curvilinear grid axes (advisory)")
        entries = self.CTgrids.get("axis_entry", {})

        def by_standard_name(key):
            standard_name = entries.get(key, {}).get("standard_name", "")
            return next(
                (
                    name
                    for name in horizontal_dims
                    if name in ds.variables
                    and _ncattr(ds.variables[name], "standard_name") == standard_name
                ),
                None,
            )

        schemes = []
        rotated = []
        for key in ("grid_latitude", "grid_longitude"):
            name = entries.get(key, {}).get("out_name", "")
            if name in horizontal_dims and name in ds.variables:
                rotated.append((name, key))
        if len(rotated) == 2:
            schemes.append(("rotated latitude/longitude", rotated))

        for label, keys in (
            ("projected x/y", ("y", "x")),
            ("angular projected x/y", ("y_deg", "x_deg")),
        ):
            matched = [(by_standard_name(key), key) for key in keys]
            if all(name is not None for name, _ in matched):
                schemes.append((label, matched))

        index_entries = {
            entry.get("out_name", ""): key
            for key, entry in entries.items()
            if key.endswith("_index") and entry.get("out_name", "")
        }
        index_matches = [
            (name, index_entries[name])
            for name in horizontal_dims
            if name in ds.variables and name in index_entries
        ]
        if len(index_matches) == 2:
            schemes.append(("explicit index axes", index_matches))

        dimension_coordinate_names = [
            name
            for name in horizontal_dims
            if name in ds.variables and ds.variables[name].dimensions == (name,)
        ]
        if not schemes and not dimension_coordinate_names:
            # Bare dimensions are a valid implicit-index representation.
            ctx.add_pass()
            return [ctx.to_result()]
        if not schemes:
            ctx.add_failure(
                "Curvilinear horizontal dimension coordinates do not match any "
                "supported CMIP7_grids.json scheme: rlat/rlon, projected x/y, "
                "angular projected x/y, or index axes."
            )
            return [ctx.to_result()]

        grid_mapping_name = ""
        data_out_name = self.var_entry.get("out_name", "") if self.var_entry else ""
        if data_out_name in ds.variables:
            mapping_attribute = _ncattr(ds.variables[data_out_name], "grid_mapping")
            # CF permits ``mapping: coordinate-list`` syntax. The first token
            # still names the grid-mapping variable.
            mapping_name = (
                mapping_attribute.split()[0].rstrip(":")
                if isinstance(mapping_attribute, str) and mapping_attribute
                else ""
            )
            if mapping_name in ds.variables:
                grid_mapping_name = _ncattr(
                    ds.variables[mapping_name], "grid_mapping_name"
                )
        if grid_mapping_name == "rotated_latitude_longitude":
            selected = next(
                (scheme for scheme in schemes if scheme[0].startswith("rotated")), None
            )
            if selected is None:
                ctx.add_failure(
                    "grid_mapping_name='rotated_latitude_longitude' requires the "
                    "rlat/rlon axis representation."
                )
                return [ctx.to_result()]
        elif grid_mapping_name and grid_mapping_name != "latitude_longitude":
            selected = next(
                (
                    scheme
                    for scheme in schemes
                    if scheme[0] in {"projected x/y", "angular projected x/y"}
                ),
                None,
            )
            if selected is None:
                ctx.add_failure(
                    f"grid_mapping_name='{grid_mapping_name}' requires projected "
                    "x/y axes in metres or degrees."
                )
                return [ctx.to_result()]
        else:
            selected = schemes[0]

        label, variables = selected
        ctx.add_pass()
        for name, key in variables:
            var = ds.variables[name]
            entry = entries[key]
            if var.dimensions != (name,):
                ctx.add_failure(
                    f"Curvilinear {label} axis '{name}' must be a 1-D coordinate "
                    f"variable; found {list(var.dimensions)}."
                )
            else:
                ctx.add_pass()
            _check_coord_attrs(ctx, low_ctx, var, name, entry, medium_ctx=medium_ctx)
            _check_coord_axis(ctx, low_ctx, var, name, entry)
            _check_coord_type(ctx, var, name, entry)
            _check_coord_units(ctx, low_ctx, var, name, entry)
            stored_direction = entry.get("stored_direction", "")
            if stored_direction:
                _check_profile_direction(
                    ctx,
                    var[:],
                    stored_direction,
                    f"Grid axis '{name}'",
                    source_ndim=var.ndim,
                )
            else:
                _check_strict_monotonicity(
                    ctx,
                    var[:],
                    f"Grid axis '{name}'",
                    source_ndim=var.ndim,
                )

        results = [ctx.to_result()]
        append_context_results(results, medium_ctx, low_ctx)
        return results

    def _check_rectilinear_grid(self, ds, lat_vars, lon_vars, grid_entries):
        """Dimension coordinate lat(lat)/lon(lon) + regular cell bounds."""
        results = []
        dim_coord_names = set(cfutil.get_coordinate_variables(ds))

        for dim_id, cf_vars, expected_axis in (
            ("latitude", lat_vars, "Y"),
            ("longitude", lon_vars, "X"),
        ):
            if dim_id not in self.requested_dims:
                continue

            ce = grid_entries.get(dim_id, {})
            expected_out_name = ce.get("out_name", dim_id)
            expected_sn = ce.get("standard_name", dim_id)
            expected_units = ce.get("units", "")
            must_have_bounds = ce.get("must_have_bounds", "no") == "yes"

            ctx = TestCtx(BaseCheck.HIGH, f"[AICC002] {dim_id} dimension coordinate")
            low_ctx = TestCtx(
                BaseCheck.LOW,
                f"[AICC002] {dim_id} dimension coordinate (advisory)",
            )
            medium_ctx = TestCtx(
                BaseCheck.MEDIUM,
                f"[AICC002] {dim_id} dimension coordinate (recommended)",
            )

            dim_matches = [name for name in cf_vars if name in dim_coord_names]
            if not dim_matches:
                dim_matches = cf_vars
            if not dim_matches:
                ctx.add_failure(
                    f"No {dim_id} coordinate variable found for rectilinear grid."
                )
                results.append(ctx.to_result())
                continue

            var_name = (
                expected_out_name
                if expected_out_name in dim_matches
                else dim_matches[0]
            )
            var = ds.variables[var_name]

            if var_name != expected_out_name:
                actual_signature = f"{var_name}({', '.join(var.dimensions)})"
                ctx.add_failure(
                    f"Rectilinear {dim_id} coordinate '{actual_signature}' was "
                    f"identified by standard_name='{expected_sn}', but CMOR requires "
                    f"'{expected_out_name}({expected_out_name})'."
                )
            else:
                ctx.add_pass()

            expected_dims = (expected_out_name,)
            if var.dimensions != expected_dims:
                ctx.add_failure(
                    f"Rectilinear {dim_id} coordinate '{var_name}' has dimensions "
                    f"{list(var.dimensions)}; expected "
                    f"'{expected_out_name}({expected_out_name})'."
                )
            else:
                ctx.add_pass()

            if _ncattr(var, "axis") != expected_axis:
                ctx.add_failure(f"'{var_name}' must have axis='{expected_axis}'.")
            else:
                ctx.add_pass()

            # Attributes from CMIP7_coordinate.json
            _check_coord_attrs(ctx, low_ctx, var, var_name, ce, medium_ctx=medium_ctx)
            _check_coord_type(ctx, var, var_name, ce)
            _check_valid_range(ctx, var, var_name, ce)

            level, msg = _compare_units(_ncattr(var, "units"), expected_units)
            if level != "ok":
                ctx.add_failure(f"'{var_name}' units: {msg}")
            else:
                ctx.add_pass()

            results.append(ctx.to_result())

            if must_have_bounds or bool(_ncattr(var, "bounds")):
                bnds_ctx = TestCtx(BaseCheck.HIGH, f"[AICC002] {dim_id} bounds")
                bnds_name, bnds_var = _check_bounds_reference(
                    bnds_ctx,
                    medium_ctx,
                    ds,
                    var,
                    var_name,
                    expected_out_name,
                )
                if bnds_var is not None:
                    if bnds_var.ncattrs():
                        bnds_ctx.add_failure(
                            f"'{bnds_name}' must have no attributes; "
                            f"found {list(bnds_var.ncattrs())}."
                        )
                    else:
                        bnds_ctx.add_pass()
                    _check_bounds_structure(
                        bnds_ctx, medium_ctx, bnds_var, bnds_name, var, ce
                    )
                    _check_bounds_direction(
                        bnds_ctx,
                        var,
                        var_name,
                        bnds_var,
                        bnds_name,
                        ce.get("stored_direction", ""),
                    )
                results.append(bnds_ctx.to_result())

            append_context_results(results, medium_ctx, low_ctx)

        return results

    def _expected_unstructured_horizontal_dimensions(
        self, ds, horizontal_dim_ids, grid_entries
    ):
        """Return the single arbitrary cell-dimension placeholder."""
        return ["<ncells>"]

    def _expected_rectilinear_horizontal_dimensions(
        self, ds, horizontal_dim_ids, grid_entries
    ):
        """Return CMOR rectilinear dimensions in expected C order."""
        return [
            grid_entries.get(dim_id, {}).get("out_name", dim_id)
            for dim_id in reversed(horizontal_dim_ids)
        ]

    def _expected_curvilinear_horizontal_dimensions(
        self, ds, horizontal_dim_ids, grid_entries
    ):
        """Return the shared two-dimensional latitude/longitude dimensions."""
        for dim_id in ("latitude", "longitude"):
            if dim_id not in horizontal_dim_ids:
                continue
            name = grid_entries.get(dim_id, {}).get("out_name", dim_id)
            if name in ds.variables and ds.variables[name].ndim == 2:
                return list(ds.variables[name].dimensions)
        return []

    # ------------------------------------------------------------------
