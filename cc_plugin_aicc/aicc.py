"""
aicc.py — AWI ICON Coordinate Checker (AICC) / AI Compliance Checker

Verifies CMIP7 coordinate compliance for AWI and ICON model output.

Variable discovery delegates to compliance_checker.cf.util (cfutil) for
consistent CF-convention application. Configuration lives in config.py;
pure utility functions live in utils.py.
"""

import ast
import json
import os
import re
from pathlib import Path

import numpy as np
from compliance_checker.base import BaseCheck, BaseNCCheck, Result, TestCtx
from compliance_checker.cf import util as cfutil

from cc_plugin_aicc import __version__
from cc_plugin_aicc.config import (
    DEFAULT_TABLES_PATH,
    HORIZONTAL_DIM_IDS,
    REALM_TO_TABLE,
    VERTICAL_GENERIC_IDS,
    load_grid_config,
    load_model_config,
    resolve_grid_type,
    resolve_model_config,
)
from cc_plugin_aicc.utils import (
    _as_list,
    _cmor_tol_val,
    _compare_units,
    _decode_char_scalar,
    _decode_char_var,
    _find_formula_entry,
    _format_attribute,
    _is_scalar_coord,
    _is_time_dim,
    _ncattr,
    _neutral_dtype,
    _parse_formula_terms,
)

# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class AICC(BaseNCCheck, BaseCheck):
    """AWI ICON Coordinate Checker for CMIP7 model output."""

    register_checker = True
    _cc_spec = "aicc"
    _cc_spec_version = __version__
    _cc_description = (
        "AWI ICON Coordinate Checks (AICC) — verifies CMIP7 coordinate compliance "
        "for configured model grids."
    )
    _cc_url = ""
    _cc_display_headers = {3: "Required", 2: "Recommended", 1: "Suggested"}
    # Adding a grid type requires a validator, a dimension resolver, and one
    # registry entry here. Central dispatch and dimension ordering stay generic.
    _grid_type_handlers = {
        "unstructured": {
            "check": "_check_unstructured_grid",
            "dimensions": "_expected_unstructured_horizontal_dimensions",
            "table": "grids",
        },
        "rectilinear": {
            "check": "_check_rectilinear_grid",
            "dimensions": "_expected_rectilinear_horizontal_dimensions",
            "table": "coordinate",
        },
    }

    def __init__(self, options=None):
        BaseCheck.__init__(self, options)

    @classmethod
    def make_result(cls, level, score, out_of, name, messages):
        return Result(level, (score, out_of), name, messages)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, dataset):
        self.ds = dataset  # netCDF4.Dataset — used by all check methods

        # Read CMIP7 CMOR tables
        tables_path = self.options.get("tables", DEFAULT_TABLES_PATH)
        self._read_cmip7_tables(tables_path)

        # Resolve the model-specific vertical mapping.
        source_id = _ncattr(dataset, "source_id")
        mc_opt = self.options.get("model_config", self.options.get("vertical_config"))
        model_config = load_model_config(mc_opt)

        self._conf_key, model_conf = resolve_model_config(source_id, model_config)
        if model_conf:
            # Support both nested and legacy flat vertical configurations.
            if "vertical" in model_conf:
                self._vert_mapping = model_conf.get("vertical") or {}
            else:
                self._vert_mapping = model_conf
        else:
            self._vert_mapping = None

        # Resolve the globally registered grid_label independently of source_id.
        self._grid_label = _ncattr(dataset, "grid_label")
        grid_config = load_grid_config(self.options.get("grid_config"))
        self._grid_type, self._grid_type_known = resolve_grid_type(
            self._grid_label, grid_config
        )

        # Identify variable and its CMOR table entry
        self.branded_variable = _ncattr(dataset, "branded_variable") or None
        self.var_entry = None
        self.table_name = None
        self.requested_dims = []

        if self.branded_variable:
            self._resolve_table_and_variable(dataset)

        if self.var_entry is not None:
            self.requested_dims = self.var_entry.get("dimensions", [])

    # ------------------------------------------------------------------
    # Table I/O
    # ------------------------------------------------------------------

    def _read_cmip7_tables(self, tables_path):
        """Read all CMIP7 CMOR tables from *tables_path*."""
        tables_path = os.path.normpath(os.path.expanduser(str(tables_path)))
        if not os.path.isdir(tables_path):
            raise FileNotFoundError(
                f"CMIP7 tables directory not found: '{tables_path}'"
            )
        json_files = sorted(
            f
            for f in os.listdir(tables_path)
            if f.endswith(".json") and not f.startswith(".")
        )
        if not json_files:
            raise FileNotFoundError(
                f"No CMIP7 JSON table files found in: '{tables_path}'"
            )

        prefixes = {f.split("_")[0] for f in json_files}
        if len(prefixes) != 1:
            raise ValueError(
                f"Expected a single table prefix in '{tables_path}', "
                f"found: {sorted(prefixes)}"
            )
        self._table_prefix = prefixes.pop()

        def _load(name):
            path = Path(tables_path, f"{self._table_prefix}_{name}.json")
            if not path.exists():
                raise FileNotFoundError(f"Required CMIP7 table not found: '{path}'")
            with open(path) as fh:
                return json.load(fh)

        self.CTcoords = _load("coordinate")
        self.CTgrids = _load("grids")
        self.CTformulas = _load("formula_terms")

        self.CT = {}
        for fname in json_files:
            tname = "_".join(fname.split("_")[1:]).rsplit(".", 1)[0]
            if tname in ("coordinate", "grids", "formula_terms"):
                continue
            data = json.load(open(Path(tables_path, fname)))
            if "variable_entry" in data:
                self.CT[tname] = data

    # ------------------------------------------------------------------
    # Variable / table resolution
    # ------------------------------------------------------------------

    def _resolve_table_and_variable(self, ds):
        """Resolve *branded_variable* to a var_entry in the appropriate CMOR table."""
        table_id = _ncattr(ds, "table_id")
        if table_id and table_id in self.CT:
            candidates = [table_id]
        else:
            realm_raw = _ncattr(ds, "realm")
            realm_first = realm_raw.split()[0] if realm_raw else ""
            mapped = REALM_TO_TABLE.get(realm_first)
            candidates = [mapped] if mapped and mapped in self.CT else list(self.CT)

        for tname in candidates:
            var_entries = self.CT[tname].get("variable_entry", {})
            if self.branded_variable in var_entries:
                self.table_name = tname
                self.var_entry = var_entries[self.branded_variable]
                return

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check_branded_variable(self, ds):
        """Verify branded_variable global attribute is set and resolvable."""
        ctx = TestCtx(BaseCheck.HIGH, "[AICC001] branded_variable identification")

        if not self.branded_variable:
            ctx.add_failure(
                "Global attribute 'branded_variable' is not set. "
                "Cannot identify the variable for CMOR table lookup."
            )
            return [ctx.to_result()]

        if self.var_entry is None:
            ctx.add_failure(
                f"branded_variable {_format_attribute(self.branded_variable)} "
                f"not found in any "
                f"CMIP7 table (tried table_id="
                f"{_format_attribute(_ncattr(ds, 'table_id'))}, realm="
                f"{_format_attribute(_ncattr(ds, 'realm'))})."
            )
            return [ctx.to_result()]

        ctx.add_pass()
        return [ctx.to_result()]

    # ------------------------------------------------------------------

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
        return getattr(self, handler["check"])(
            ds, lat_vars, lon_vars, grid_entries
        )

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
            _check_coord_attrs(ctx, low_ctx, aux_var, aux_var_name, grid_entry)

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
            if low_ctx.messages:
                results.append(low_ctx.to_result())

            # --- Vertex bounds ---
            vtx_ctx = TestCtx(BaseCheck.HIGH, f"[AICC002] {vtx_key}")
            vtx_low_ctx = TestCtx(
                BaseCheck.LOW, f"[AICC002] {vtx_key} (advisory)"
            )
            # Look for the vertices variable by exact out_name or 'vertices_*' naming
            vtx_var_name = vtx_out_name if vtx_out_name in ds.variables else next(
                (v for v in ds.variables
                 if v == vtx_out_name or (
                     "vertices" in v.lower()
                     and dim_id in v.lower())),
                None,
            )
            if vtx_var_name is None:
                vtx_ctx.add_failure(
                    f"Vertex bounds variable '{vtx_out_name}' not found for '{dim_id}'."
                )
                results.append(vtx_ctx.to_result())
                continue
            vtx_ctx.add_pass()

            vtx_var = ds.variables[vtx_var_name]

            # Attributes from CMIP7_grids.json
            _check_coord_attrs(
                vtx_ctx, vtx_low_ctx, vtx_var, vtx_var_name, vtx_entry
            )

            if vtx_var.ndim != 2:
                vtx_ctx.add_failure(
                    f"'{vtx_var_name}' must have 2 dimensions (ncells, ncorners); "
                    f"found ndim={vtx_var.ndim}."
                )
            else:
                vtx_ctx.add_pass()

            level, msg = _compare_units(_ncattr(vtx_var, "units"), vtx_expected_units)
            if level != "ok":
                vtx_ctx.add_failure(f"'{vtx_var_name}' units: {msg}")
            else:
                vtx_ctx.add_pass()

            results.append(vtx_ctx.to_result())
            if vtx_low_ctx.messages:
                results.append(vtx_low_ctx.to_result())

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
                cell_dims = set(horizontal_coord_dims.values())
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
                    cell_dim = next(iter(cell_dims))
                    if cell_dim not in data_var.dimensions:
                        ctx.add_failure(
                            f"'{data_out_name}' must use the unstructured cell "
                            f"dimension '{cell_dim}' used by its latitude/longitude "
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

            dim_matches = [
                name for name in cf_vars if name in dim_coord_names
            ]
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
                actual_signature = (
                    f"{var_name}({', '.join(var.dimensions)})"
                )
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
            _check_coord_attrs(ctx, low_ctx, var, var_name, ce)

            level, msg = _compare_units(_ncattr(var, "units"), expected_units)
            if level != "ok":
                ctx.add_failure(f"'{var_name}' units: {msg}")
            else:
                ctx.add_pass()

            results.append(ctx.to_result())
            if low_ctx.messages:
                results.append(low_ctx.to_result())

            if must_have_bounds:
                expected_bnds_name = f"{expected_out_name}_bnds"
                declared_bnds = _ncattr(var, "bounds")
                bnds_ctx = TestCtx(BaseCheck.HIGH, f"[AICC002] {dim_id} bounds")

                if declared_bnds != expected_bnds_name:
                    bnds_ctx.add_failure(
                        f"'{var_name}' bounds={_format_attribute(declared_bnds)}; "
                        f"expected "
                        f"'{expected_bnds_name}'."
                    )
                else:
                    bnds_ctx.add_pass()

                if expected_bnds_name not in ds.variables:
                    bnds_ctx.add_failure(
                        f"Bounds variable '{expected_bnds_name}' not found."
                    )
                else:
                    bnds_ctx.add_pass()
                    bnds_var = ds.variables[expected_bnds_name]
                    if bnds_var.ncattrs():
                        bnds_ctx.add_failure(
                            f"'{expected_bnds_name}' must have no attributes; "
                            f"found {list(bnds_var.ncattrs())}."
                        )
                    else:
                        bnds_ctx.add_pass()
                    if bnds_var.ndim != 2 or bnds_var.shape[1] != 2:
                        bnds_ctx.add_failure(
                            f"'{expected_bnds_name}' must have shape (n, 2); "
                            f"found shape {bnds_var.shape}."
                        )
                    else:
                        bnds_ctx.add_pass()
                results.append(bnds_ctx.to_result())

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

    # ------------------------------------------------------------------

    def check_vertical(self, ds):
        """Verify vertical coordinate(s) against the CMIP7 coordinate table."""
        vert_dims = [d for d in self.requested_dims if d in VERTICAL_GENERIC_IDS]
        if not vert_dims:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC003] Vertical coordinates")
            ctx.add_pass()
            return [ctx.to_result()]

        if self._vert_mapping is None:
            ctx = _vertical_config_ctx(ds, "[AICC003] Vertical coordinates")
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

            ce = axis_entries.get(coord_key, {})
            out_name = ce.get("out_name", "lev")
            expected_sn = ce.get("standard_name", "")
            expected_units = ce.get("units", "")
            expected_positive = ce.get("positive", "")
            must_have_bounds = ce.get("must_have_bounds", "no") == "yes"
            z_factors_str = ce.get("z_factors", "")
            z_bounds_factors_str = ce.get("z_bounds_factors", "")

            ctx = TestCtx(
                BaseCheck.HIGH,
                f"[AICC003] Vertical coordinate '{out_name}' ({generic_id})",
            )
            low_ctx = TestCtx(
                BaseCheck.LOW,
                f"[AICC003] Vertical coordinate '{out_name}' "
                f"({generic_id}, advisory)",
            )

            # Locate lev variable: exact out_name, then cfutil Z vars by standard_name
            if out_name in ds.variables:
                lev_var_name = out_name
            else:
                lev_var_name = next(
                    (v for v in z_var_names
                     if expected_sn
                     and _ncattr(ds.variables[v], "standard_name") == expected_sn),
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

            # axis=Z
            if _ncattr(lev_var, "axis") != "Z":
                ctx.add_failure(f"'{lev_var_name}' must have attribute axis='Z'.")
            else:
                ctx.add_pass()

            # Attributes from CMIP7_coordinate.json
            _check_coord_attrs(ctx, low_ctx, lev_var, lev_var_name, ce)

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
            if must_have_bounds:
                bnds_name = f"{lev_var_name}_bnds"
                declared_bnds = _ncattr(lev_var, "bounds")
                if declared_bnds != bnds_name:
                    ctx.add_failure(
                        f"'{lev_var_name}' bounds="
                        f"{_format_attribute(declared_bnds)}; "
                        f"expected '{bnds_name}'."
                    )
                else:
                    ctx.add_pass()

                if bnds_name not in ds.variables:
                    ctx.add_failure(
                        f"Bounds variable '{bnds_name}' for '{lev_var_name}' not found."
                    )
                else:
                    ctx.add_pass()
                    bnds_var = ds.variables[bnds_name]
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
                                formula_entries, var_name, generic_id
                            )
                            if ft_entry:
                                _check_formula_var_attrs(
                                    ctx,
                                    low_ctx,
                                    ds.variables[var_name],
                                    var_name,
                                    ft_entry,
                                )

            # bounds formula terms
            if must_have_bounds and z_bounds_factors_str:
                zbf_map = _parse_formula_terms(z_bounds_factors_str)
                for term, var_name in zbf_map.items():
                    if var_name not in ds.variables:
                        ctx.add_failure(
                            f"bounds formula_terms: term '{term}' references "
                            f"'{var_name}' which does not exist."
                        )
                    else:
                        ctx.add_pass()
                        ft_entry = _find_formula_entry(
                            formula_entries, var_name, generic_id
                        )
                        if ft_entry:
                            _check_formula_var_attrs(
                                ctx,
                                low_ctx,
                                ds.variables[var_name],
                                var_name,
                                ft_entry,
                            )

            results.append(ctx.to_result())
            if low_ctx.messages:
                results.append(low_ctx.to_result())

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
            ctx = _vertical_config_ctx(ds, "[AICC003b] Vertical direction")
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
                expected_formula_terms = _parse_formula_terms(
                    ce.get("z_factors", "")
                )
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
                                f"Formula-derived profile for '{var_name}' at "
                                f"{sample}",
                            )
            elif table_standard_name in _DIRECT_VERTICAL_STANDARD_NAMES:
                _check_direct_vertical_values(
                    ctx, coord_var[:], var_name, table_standard_name
                )

            results.append(ctx.to_result())

        return results

    # ------------------------------------------------------------------

    def check_time(self, ds):
        """Verify time coordinate(s) against the CMIP7 coordinate table."""
        time_dims = [d for d in self.requested_dims if _is_time_dim(d)]
        if not time_dims:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC004] Time coordinate")
            ctx.add_pass()
            return [ctx.to_result()]

        results = []
        axis_entries = self.CTcoords.get("axis_entry", {})

        # Use cfutil to find the time variable (CF-aware, not name-based)
        t_var_name = cfutil.get_time_variable(ds)

        for time_dim_id in time_dims:
            ce = axis_entries.get(time_dim_id, {})
            out_name = ce.get("out_name", "time")
            must_have_bounds = ce.get("must_have_bounds", "no") == "yes"
            climatology_setting = ce.get("climatology", "")
            is_climatology = time_dim_id != "time4" and (
                climatology_setting is True
                or str(climatology_setting).lower() == "yes"
            )

            ctx = TestCtx(
                BaseCheck.HIGH, f"[AICC004] Time coordinate ({time_dim_id})"
            )
            low_ctx = TestCtx(
                BaseCheck.LOW,
                f"[AICC004] Time coordinate ({time_dim_id}, advisory)",
            )

            data_out_name = (
                self.var_entry.get("out_name", "") if self.var_entry else ""
            )
            if out_name not in ds.dimensions:
                ctx.add_failure(
                    f"Required time dimension '{out_name}' "
                    f"(dim_id='{time_dim_id}') not found in file."
                )
            elif (
                data_out_name
                and data_out_name in ds.variables
                and out_name not in ds.variables[data_out_name].dimensions
            ):
                ctx.add_failure(
                    f"Time-dependent variable '{data_out_name}' must use "
                    f"dimension '{out_name}'; found dimensions "
                    f"{list(ds.variables[data_out_name].dimensions)}."
                )
            else:
                ctx.add_pass()

            resolved_t = t_var_name or (out_name if out_name in ds.variables else None)
            if resolved_t is None:
                ctx.add_failure(
                    f"Time coordinate variable '{out_name}' (dim_id='{time_dim_id}') "
                    f"not found in file."
                )
                results.append(ctx.to_result())
                continue
            ctx.add_pass()

            t_var = ds.variables[resolved_t]

            # axis=T
            if _ncattr(t_var, "axis") != "T":
                ctx.add_failure(f"'{resolved_t}' must have attribute axis='T'.")
            else:
                ctx.add_pass()

            # Attributes from CMIP7_coordinate.json
            _check_coord_attrs(ctx, low_ctx, t_var, resolved_t, ce)

            # Units must match the CMOR template exactly (reference date is free).
            units = _ncattr(t_var, "units")
            level, msg = _compare_units(units, ce.get("units", ""))
            if level != "ok":
                ctx.add_failure(f"'{resolved_t}' units: {msg}")
            else:
                ctx.add_pass()

            # calendar attribute
            if not _ncattr(t_var, "calendar"):
                ctx.add_failure(f"'{resolved_t}' is missing 'calendar' attribute.")
            else:
                ctx.add_pass()

            if must_have_bounds:
                if is_climatology:
                    # CF §7.4: time must carry a 'climatology' attribute
                    clim_attr = _ncattr(t_var, "climatology")
                    if not clim_attr:
                        ctx.add_failure(
                            f"Climatology time variable '{resolved_t}' must have a "
                            f"'climatology' attribute pointing to the bounds variable."
                        )
                    else:
                        ctx.add_pass()
                        # verify the bounds variable exists (cfutil agrees)
                        clim_bnds = cfutil.get_climatology_variable(ds)
                        if clim_attr not in ds.variables:
                            ctx.add_failure(
                                f"Climatology bounds variable "
                                f"{_format_attribute(clim_attr)} "
                                f"(referenced by '{resolved_t}:climatology') not found."
                            )
                        else:
                            ctx.add_pass()
                else:
                    # Regular time bounds
                    bnds_name = f"{out_name}_bnds"
                    declared_bnds = _ncattr(t_var, "bounds")
                    if declared_bnds != bnds_name:
                        ctx.add_failure(
                            f"'{resolved_t}' bounds="
                            f"{_format_attribute(declared_bnds)}; "
                            f"expected '{bnds_name}'."
                        )
                    else:
                        ctx.add_pass()

                    if bnds_name not in ds.variables:
                        ctx.add_failure(
                            f"Time bounds variable '{bnds_name}' not found in file."
                        )
                    else:
                        ctx.add_pass()
                        bnds_var = ds.variables[bnds_name]
                        if bnds_var.ncattrs():
                            ctx.add_failure(
                                f"'{bnds_name}' must have no attributes; "
                                f"found: {list(bnds_var.ncattrs())}."
                            )
                        else:
                            ctx.add_pass()

            results.append(ctx.to_result())
            if low_ctx.messages:
                results.append(low_ctx.to_result())

        return results

    # ------------------------------------------------------------------

    def check_coord(self, ds):
        """Verify non-grid, non-vertical, non-time coordinate dimensions."""
        other_dims = [
            d for d in self.requested_dims
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
        bnds_map = cfutil.get_cell_boundary_map(ds)  # {var_name: bnds_name}

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
            expected_units = ce.get("units", "")
            is_character = coord_type == "character"
            is_scalar = _is_scalar_coord(ce)

            if is_scalar:
                bounds_values = ce.get("bounds_values", "")
                low = _check_scalar_coord(
                    ctx, ds, dim_id, out_name, ce, value, is_character,
                    data_out_name, dim_coord_names, aux_coord_names,
                    must_have_bounds, bounds_values,
                )
            else:
                low = _check_multi_value_coord(
                    ctx, ds, out_name, ce, requested, requested_bounds,
                    must_have_bounds, is_character, expected_units, bnds_map,
                )

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
            if (
                ce
                and not _is_scalar_coord(ce)
                and ce.get("type", "") != "character"
                and (
                    ce.get("stored_direction", "")
                    or ce.get("standard_name", "")
                    in _DIRECT_VERTICAL_STANDARD_NAMES
                )
            ):
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
                matches = ds.get_variables_by_attributes(
                    standard_name=standard_name
                )
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

            if (
                standard_name in _DIRECT_VERTICAL_STANDARD_NAMES
                and not _ncattr(coord_var, "formula_terms")
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
            ctx = _vertical_config_ctx(
                ds, "[AICC006] Variable dimension ordering"
            )
            return [ctx.to_result()]

        data_out_name = self.var_entry.get("out_name", "")
        if not data_out_name or data_out_name not in ds.variables:
            ctx.add_pass()
            return [ctx.to_result()]

        axis_entries = self.CTcoords.get("axis_entry", {})
        horizontal_dim_ids = [
            dim_id
            for dim_id in self.requested_dims
            if dim_id in HORIZONTAL_DIM_IDS
        ]
        expected_horizontal = []
        if horizontal_dim_ids:
            handler = self._grid_type_handlers.get(self._grid_type)
            if handler is None:
                ctx.add_failure(
                    f"Cannot determine expected dimensions for unsupported grid "
                    f"type '{self._grid_type}'. Supported grid types: "
                    f"{sorted(self._grid_type_handlers)}."
                )
                return [ctx.to_result()]
            horizontal_entries = self._grid_table_entries(handler)
            expected_horizontal = getattr(self, handler["dimensions"])(
                ds, horizontal_dim_ids, horizontal_entries
            )

        # Build expected C-order dims (reverse of CMOR Fortran order, scalars excluded)
        expected = []
        horizontal_added = False
        for dim_id in reversed(self.requested_dims):
            if dim_id in HORIZONTAL_DIM_IDS:
                if not horizontal_added:
                    expected.extend(expected_horizontal)
                    horizontal_added = True
                continue
            if dim_id in VERTICAL_GENERIC_IDS:
                expected.append("lev")
                continue
            if _is_time_dim(dim_id):
                ce = axis_entries.get(dim_id, {})
                expected.append(ce.get("out_name", "time"))
                continue
            ce = axis_entries.get(dim_id, {})
            if ce and _is_scalar_coord(ce):
                continue  # scalar coords never appear in variable dims
            expected.append(ce.get("out_name", dim_id) if ce else dim_id)

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

        # Unstructured horizontal coordinates are auxiliary coordinates.
        if self._grid_type == "unstructured":
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
                    if name not in dim_coord_names and ds.variables[name].ndim == 1
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
            if _is_scalar_coord(ce):
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

    def check_quantization(self, ds):
        """Verify CF-1.12 lossy-compression metadata for quantized variables."""
        ctx = TestCtx(BaseCheck.HIGH, "[AICC008] CF-1.12 quantization metadata")
        algorithms = {
            "bitround": ("quantization_nsb", 23, 52),
            "bitgroom": ("quantization_nsd", 7, 15),
            "digitround": ("quantization_nsd", 7, 15),
            "granular_bitround": ("quantization_nsd", 7, 15),
        }
        library_attributes = {
            "_QuantizeBitRoundNumberOfSignificantBits": (
                "bitround",
                "quantization_nsb",
            ),
            "_QuantizeBitGroomNumberOfSignificantDigits": (
                "bitgroom",
                "quantization_nsd",
            ),
            "_QuantizeGranularBitRoundNumberOfSignificantDigits": (
                "granular_bitround",
                "quantization_nsd",
            ),
        }

        quantized_vars = {
            name: var
            for name, var in ds.variables.items()
            if "quantization" in var.ncattrs()
        }
        parameter_only_vars = {
            name: var
            for name, var in ds.variables.items()
            if (
                "quantization_nsb" in var.ncattrs()
                or "quantization_nsd" in var.ncattrs()
                or any(attr in var.ncattrs() for attr in library_attributes)
            )
            and name not in quantized_vars
        }
        container_names = {
            name
            for name, var in ds.variables.items()
            if "algorithm" in var.ncattrs() or "implementation" in var.ncattrs()
        }
        container_names.update(
            var.getncattr("quantization")
            for var in quantized_vars.values()
            if isinstance(var.getncattr("quantization"), str)
            and var.getncattr("quantization")
        )

        if not quantized_vars and not parameter_only_vars and not container_names:
            ctx.add_pass()
            return [ctx.to_result()]

        protected_vars = set(cfutil.get_coordinate_variables(ds))
        for var in ds.variables.values():
            coordinates = _ncattr(var, "coordinates")
            if isinstance(coordinates, str):
                protected_vars.update(coordinates.split())
            for attr in ("formula_terms", "cell_measures"):
                value = _ncattr(var, attr)
                if isinstance(value, str):
                    protected_vars.update(_parse_formula_terms(value).values())

        container_algorithms = {}
        implementation_pattern = re.compile(
            r"\S+ version \S+(?: \([^()]+\))?"
        )
        for container_name in sorted(container_names):
            if container_name not in ds.variables:
                ctx.add_failure(
                    f"Quantization container variable "
                    f"{_format_attribute(container_name)} not found."
                )
                continue

            container = ds.variables[container_name]
            algorithm = _ncattr(container, "algorithm", None)
            implementation = _ncattr(container, "implementation", None)

            if not isinstance(algorithm, str) or not algorithm:
                ctx.add_failure(
                    f"Quantization container {_format_attribute(container_name)} "
                    f"must have a "
                    f"non-empty string attribute 'algorithm'."
                )
            elif algorithm not in algorithms:
                ctx.add_failure(
                    f"Quantization container {_format_attribute(container_name)} "
                    f"algorithm="
                    f"{_format_attribute(algorithm)}; expected one of "
                    f"{sorted(algorithms)}."
                )
            else:
                ctx.add_pass()
                container_algorithms[container_name] = algorithm

            if not isinstance(implementation, str) or not implementation:
                ctx.add_failure(
                    f"Quantization container {_format_attribute(container_name)} "
                    f"must have a "
                    f"non-empty string attribute 'implementation'."
                )
            elif implementation_pattern.fullmatch(implementation) is None:
                ctx.add_failure(
                    f"Quantization container {_format_attribute(container_name)} "
                    f"implementation="
                    f"{_format_attribute(implementation)} does not match "
                    f"'software-name version version-string "
                    f"[(optional-information)]'."
                )
            else:
                ctx.add_pass()

        for var_name, var in sorted(parameter_only_vars.items()):
            ctx.add_failure(
                f"Variable '{var_name}' has quantization parameter or library "
                f"metadata but is missing the CF 'quantization' attribute."
            )

        for var_name, var in sorted(quantized_vars.items()):
            quantization = var.getncattr("quantization")
            if not isinstance(quantization, str) or not quantization:
                ctx.add_failure(
                    f"Variable '{var_name}' quantization attribute must be a "
                    f"non-empty string naming a quantization container."
                )
                continue
            if quantization not in ds.variables:
                # The missing container is reported above; retain a per-variable issue.
                ctx.add_failure(
                    f"Variable '{var_name}' quantization="
                    f"{_format_attribute(quantization)} references "
                    f"a container that does not exist."
                )
                continue
            ctx.add_pass()

            dtype = np.dtype(var.dtype)
            if dtype.kind != "f" or dtype.itemsize not in (4, 8):
                ctx.add_failure(
                    f"Variable '{var_name}' has dtype '{dtype}'; CF quantization "
                    f"is permitted only for float or double variables."
                )
            else:
                ctx.add_pass()

            if var_name in protected_vars:
                ctx.add_failure(
                    f"Variable '{var_name}' must not be quantized because it is a "
                    f"coordinate variable or is referenced by a coordinates, "
                    f"formula_terms, or cell_measures attribute."
                )
            else:
                ctx.add_pass()

            algorithm = container_algorithms.get(quantization)
            if algorithm is None:
                continue
            parameter_name, float_max, double_max = algorithms[algorithm]
            other_parameter = (
                "quantization_nsd"
                if parameter_name == "quantization_nsb"
                else "quantization_nsb"
            )
            parameter = (
                var.getncattr(parameter_name)
                if parameter_name in var.ncattrs()
                else None
            )

            if not isinstance(parameter, (int, np.integer)):
                ctx.add_failure(
                    f"Variable '{var_name}' using algorithm '{algorithm}' must "
                    f"have integer attribute '{parameter_name}'."
                )
            elif dtype.kind == "f" and dtype.itemsize in (4, 8):
                maximum = float_max if dtype.itemsize == 4 else double_max
                if not 1 <= int(parameter) <= maximum:
                    ctx.add_failure(
                        f"Variable '{var_name}' {parameter_name}={parameter}; "
                        f"expected an integer in [1, {maximum}] for dtype '{dtype}'."
                    )
                else:
                    ctx.add_pass()

            if other_parameter in var.ncattrs():
                ctx.add_failure(
                    f"Variable '{var_name}' uses algorithm '{algorithm}' and must "
                    f"use '{parameter_name}', not '{other_parameter}'."
                )

            system_attrs = [
                attr for attr in library_attributes if attr in var.ncattrs()
            ]
            if len(system_attrs) > 1:
                ctx.add_failure(
                    f"Variable '{var_name}' has multiple netCDF quantization "
                    f"attributes: {system_attrs}."
                )
            for system_attr in system_attrs:
                system_algorithm, system_parameter = library_attributes[system_attr]
                if algorithm != system_algorithm:
                    ctx.add_failure(
                        f"Variable '{var_name}' {system_attr} indicates algorithm "
                        f"'{system_algorithm}', but container '{quantization}' "
                        f"specifies '{algorithm}'."
                    )
                if parameter_name == system_parameter and isinstance(
                    parameter, (int, np.integer)
                ):
                    system_value = var.getncattr(system_attr)
                    if int(system_value) != int(parameter):
                        ctx.add_failure(
                            f"Variable '{var_name}' {system_attr}={system_value} "
                            f"does not match {parameter_name}={parameter}."
                        )
                    else:
                        ctx.add_pass()

        return [ctx.to_result()]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------






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


def _sample_formula_term(var, vertical_dim: str, point: dict):
    """Read one point from nonvertical dimensions and retain the vertical axis."""
    key = tuple(
        slice(None) if dim == vertical_dim else point.get(dim, 0)
        for dim in var.dimensions
    )
    values = np.ma.asarray(var[key], dtype="float64")
    if np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"formula term '{var.name}' is masked at the sampled point")
    values = np.asarray(values, dtype="float64").squeeze()
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"formula term '{var.name}' is non-finite at the sampled point"
        )
    if vertical_dim in var.dimensions:
        return np.asarray(values).reshape(-1)
    if np.asarray(values).size != 1:
        raise ValueError(
            f"formula term '{var.name}' did not reduce to a scalar at the "
            f"sampled point"
        )
    return float(np.asarray(values).reshape(-1)[0])


def _eval_formula_node(node, terms):
    """Evaluate the arithmetic subset used by CMOR vertical formulas."""
    if isinstance(node, ast.Expression):
        return _eval_formula_node(node.body, terms)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in terms:
            raise KeyError(f"formula term '{node.id}' is not declared")
        return terms[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_formula_node(node.operand, terms)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_formula_node(node.left, terms)
        right = _eval_formula_node(node.right, terms)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: left ** right,
        }
        operation = operations.get(type(node.op))
        if operation is None:
            raise ValueError("formula contains an unsupported operator")
        return operation()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [_eval_formula_node(arg, terms) for arg in node.args]
        functions = {
            "min": np.minimum,
            "max": np.maximum,
            "exp": np.exp,
            "log": np.log,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
        }
        function = functions.get(node.func.id)
        if function is None or not arguments:
            raise ValueError(f"unsupported formula function '{node.func.id}'")
        if node.func.id in {"min", "max"}:
            result = arguments[0]
            for argument in arguments[1:]:
                result = function(result, argument)
            return result
        if len(arguments) != 1:
            raise ValueError(f"formula function '{node.func.id}' expects one argument")
        return function(arguments[0])
    raise ValueError("formula contains unsupported syntax")


def _evaluate_vertical_formula(formula: str, standard_name: str, terms, nlevels: int):
    """Evaluate a CMOR vertical formula for one sampled nonvertical point."""
    if standard_name == "ocean_sigma_z_coordinate":
        required = {"eta", "sigma", "depth", "depth_c", "nsigma", "zlev"}
        missing = sorted(required - terms.keys())
        if missing:
            raise KeyError(f"formula term(s) {missing} are not declared")
        count = int(np.asarray(terms["nsigma"]).reshape(-1)[0])
        count = max(0, min(count, nlevels))
        sigma = np.broadcast_to(terms["sigma"], (nlevels,))
        zlev = np.broadcast_to(terms["zlev"], (nlevels,))
        profile = np.array(zlev, dtype="float64", copy=True)
        profile[:count] = terms["eta"] + sigma[:count] * (
            min(terms["depth_c"], terms["depth"]) + terms["eta"]
        )
        return profile

    if "=" not in formula:
        raise ValueError("formula has no '=' expression")
    expression = formula.split("=", 1)[1].strip().replace("^", "**")
    for term in sorted(terms, key=len, reverse=True):
        expression = re.sub(
            rf"\b{re.escape(term)}\s*\([^()]*\)", term, expression
        )
    parsed = ast.parse(expression, mode="eval")
    result = _eval_formula_node(parsed, terms)
    try:
        return np.asarray(np.broadcast_to(result, (nlevels,)), dtype="float64")
    except ValueError as exc:
        raise ValueError(
            f"formula result has shape {np.shape(result)}, expected {nlevels} levels"
        ) from exc


def _formula_vertical_profile(
    ds,
    coord_var,
    ce: dict,
    formula_terms: dict,
):
    """Calculate one usable formula-derived profile across all vertical levels."""
    if coord_var.ndim != 1:
        raise ValueError(
            f"coordinate '{coord_var.name}' must be one-dimensional; found "
            f"dimensions {list(coord_var.dimensions)}"
        )
    vertical_dim = coord_var.dimensions[0]
    if not formula_terms:
        raise ValueError("CMOR formula_terms mapping is missing or empty")

    variables = {}
    for term, var_name in formula_terms.items():
        if var_name not in ds.variables:
            raise KeyError(f"formula term '{term}' references missing '{var_name}'")
        variables[term] = ds.variables[var_name]

    sample_dims = []
    for var in variables.values():
        for dim in var.dimensions:
            if dim != vertical_dim and dim not in sample_dims:
                sample_dims.append(dim)
    sample_shape = tuple(len(ds.dimensions[dim]) for dim in sample_dims)
    sample_count = int(np.prod(sample_shape)) if sample_shape else 1

    last_error = None
    for flat_index in range(min(sample_count, 10000)):
        indices = np.unravel_index(flat_index, sample_shape) if sample_shape else ()
        point = dict(zip(sample_dims, indices))
        try:
            terms = {
                term: _sample_formula_term(var, vertical_dim, point)
                for term, var in variables.items()
            }
        except (TypeError, ValueError) as exc:
            last_error = exc
            continue

        try:
            profile = _evaluate_vertical_formula(
                ce.get("formula", ""),
                ce.get("standard_name", ""),
                terms,
                len(coord_var),
            )
        except (KeyError, TypeError, ValueError, SyntaxError):
            # Formula syntax and term declarations are independent of the point.
            raise

        try:
            profile = _numeric_profile(profile, "Formula-derived profile")
        except (TypeError, ValueError) as exc:
            last_error = exc
            continue

        sample = (
            "sampled point "
            + ", ".join(f"{dim}={point[dim]}" for dim in sample_dims)
            if sample_dims
            else "vertical-only formula terms"
        )
        return profile, sample

    detail = f": {last_error}" if last_error else ""
    raise ValueError(f"no usable sampled point was found{detail}")


def _vertical_config_ctx(ds, name: str) -> TestCtx:
    """Return the common prerequisite failure for model-specific Z checks."""
    ctx = TestCtx(BaseCheck.HIGH, name)
    source_id = _ncattr(ds, "source_id")
    ctx.add_failure(
        f"Vertical checks cannot run because source_id "
        f"{_format_attribute(source_id)} is not registered in the model "
        f"configuration. Add a matching source_id key to the default "
        f"configuration or pass it through the 'model_config' option."
    )
    return ctx


def _check_formula_var_attrs(
    ctx: TestCtx,
    low_ctx: TestCtx,
    var,
    var_name: str,
    ft_entry: dict,
):
    """Check a formula-term variable's table-defined attributes."""
    expected_units = ft_entry.get("units", "")
    if expected_units:
        level, msg = _compare_units(_ncattr(var, "units"), expected_units)
        if level != "ok":
            ctx.add_failure(f"Formula term '{var_name}' units: {msg}")
        else:
            ctx.add_pass()

    _check_coord_attrs(ctx, low_ctx, var, var_name, ft_entry)


def _check_coord_attrs(
    ctx: TestCtx,
    low_ctx: TestCtx,
    var,
    var_name: str,
    ce: dict,
):
    """Check required standard_name and advisory long_name attributes."""
    expected_sn = ce.get("standard_name", "")
    if expected_sn:
        actual_sn = _ncattr(var, "standard_name")
        if actual_sn != expected_sn:
            ctx.add_failure(
                f"'{var_name}' standard_name={_format_attribute(actual_sn)}; "
                f"expected '{expected_sn}'."
            )
        else:
            ctx.add_pass()

    expected_ln = ce.get("long_name", "")
    if expected_ln:
        actual_ln = _ncattr(var, "long_name")
        if actual_ln != expected_ln:
            low_ctx.add_failure(
                f"'{var_name}' long_name={_format_attribute(actual_ln)}; "
                f"expected '{expected_ln}'."
            )




# kept for scalar-only callers


def _check_scalar_coord(ctx: TestCtx, ds, dim_id: str, out_name: str,
                         ce: dict, value: str, is_character: bool,
                         data_out_name: str,
                         dim_coord_names: set, aux_coord_names: set,
                         must_have_bounds: bool = False,
                         bounds_values: str = "") -> list:
    """Check a scalar coordinate. Returns list[Result] for LOW advisories."""
    low_ctx = TestCtx(BaseCheck.LOW, f"[AICC005] Coordinate '{out_name}' (advisory)")

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

    coord_var = ds.variables[coord_var_name]
    _check_coord_attrs(ctx, low_ctx, coord_var, coord_var_name, ce)

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
                        valid_min_s = ce.get("valid_min", "")
                        valid_max_s = ce.get("valid_max", "")
                        if valid_min_s and valid_max_s:
                            vmin, vmax = float(valid_min_s), float(valid_max_s)
                            if vmin <= actual <= vmax:
                                # Within valid range: advisory only
                                ctx.add_pass()
                                low_ctx.add_failure(
                                    f"'{coord_var_name}' value={actual} differs from "
                                    f"expected {expected_val} but is within valid range "
                                    f"[{vmin}, {vmax}]."
                                )
                            else:
                                ctx.add_failure(
                                    f"'{coord_var_name}' value={actual}; expected "
                                    f"{expected_val} (valid range [{vmin}, {vmax}])."
                                )
                        else:
                            ctx.add_failure(
                                f"'{coord_var_name}' value={actual}; expected {expected_val}."
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
    if must_have_bounds and not is_character:
        bnds_name = f"{out_name}_bnds"
        coord_var_ref = ds.variables.get(coord_var_name)
        if coord_var_ref is not None:
            declared_bnds = _ncattr(coord_var_ref, "bounds")
            if declared_bnds != bnds_name:
                ctx.add_failure(
                    f"'{coord_var_name}' bounds="
                    f"{_format_attribute(declared_bnds)}; expected '{bnds_name}'."
                )
            else:
                ctx.add_pass()

        if bnds_name not in ds.variables:
            ctx.add_failure(
                f"Scalar bounds variable '{bnds_name}' for '{out_name}' not found."
            )
        else:
            ctx.add_pass()
            bnds_var = ds.variables[bnds_name]
            if bnds_var.ncattrs():
                ctx.add_failure(
                    f"'{bnds_name}' must have no attributes; "
                    f"found: {list(bnds_var.ncattrs())}."
                )
            else:
                ctx.add_pass()

            if bounds_values and value:
                try:
                    parts = bounds_values.split()
                    exp_lo, exp_hi = float(parts[0]), float(parts[1])
                    file_bnds = np.asarray(bnds_var[:]).flatten()
                    act_lo, act_hi = float(file_bnds[0]), float(file_bnds[1])

                    tol_str = ce.get("tolerance", "")
                    if tol_str:
                        # CMOR tolerance for scalar (i=0, one value)
                        scalar_val = float(value)
                        tol = _cmor_tol_val(0, [scalar_val], [(exp_lo, exp_hi)], float(tol_str))
                        lo_diff = abs(act_lo - exp_lo)
                        hi_diff = abs(act_hi - exp_hi)
                        if lo_diff > tol or hi_diff > tol:
                            ctx.add_failure(
                                f"'{bnds_name}' bounds=[{act_lo}, {act_hi}] outside "
                                f"tolerance {tol:.3g} of expected [{exp_lo}, {exp_hi}]."
                            )
                        else:
                            ctx.add_pass()
                            if lo_diff > 0 or hi_diff > 0:
                                low_ctx.add_failure(
                                    f"'{bnds_name}' bounds=[{act_lo}, {act_hi}] within "
                                    f"tolerance but not exact; expected [{exp_lo}, {exp_hi}]."
                                )
                    else:
                        default_tol = 1e-6 * max(1.0, abs(exp_lo), abs(exp_hi))
                        if not (np.isclose(act_lo, exp_lo, atol=default_tol)
                                and np.isclose(act_hi, exp_hi, atol=default_tol)):
                            ctx.add_failure(
                                f"'{bnds_name}' bounds=[{act_lo}, {act_hi}]; "
                                f"expected [{exp_lo}, {exp_hi}]."
                            )
                        else:
                            ctx.add_pass()
                except Exception as exc:
                    ctx.add_failure(f"Could not check bounds_values of '{bnds_name}': {exc}")

    return [low_ctx.to_result()] if low_ctx.messages else []


def _check_multi_value_coord(ctx: TestCtx, ds, out_name: str, ce: dict,
                              requested: list, requested_bounds: list,
                              must_have_bounds: bool, is_character: bool,
                              expected_units: str, bnds_map: dict) -> list:
    """Check a multi-value coordinate. Returns list[Result] for LOW advisories."""
    low_ctx = TestCtx(BaseCheck.LOW, f"[AICC005] Coordinate '{out_name}' (advisory)")

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
    _check_coord_attrs(ctx, low_ctx, coord_var, coord_var_name, ce)

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

        # units via udunits
        if expected_units:
            level, msg = _compare_units(_ncattr(coord_var, "units"), expected_units)
            if level != "ok":
                ctx.add_failure(f"'{coord_var_name}' units: {msg}")
            else:
                ctx.add_pass()

        # Build requested bound pairs aligned with requested values (for tol calc)
        req_floats = [float(r) for r in requested] if requested else []
        req_pairs: list = []
        if requested_bounds:
            req_pairs = list(zip(
                [float(requested_bounds[i]) for i in range(0, len(requested_bounds), 2)],
                [float(requested_bounds[i]) for i in range(1, len(requested_bounds), 2)],
            ))

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
                        + (f" (outside tolerance, factor={tol_factor})." if tol_factor else ".")
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

        if must_have_bounds:
            bnds_name = (bnds_map.get(coord_var_name)
                         or _ncattr(coord_var, "bounds")
                         or f"{coord_var_name}_bnds")
            declared_bnds = _ncattr(coord_var, "bounds")
            if not declared_bnds:
                ctx.add_failure(
                    f"'{coord_var_name}' must have a 'bounds' attribute "
                    f"set to '{bnds_name}'."
                )
            else:
                ctx.add_pass()

            if bnds_name not in ds.variables:
                ctx.add_failure(
                    f"Bounds variable '{bnds_name}' for '{coord_var_name}' not found."
                )
            else:
                ctx.add_pass()
                bnds_var = ds.variables[bnds_name]
                if bnds_var.ncattrs():
                    ctx.add_failure(
                        f"'{bnds_name}' must have no attributes; "
                        f"found {list(bnds_var.ncattrs())}."
                    )
                else:
                    ctx.add_pass()

                if req_pairs:
                    try:
                        file_bnds = np.asarray(bnds_var[:]).reshape(-1, 2)
                        outside_tol_bnds, within_tol_bnds = [], []
                        for i, (exp_lo, exp_hi) in enumerate(req_pairs):
                            if tol_factor is not None:
                                tol = _cmor_tol_val(i, req_floats, req_pairs, tol_factor)
                                matches = [fb for fb in file_bnds
                                           if abs(fb[0] - exp_lo) <= tol and abs(fb[1] - exp_hi) <= tol]
                                if not matches:
                                    outside_tol_bnds.append((exp_lo, exp_hi))
                                elif not any(fb[0] == exp_lo and fb[1] == exp_hi for fb in file_bnds):
                                    within_tol_bnds.append((exp_lo, exp_hi))
                            else:
                                if not any(fb[0] == exp_lo and fb[1] == exp_hi for fb in file_bnds):
                                    outside_tol_bnds.append((exp_lo, exp_hi))
                        if outside_tol_bnds:
                            ctx.add_failure(
                                f"'{bnds_name}' missing requested bound pair(s) {outside_tol_bnds}"
                                + (f" (outside tolerance, factor={tol_factor})." if tol_factor else ".")
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

    return [low_ctx.to_result()] if low_ctx.messages else []
