"""
aicc.py — AWI ICON Coordinate Checker (AICC) / AI Compliance Checker

Verifies CMIP7 coordinate compliance for AWI and ICON model output
with unstructured horizontal grids.

Variable discovery delegates entirely to compliance_checker.cf.util (cfutil)
so that CF conventions are applied consistently across the whole checker
ecosystem rather than being re-implemented here.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
from compliance_checker.base import BaseCheck, BaseNCCheck, Result, TestCtx
from compliance_checker.cf import util as cfutil

from cc_plugin_aicc import __version__

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default CMIP7 tables path: honour env var, fall back to a sibling checkout.
# Override at runtime via the 'tables' checker option.
_DEFAULT_TABLES_PATH = os.environ.get(
    "CMIP7_TABLES_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "cmip7-cmor-tables" / "tables"),
)

# Generic vertical level dimension_id → specific coordinate-table entry name
_VERTICAL_MAPPING = {
    "AWI": {
        "alevel": "alternate_hybrid_sigma",
        "alevhalf": "alternate_hybrid_sigma_half",
        "olevel": "depth_coord",
        "olevhalf": "depth_coord_half",
    },
    "ICON": {
        "alevel": "modified_sleve_model_level",
        "alevhalf": "modified_sleve_half_level",
        "olevel": "depth_coord",
        "olevhalf": "depth_coord_half",
    },
}

_VERTICAL_GENERIC_IDS = frozenset({"alevel", "alevhalf", "olevel", "olevhalf"})
_HORIZONTAL_DIM_IDS = frozenset({"latitude", "longitude"})

# First word of the realm global attribute → CMIP7 table name fragment
_REALM_TO_TABLE = {
    "atmos": "atmos",
    "land": "land",
    "ocean": "ocean",
    "seaIce": "seaIce",
    "landIce": "landIce",
    "aerosol": "aerosol",
    "atmosChem": "atmosChem",
    "ocnBgchem": "ocnBgchem",
}


def _is_time_dim(dim_id: str) -> bool:
    return dim_id.startswith("time")


# Adapted from swarnaleem's attr() — github.com/swarnaleem/cc-plugin-wcrp feature/coordinate-standard db0791d plugins/coordinate_standard/classify.py
def _ncattr(var_or_ds, name: str, default=""):
    """Safe attribute read for both netCDF4 variables and Datasets."""
    return getattr(var_or_ds, name, default) or default


# Adapted from swarnaleem's neutral_dtype() — github.com/swarnaleem/cc-plugin-wcrp feature/coordinate-standard db0791d plugins/coordinate_standard/classify.py
def _neutral_dtype(var) -> str:
    """Return 'character', 'integer', or 'double' for a netCDF4 variable."""
    kind = getattr(getattr(var, "dtype", None), "kind", "")
    if kind in ("S", "U"):
        return "character"
    if kind in ("i", "u"):
        return "integer"
    return "double"


# Adapted from swarnaleem's _compare_units() — github.com/swarnaleem/cc-plugin-wcrp feature/coordinate-standard db0791d plugins/coordinate_standard/matching.py
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


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class AICC(BaseNCCheck, BaseCheck):
    """AWI ICON Coordinate Checker for CMIP7 unstructured model output."""

    register_checker = True
    _cc_spec = "aicc"
    _cc_spec_version = __version__
    _cc_description = (
        "AWI ICON Coordinate Checks (AICC) — verifies CMIP7 coordinate compliance "
        "for AWI/ICON unstructured model output."
    )
    _cc_url = ""
    _cc_display_headers = {3: "Required", 2: "Recommended", 1: "Suggested"}

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
        tables_path = self.options.get("tables", _DEFAULT_TABLES_PATH)
        self._read_cmip7_tables(tables_path)

        # Determine model type from source_id global attribute
        source_id = _ncattr(dataset, "source_id")
        if "AWI" in source_id:
            self.model_type = "AWI"
        elif "ICON" in source_id:
            self.model_type = "ICON"
        else:
            self.model_type = None

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
            mapped = _REALM_TO_TABLE.get(realm_first)
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
                f"branded_variable '{self.branded_variable}' not found in any "
                f"CMIP7 table (tried table_id='{_ncattr(ds, 'table_id')}', "
                f"realm='{_ncattr(ds, 'realm')}')."
            )
            return [ctx.to_result()]

        ctx.add_pass()
        return [ctx.to_result()]

    # ------------------------------------------------------------------

    def check_grid(self, ds):
        """Verify unstructured horizontal grid auxiliary coordinates and vertices."""
        has_lat = "latitude" in self.requested_dims
        has_lon = "longitude" in self.requested_dims
        if not (has_lat or has_lon):
            ctx = TestCtx(BaseCheck.HIGH, "[AICC002] Horizontal grid coordinates")
            ctx.add_pass()
            return [ctx.to_result()]

        results = []
        grid_var_entries = self.CTgrids.get("variable_entry", {})

        # Use cfutil for CF-aware variable discovery
        aux_coord_names = set(cfutil.get_auxiliary_coordinate_variables(ds))
        lat_vars = cfutil.get_true_latitude_variables(ds)
        lon_vars = cfutil.get_true_longitude_variables(ds)

        for dim_id, cf_vars in (("latitude", lat_vars), ("longitude", lon_vars)):
            if dim_id not in self.requested_dims:
                continue

            ctx = TestCtx(BaseCheck.HIGH, f"[AICC002] {dim_id} auxiliary coordinate")
            grid_entry = grid_var_entries.get(dim_id, {})
            expected_units = grid_entry.get("units", "")
            vtx_key = f"vertices_{dim_id}"
            vtx_entry = grid_var_entries.get(vtx_key, {})
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

            # Must be 1-D (single unstructured cell dimension)
            if aux_var.ndim != 1:
                ctx.add_failure(
                    f"'{aux_var_name}' must have exactly one dimension (cell/ncells) "
                    f"for an unstructured grid; found ndim={aux_var.ndim}."
                )
            else:
                ctx.add_pass()

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
            if level == "fail":
                ctx.add_failure(f"'{aux_var_name}' units: {msg}")
            else:
                ctx.add_pass()  # pass even for convertible (advisory only)

            results.append(ctx.to_result())

            # --- Vertex bounds ---
            vtx_ctx = TestCtx(BaseCheck.HIGH, f"[AICC002] {vtx_key}")
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

            if vtx_var.ndim != 2:
                vtx_ctx.add_failure(
                    f"'{vtx_var_name}' must have 2 dimensions (ncells, ncorners); "
                    f"found ndim={vtx_var.ndim}."
                )
            else:
                vtx_ctx.add_pass()

            level, msg = _compare_units(_ncattr(vtx_var, "units"), vtx_expected_units)
            if level == "fail":
                vtx_ctx.add_failure(f"'{vtx_var_name}' units: {msg}")
            else:
                vtx_ctx.add_pass()

            results.append(vtx_ctx.to_result())

        # Data variable must list lat/lon in its 'coordinates' attribute
        data_out_name = self.var_entry.get("out_name", "") if self.var_entry else ""
        if data_out_name and data_out_name in ds.variables:
            data_var = ds.variables[data_out_name]
            coords_attr = _ncattr(data_var, "coordinates")
            coords_listed = coords_attr.split() if coords_attr else []

            for dim_id, cf_vars in (("latitude", lat_vars), ("longitude", lon_vars)):
                if dim_id not in self.requested_dims:
                    continue
                ctx = TestCtx(
                    BaseCheck.HIGH,
                    f"[AICC002] '{data_out_name}' coordinates attribute ({dim_id})",
                )
                found = any(
                    getattr(ds.variables.get(v), "standard_name", None) == dim_id
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

    # ------------------------------------------------------------------

    def check_vertical(self, ds):
        """Verify vertical coordinate(s) against the CMIP7 coordinate table."""
        vert_dims = [d for d in self.requested_dims if d in _VERTICAL_GENERIC_IDS]
        if not vert_dims:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC003] Vertical coordinates")
            ctx.add_pass()
            return [ctx.to_result()]

        if self.model_type is None:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC003] Vertical coordinates")
            ctx.add_failure(
                "source_id contains neither 'AWI' nor 'ICON'; "
                "cannot resolve generic vertical coordinate mapping."
            )
            return [ctx.to_result()]

        results = []
        axis_entries = self.CTcoords.get("axis_entry", {})
        formula_entries = self.CTformulas.get("formula_entry", {})

        # Use cfutil for CF-aware Z variable discovery
        z_var_names = set(cfutil.get_z_variables(ds))

        for generic_id in vert_dims:
            coord_key = _VERTICAL_MAPPING[self.model_type].get(generic_id)
            if not coord_key:
                ctx = TestCtx(BaseCheck.HIGH, f"[AICC003] {generic_id}")
                ctx.add_failure(
                    f"No vertical coordinate mapping for '{generic_id}' "
                    f"and model_type='{self.model_type}'."
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

            # Locate lev variable: exact out_name, then cfutil Z vars by standard_name
            if out_name in ds.variables:
                lev_var_name = out_name
            else:
                lev_var_name = next(
                    (v for v in z_var_names
                     if expected_sn and getattr(ds.variables[v], "standard_name", None) == expected_sn),
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

            # standard_name
            if expected_sn:
                actual_sn = _ncattr(lev_var, "standard_name")
                if actual_sn != expected_sn:
                    ctx.add_failure(
                        f"'{lev_var_name}' standard_name='{actual_sn}'; "
                        f"expected '{expected_sn}'."
                    )
                else:
                    ctx.add_pass()

            # units (via udunits-backed comparison)
            if expected_units:
                level, msg = _compare_units(_ncattr(lev_var, "units"), expected_units)
                if level == "fail":
                    ctx.add_failure(f"'{lev_var_name}' units: {msg}")
                else:
                    ctx.add_pass()

            # positive
            if expected_positive:
                actual_pos = _ncattr(lev_var, "positive")
                if actual_pos != expected_positive:
                    ctx.add_failure(
                        f"'{lev_var_name}' positive='{actual_pos}'; "
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
                        f"'{lev_var_name}' bounds='{declared_bnds}'; "
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
                    if bnds_var.ncattrs():
                        ctx.add_failure(
                            f"'{bnds_name}' must have no attributes; "
                            f"found: {list(bnds_var.ncattrs())}."
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
                    ctx.add_pass()
                    ft_map = _parse_formula_terms(ft_attr)
                    for term, var_name in ft_map.items():
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
                                    ctx, ds.variables[var_name], var_name, ft_entry
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
                                ctx, ds.variables[var_name], var_name, ft_entry
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
            is_climatology = ce.get("climatology", "") == "yes"

            ctx = TestCtx(
                BaseCheck.HIGH, f"[AICC004] Time coordinate ({time_dim_id})"
            )

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

            # standard_name=time
            if _ncattr(t_var, "standard_name") != "time":
                ctx.add_failure(
                    f"'{resolved_t}' standard_name='"
                    f"{_ncattr(t_var, 'standard_name')}'; expected 'time'."
                )
            else:
                ctx.add_pass()

            # units: must contain "since" and have a udunits-known base unit
            units = _ncattr(t_var, "units")
            if not units or "since" not in units:
                ctx.add_failure(
                    f"'{resolved_t}' units='{units}' is missing or invalid "
                    f"(expected format 'X since Y-M-D ...')."
                )
            elif not cfutil.units_known(units.split(" since ")[0]):
                ctx.add_failure(
                    f"'{resolved_t}' time base unit "
                    f"'{units.split(' since ')[0]}' not recognized by udunits."
                )
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
                                f"Climatology bounds variable '{clim_attr}' "
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
                            f"'{resolved_t}' bounds='{declared_bnds}'; "
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

        return results

    # ------------------------------------------------------------------

    def check_coord(self, ds):
        """Verify non-grid, non-vertical, non-time coordinate dimensions."""
        other_dims = [
            d for d in self.requested_dims
            if d not in _HORIZONTAL_DIM_IDS
            and d not in _VERTICAL_GENERIC_IDS
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
            is_multi = bool(requested)

            if is_multi:
                _check_multi_value_coord(
                    ctx, ds, out_name, ce, requested, requested_bounds,
                    must_have_bounds, is_character, expected_units, bnds_map,
                )
            else:
                _check_scalar_coord(
                    ctx, ds, dim_id, out_name, ce, value, is_character,
                    data_out_name, dim_coord_names, aux_coord_names,
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

        data_out_name = self.var_entry.get("out_name", "")
        if not data_out_name or data_out_name not in ds.variables:
            ctx.add_pass()
            return [ctx.to_result()]

        axis_entries = self.CTcoords.get("axis_entry", {})
        has_horizontal = any(d in _HORIZONTAL_DIM_IDS for d in self.requested_dims)

        # Build expected C-order dims (reverse of CMOR Fortran order, scalars excluded)
        expected = []
        for dim_id in reversed(self.requested_dims):
            if dim_id in _HORIZONTAL_DIM_IDS:
                continue  # merged into single cell-dim placeholder below
            if dim_id in _VERTICAL_GENERIC_IDS:
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

        if has_horizontal:
            expected.append("<ncells>")  # any name acceptable for unstructured cell dim

        actual = list(ds.variables[data_out_name].dimensions)

        if len(actual) != len(expected):
            ctx.add_failure(
                f"'{data_out_name}' has {len(actual)} dimension(s) {actual}; "
                f"expected {len(expected)} based on CMOR dims "
                f"{self.requested_dims} → {expected}."
            )
        else:
            mismatches = [
                f"position {i}: '{act}' (expected '{exp}')"
                for i, (exp, act) in enumerate(zip(expected, actual))
                if exp != "<ncells>" and exp != act
            ]
            if mismatches:
                ctx.add_failure(
                    f"'{data_out_name}' dimension order mismatch: "
                    + "; ".join(mismatches)
                    + f". CMOR dims: {self.requested_dims}."
                )
            else:
                ctx.add_pass()

        return [ctx.to_result()]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


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


def _check_formula_var_attrs(ctx: TestCtx, var, var_name: str, ft_entry: dict):
    """Spot-check units and standard_name of a formula-term variable."""
    expected_units = ft_entry.get("units", "")
    expected_sn = ft_entry.get("standard_name", "")
    if expected_units:
        level, msg = _compare_units(_ncattr(var, "units"), expected_units)
        if level == "fail":
            ctx.add_failure(f"Formula term '{var_name}' units: {msg}")
        else:
            ctx.add_pass()
    if expected_sn:
        if _ncattr(var, "standard_name") != expected_sn:
            ctx.add_failure(
                f"Formula term '{var_name}' standard_name="
                f"'{_ncattr(var, 'standard_name')}'; expected '{expected_sn}'."
            )
        else:
            ctx.add_pass()


def _check_coord_attrs(ctx: TestCtx, var, var_name: str, ce: dict):
    """Check standard_name and long_name of a coordinate variable against the table."""
    expected_sn = ce.get("standard_name", "")
    if expected_sn:
        actual_sn = _ncattr(var, "standard_name")
        if actual_sn != expected_sn:
            ctx.add_failure(
                f"'{var_name}' standard_name='{actual_sn}'; expected '{expected_sn}'."
            )
        else:
            ctx.add_pass()

    expected_ln = ce.get("long_name", "")
    if expected_ln:
        actual_ln = _ncattr(var, "long_name")
        if actual_ln != expected_ln:
            ctx.add_failure(
                f"'{var_name}' long_name='{actual_ln}'; expected '{expected_ln}'."
            )
        else:
            ctx.add_pass()


def _decode_char_scalar(var) -> str:
    """Decode a netCDF4 character-array scalar variable to a plain string."""
    import netCDF4 as nc4
    try:
        return nc4.chartostring(var[:])[()].decode("utf-8").rstrip("\x00").strip()
    except Exception:
        pass
    try:
        return b"".join(bytes(c) for c in var[:]).decode("utf-8").rstrip("\x00").strip()
    except Exception:
        return ""


def _check_scalar_coord(ctx: TestCtx, ds, dim_id: str, out_name: str,
                         ce: dict, value: str, is_character: bool,
                         data_out_name: str,
                         dim_coord_names: set, aux_coord_names: set):
    """Check a scalar coordinate (dimensionless or strlen-only)."""
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
        return
    ctx.add_pass()

    coord_var = ds.variables[coord_var_name]

    # Verify standard_name and long_name against the table
    _check_coord_attrs(ctx, coord_var, coord_var_name, ce)

    if is_character:
        dims = list(coord_var.dimensions)
        if dims != ["strlen"]:
            ctx.add_failure(
                f"Character scalar coordinate '{coord_var_name}' must have only "
                f"dimension 'strlen'; found {dims}."
            )
        else:
            ctx.add_pass()
        # Verify the string value matches the table entry
        if value:
            actual_str = _decode_char_scalar(coord_var)
            if actual_str != value:
                ctx.add_failure(
                    f"'{coord_var_name}' value='{actual_str}'; expected '{value}'."
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
                    if not np.isclose(actual, float(value), rtol=1e-5, atol=0):
                        ctx.add_failure(
                            f"Scalar coordinate '{coord_var_name}' value={actual}; "
                            f"expected {float(value)}."
                        )
                    else:
                        ctx.add_pass()
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
                f"current value: '{coord_attr}'."
            )
        else:
            ctx.add_pass()


def _check_multi_value_coord(ctx: TestCtx, ds, out_name: str, ce: dict,
                              requested: list, requested_bounds: list,
                              must_have_bounds: bool, is_character: bool,
                              expected_units: str, bnds_map: dict):
    """Check a multi-value coordinate (requested values must be present in file)."""
    import netCDF4 as nc4

    coord_var_name = out_name if out_name in ds.variables else None
    if coord_var_name is None:
        sn = ce.get("standard_name", "")
        if sn:
            matches = ds.get_variables_by_attributes(standard_name=sn)
            if matches:
                coord_var_name = matches[0].name

    if coord_var_name is None:
        ctx.add_failure(f"Coordinate variable '{out_name}' not found in file.")
        return
    ctx.add_pass()

    coord_var = ds.variables[coord_var_name]

    # Verify standard_name and long_name against the table
    _check_coord_attrs(ctx, coord_var, coord_var_name, ce)

    if is_character:
        # Dims must be (out_name, strlen)
        dims = list(coord_var.dimensions)
        if len(dims) != 2 or dims[1] != "strlen":
            ctx.add_failure(
                f"Character coordinate '{coord_var_name}' must have dims "
                f"('{out_name}', 'strlen'); found {dims}."
            )
        else:
            ctx.add_pass()

        if requested:
            try:
                vals = [v.rstrip("\x00").strip()
                        for v in nc4.chartostring(coord_var[:])]
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
        tol = float(tol_str) if tol_str else 1e-6

        # units via udunits
        if expected_units:
            level, msg = _compare_units(_ncattr(coord_var, "units"), expected_units)
            if level == "fail":
                ctx.add_failure(f"'{coord_var_name}' units: {msg}")
            else:
                ctx.add_pass()

        if requested:
            try:
                file_vals = list(np.asarray(coord_var[:]).flat)
                req_floats = [float(r) for r in requested]
                missing = [
                    r for r in req_floats
                    if not any(abs(r - fv) <= tol for fv in file_vals)
                ]
                if missing:
                    ctx.add_failure(
                        f"'{coord_var_name}' missing requested value(s) "
                        f"{missing} (tolerance={tol})."
                    )
                else:
                    ctx.add_pass()
            except Exception as exc:
                ctx.add_failure(f"Could not check values of '{coord_var_name}': {exc}")

        if must_have_bounds:
            # cfutil-derived bounds map takes precedence
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

                if requested_bounds:
                    try:
                        file_bnds = np.asarray(bnds_var[:]).reshape(-1, 2)
                        req_pairs = list(zip(
                            [float(requested_bounds[i]) for i in range(0, len(requested_bounds), 2)],
                            [float(requested_bounds[i]) for i in range(1, len(requested_bounds), 2)],
                        ))
                        missing_pairs = [
                            pair for pair in req_pairs
                            if not any(
                                abs(pair[0] - fb[0]) <= tol and abs(pair[1] - fb[1]) <= tol
                                for fb in file_bnds
                            )
                        ]
                        if missing_pairs:
                            ctx.add_failure(
                                f"'{bnds_name}' missing requested bound pair(s) "
                                f"{missing_pairs} (tolerance={tol})."
                            )
                        else:
                            ctx.add_pass()
                    except Exception as exc:
                        ctx.add_failure(
                            f"Could not check bounds of '{coord_var_name}': {exc}"
                        )


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
