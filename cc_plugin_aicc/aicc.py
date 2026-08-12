"""
aicc.py — AWI ICON Coordinate Checker (AICC) / AI Compliance Checker

Verifies CMIP7 coordinate compliance for AWI and ICON model output
with unstructured horizontal grids.
"""

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr
from compliance_checker.base import BaseCheck, BaseNCCheck, Result

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


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class AICC(BaseNCCheck):
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

    @classmethod
    def make_result(cls, level, score, out_of, name, messages):
        return Result(level, (score, out_of), name, messages)

    def __del__(self):
        xrds = getattr(self, "xrds", None)
        if xrds is not None and hasattr(xrds, "close"):
            xrds.close()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, dataset):
        self.dataset = dataset
        self.filepath = os.path.realpath(
            os.path.normpath(os.path.expanduser(dataset.filepath()))
        )
        self.xrds = xr.open_dataset(self.filepath, decode_times=False)
        self._all_file_vars = (
            list(self.xrds.data_vars.keys()) + list(self.xrds.coords.keys())
        )

        # Read CMIP7 CMOR tables
        tables_path = self.options.get("tables", _DEFAULT_TABLES_PATH)
        self._read_cmip7_tables(tables_path)

        # Determine model type from source_id
        source_id = self._get_attr("source_id")
        if "AWI" in source_id:
            self.model_type = "AWI"
        elif "ICON" in source_id:
            self.model_type = "ICON"
        else:
            self.model_type = None

        # Identify variable and its CMOR table entry
        self.branded_variable = self._get_attr("branded_variable") or None
        self.var_entry = None
        self.table_name = None
        self.requested_dims = []

        if self.branded_variable:
            self._resolve_table_and_variable()

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
                raise FileNotFoundError(
                    f"Required CMIP7 table not found: '{path}'"
                )
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

    def _get_attr(self, attr, default=""):
        try:
            return self.dataset.getncattr(attr)
        except AttributeError:
            return default

    def _resolve_table_and_variable(self):
        """Resolve *branded_variable* to a var_entry in the appropriate CMOR table."""
        table_id = self._get_attr("table_id")
        if table_id and table_id in self.CT:
            candidates = [table_id]
        else:
            realm_raw = self._get_attr("realm")
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
        """Verify that branded_variable is set and found in the CMIP7 tables."""
        desc = "branded_variable identification"
        level = BaseCheck.HIGH
        messages = []

        if not self.branded_variable:
            return self.make_result(
                level, 0, 1, desc,
                ["Global attribute 'branded_variable' is not set. "
                 "Cannot identify the variable for CMOR table lookup."],
            )

        if self.var_entry is None:
            return self.make_result(
                level, 0, 1, desc,
                [f"branded_variable '{self.branded_variable}' not found in any "
                 f"CMIP7 table (tried table_id='{self._get_attr('table_id')}', "
                 f"realm='{self._get_attr('realm')}')."],
            )

        return self.make_result(level, 1, 1, desc, messages)

    # ------------------------------------------------------------------

    def check_grid(self, ds):
        """Verify unstructured horizontal grid auxiliary coordinates and vertices."""
        desc = "Horizontal grid coordinates"
        level = BaseCheck.HIGH
        messages = []

        has_lat = "latitude" in self.requested_dims
        has_lon = "longitude" in self.requested_dims
        if not (has_lat or has_lon):
            return self.make_result(level, 1, 1, desc, messages)

        out_of = 0
        score = 0
        grid_var_entries = self.CTgrids.get("variable_entry", {})
        data_var_out_name = (
            self.var_entry.get("out_name", "") if self.var_entry else ""
        )

        for dim_id in ("latitude", "longitude"):
            if dim_id not in self.requested_dims:
                continue

            grid_entry = grid_var_entries.get(dim_id, {})
            out_name = grid_entry.get("out_name", dim_id)
            expected_sn = grid_entry.get("standard_name", "")
            expected_units = grid_entry.get("units", "")

            # Auxiliary lat/lon variable must exist
            out_of += 1
            aux_var = self._find_var_by(standard_name=expected_sn)
            if aux_var is None:
                messages.append(
                    f"No auxiliary coordinate with standard_name='{expected_sn}' "
                    f"found for horizontal dim_id='{dim_id}' (expected out_name='{out_name}')."
                )
                continue
            score += 1

            # Check units
            out_of += 1
            actual_units = self.xrds[aux_var].attrs.get("units", "")
            if actual_units != expected_units:
                messages.append(
                    f"Auxiliary coordinate '{aux_var}' (standard_name='{expected_sn}') "
                    f"units='{actual_units}', expected '{expected_units}'."
                )
            else:
                score += 1

            # Must be 1-D (single cell/ncells dimension)
            out_of += 1
            if len(self.xrds[aux_var].dims) != 1:
                messages.append(
                    f"Auxiliary coordinate '{aux_var}' must have exactly one dimension "
                    f"(cell/ncells) for unstructured grids; "
                    f"found dims={list(self.xrds[aux_var].dims)}."
                )
            else:
                score += 1

            # Vertex bounds variable (vertices_latitude / vertices_longitude)
            vtx_key = f"vertices_{dim_id}"
            vtx_entry = grid_var_entries.get(vtx_key, {})
            vtx_out_name = vtx_entry.get("out_name", vtx_key)
            vtx_sn = vtx_entry.get("standard_name", "")

            out_of += 1
            vtx_var = self._find_var_by(name=vtx_out_name, standard_name=vtx_sn)
            if vtx_var is None:
                messages.append(
                    f"Vertex bounds variable '{vtx_out_name}' not found "
                    f"for '{dim_id}' auxiliary coordinate."
                )
            else:
                score += 1
                # units
                out_of += 1
                vtx_units = vtx_entry.get("units", "")
                if self.xrds[vtx_var].attrs.get("units", "") != vtx_units:
                    messages.append(
                        f"Vertex variable '{vtx_var}' units="
                        f"'{self.xrds[vtx_var].attrs.get('units', '')}', "
                        f"expected '{vtx_units}'."
                    )
                else:
                    score += 1
                # Must be 2-D (ncells, ncorners)
                out_of += 1
                if len(self.xrds[vtx_var].dims) != 2:
                    messages.append(
                        f"Vertex variable '{vtx_var}' must have 2 dimensions "
                        f"(ncells, ncorners); "
                        f"found dims={list(self.xrds[vtx_var].dims)}."
                    )
                else:
                    score += 1

        # Data variable must have lat/lon auxiliary coords in 'coordinates' attribute
        if data_var_out_name and data_var_out_name in self.xrds:
            coord_attr = self.xrds[data_var_out_name].attrs.get("coordinates", "")
            coord_vars_listed = coord_attr.split()
            for dim_id, exp_sn in [
                ("latitude", grid_var_entries.get("latitude", {}).get("standard_name", "latitude")),
                ("longitude", grid_var_entries.get("longitude", {}).get("standard_name", "longitude")),
            ]:
                if dim_id not in self.requested_dims:
                    continue
                out_of += 1
                found_in_coords = any(
                    self.xrds[v].attrs.get("standard_name") == exp_sn
                    for v in coord_vars_listed
                    if v in self.xrds
                )
                if not found_in_coords:
                    messages.append(
                        f"Data variable '{data_var_out_name}' 'coordinates' attribute "
                        f"must include the {dim_id} auxiliary coordinate "
                        f"(standard_name='{exp_sn}')."
                    )
                else:
                    score += 1

        if out_of == 0:
            return self.make_result(level, 1, 1, desc, messages)
        return self.make_result(level, score, out_of, desc, messages)

    # ------------------------------------------------------------------

    def check_vertical(self, ds):
        """Verify vertical coordinate(s) against the CMIP7 coordinate table."""
        desc = "Vertical coordinates"
        level = BaseCheck.HIGH
        messages = []

        vert_dims = [d for d in self.requested_dims if d in _VERTICAL_GENERIC_IDS]
        if not vert_dims:
            return self.make_result(level, 1, 1, desc, messages)

        if self.model_type is None:
            return self.make_result(
                level, 0, 1, desc,
                ["source_id contains neither 'AWI' nor 'ICON'; "
                 "cannot resolve generic vertical coordinate mapping."],
            )

        out_of = 0
        score = 0
        axis_entries = self.CTcoords.get("axis_entry", {})
        formula_entries = self.CTformulas.get("formula_entry", {})

        for generic_id in vert_dims:
            coord_key = _VERTICAL_MAPPING[self.model_type].get(generic_id)
            if not coord_key:
                out_of += 1
                messages.append(
                    f"No vertical coordinate mapping for generic_id='{generic_id}' "
                    f"and model_type='{self.model_type}'."
                )
                continue

            ce = axis_entries.get(coord_key, {})
            out_name = ce.get("out_name", "lev")
            expected_sn = ce.get("standard_name", "")
            expected_units = ce.get("units", "")
            expected_positive = ce.get("positive", "")
            must_have_bounds = ce.get("must_have_bounds", "no") == "yes"
            z_factors_str = ce.get("z_factors", "")
            z_bounds_factors_str = ce.get("z_bounds_factors", "")

            # Locate the lev variable
            out_of += 1
            lev_var = self._find_var_by(name=out_name, standard_name=expected_sn, axis="Z")
            if lev_var is None:
                messages.append(
                    f"Vertical coordinate variable not found for '{generic_id}' "
                    f"(expected out_name='{out_name}', "
                    f"standard_name='{expected_sn}')."
                )
                continue
            score += 1

            # axis=Z
            out_of += 1
            if self.xrds[lev_var].attrs.get("axis", "") != "Z":
                messages.append(
                    f"Vertical coordinate '{lev_var}' must have attribute axis='Z'."
                )
            else:
                score += 1

            # standard_name
            if expected_sn:
                out_of += 1
                if self.xrds[lev_var].attrs.get("standard_name", "") != expected_sn:
                    messages.append(
                        f"'{lev_var}' standard_name="
                        f"'{self.xrds[lev_var].attrs.get('standard_name', '')}', "
                        f"expected '{expected_sn}'."
                    )
                else:
                    score += 1

            # units
            if expected_units:
                out_of += 1
                if self.xrds[lev_var].attrs.get("units", "") != expected_units:
                    messages.append(
                        f"'{lev_var}' units="
                        f"'{self.xrds[lev_var].attrs.get('units', '')}', "
                        f"expected '{expected_units}'."
                    )
                else:
                    score += 1

            # positive
            if expected_positive:
                out_of += 1
                if self.xrds[lev_var].attrs.get("positive", "") != expected_positive:
                    messages.append(
                        f"'{lev_var}' positive="
                        f"'{self.xrds[lev_var].attrs.get('positive', '')}', "
                        f"expected '{expected_positive}'."
                    )
                else:
                    score += 1

            # bounds
            if must_have_bounds:
                bnds_name = f"{lev_var}_bnds"
                out_of += 1
                # lev must declare its bounds
                declared_bnds = self.xrds[lev_var].attrs.get("bounds", "")
                if declared_bnds != bnds_name:
                    messages.append(
                        f"'{lev_var}' bounds attribute is '{declared_bnds}', "
                        f"expected '{bnds_name}'."
                    )
                else:
                    score += 1

                out_of += 1
                if bnds_name not in self.xrds:
                    messages.append(
                        f"Bounds variable '{bnds_name}' for '{lev_var}' not found in file."
                    )
                else:
                    score += 1
                    # lev_bnds must have NO attributes
                    out_of += 1
                    if self.xrds[bnds_name].attrs:
                        messages.append(
                            f"'{bnds_name}' must have no attributes; "
                            f"found: {list(self.xrds[bnds_name].attrs.keys())}."
                        )
                    else:
                        score += 1

            # formula_terms (for hybrid / formula-based coordinates)
            if z_factors_str:
                out_of += 1
                ft_attr = self.xrds[lev_var].attrs.get("formula_terms", "")
                if not ft_attr:
                    messages.append(
                        f"'{lev_var}' must have 'formula_terms' attribute "
                        f"(formula: '{ce.get('formula', '')}')."
                    )
                else:
                    score += 1
                    ft_map = _parse_formula_terms(ft_attr)
                    for term, var_name in ft_map.items():
                        out_of += 1
                        if var_name not in self.xrds:
                            messages.append(
                                f"Formula term '{term}' references '{var_name}' "
                                f"which is not found in the file."
                            )
                        else:
                            score += 1
                            # Verify formula term variable attributes against formula_terms table
                            ft_entry = _find_formula_entry(formula_entries, var_name, generic_id)
                            if ft_entry:
                                out_of, score, msgs = _check_formula_var_attrs(
                                    self.xrds[var_name], var_name, ft_entry, out_of, score
                                )
                                messages.extend(msgs)

            # bounds formula terms
            if must_have_bounds and z_bounds_factors_str:
                zbf_map = _parse_formula_terms(z_bounds_factors_str)
                for term, var_name in zbf_map.items():
                    out_of += 1
                    if var_name not in self.xrds:
                        messages.append(
                            f"Bounds formula term '{term}' references '{var_name}' "
                            f"which is not found in the file."
                        )
                    else:
                        score += 1
                        ft_entry = _find_formula_entry(formula_entries, var_name, generic_id)
                        if ft_entry:
                            out_of, score, msgs = _check_formula_var_attrs(
                                self.xrds[var_name], var_name, ft_entry, out_of, score
                            )
                            messages.extend(msgs)

        if out_of == 0:
            return self.make_result(level, 1, 1, desc, messages)
        return self.make_result(level, score, out_of, desc, messages)

    # ------------------------------------------------------------------

    def check_time(self, ds):
        """Verify time coordinate(s) against the CMIP7 coordinate table."""
        desc = "Time coordinate"
        level = BaseCheck.HIGH
        messages = []

        time_dims = [d for d in self.requested_dims if _is_time_dim(d)]
        if not time_dims:
            return self.make_result(level, 1, 1, desc, messages)

        out_of = 0
        score = 0
        axis_entries = self.CTcoords.get("axis_entry", {})

        for time_dim_id in time_dims:
            ce = axis_entries.get(time_dim_id, {})
            out_name = ce.get("out_name", "time")
            must_have_bounds = ce.get("must_have_bounds", "no") == "yes"
            is_climatology = ce.get("climatology", "") == "yes"

            # Locate time variable
            out_of += 1
            time_var = self._find_var_by(
                name=out_name, standard_name="time", axis="T"
            )
            if time_var is None:
                messages.append(
                    f"Time coordinate variable '{out_name}' (dim_id='{time_dim_id}') "
                    f"not found in file."
                )
                continue
            score += 1

            # axis=T
            out_of += 1
            if self.xrds[time_var].attrs.get("axis", "") != "T":
                messages.append(
                    f"Time variable '{time_var}' must have attribute axis='T'."
                )
            else:
                score += 1

            # standard_name=time
            out_of += 1
            if self.xrds[time_var].attrs.get("standard_name", "") != "time":
                messages.append(
                    f"Time variable '{time_var}' standard_name="
                    f"'{self.xrds[time_var].attrs.get('standard_name', '')}', "
                    f"expected 'time'."
                )
            else:
                score += 1

            # units: must contain "since"
            out_of += 1
            units = self.xrds[time_var].attrs.get(
                "units", self.xrds[time_var].encoding.get("units", "")
            )
            if not units or "since" not in str(units):
                messages.append(
                    f"Time variable '{time_var}' units='{units}' is missing or invalid "
                    f"(expected format 'X since Y')."
                )
            else:
                score += 1

            # calendar attribute
            out_of += 1
            calendar = self.xrds[time_var].attrs.get(
                "calendar", self.xrds[time_var].encoding.get("calendar", "")
            )
            if not calendar:
                messages.append(
                    f"Time variable '{time_var}' is missing 'calendar' attribute."
                )
            else:
                score += 1

            if must_have_bounds:
                if is_climatology:
                    # Expect a 'climatology' attribute pointing to the climatology-bounds variable
                    out_of += 1
                    clim_attr = self.xrds[time_var].attrs.get("climatology", "")
                    if not clim_attr:
                        messages.append(
                            f"Climatology time variable '{time_var}' must have a "
                            f"'climatology' attribute pointing to the bounds variable."
                        )
                    else:
                        score += 1
                        out_of += 1
                        if clim_attr not in self.xrds:
                            messages.append(
                                f"Climatology bounds variable '{clim_attr}' "
                                f"(referenced by '{time_var}:climatology') "
                                f"not found in file."
                            )
                        else:
                            score += 1
                else:
                    # Regular time bounds
                    bnds_name = f"{out_name}_bnds"
                    out_of += 1
                    declared_bnds = self.xrds[time_var].attrs.get("bounds", "")
                    if declared_bnds != bnds_name:
                        messages.append(
                            f"'{time_var}' bounds attribute='{declared_bnds}', "
                            f"expected '{bnds_name}'."
                        )
                    else:
                        score += 1

                    out_of += 1
                    if bnds_name not in self.xrds:
                        messages.append(
                            f"Time bounds variable '{bnds_name}' not found in file."
                        )
                    else:
                        score += 1
                        # time_bnds must have NO attributes
                        out_of += 1
                        if self.xrds[bnds_name].attrs:
                            messages.append(
                                f"'{bnds_name}' must have no attributes; "
                                f"found: {list(self.xrds[bnds_name].attrs.keys())}."
                            )
                        else:
                            score += 1

        if out_of == 0:
            return self.make_result(level, 1, 1, desc, messages)
        return self.make_result(level, score, out_of, desc, messages)

    # ------------------------------------------------------------------

    def check_coord(self, ds):
        """Verify non-grid, non-vertical, non-time coordinate dimensions."""
        desc = "Other coordinates"
        level = BaseCheck.HIGH
        messages = []

        other_dims = [
            d for d in self.requested_dims
            if d not in _HORIZONTAL_DIM_IDS
            and d not in _VERTICAL_GENERIC_IDS
            and not _is_time_dim(d)
        ]
        if not other_dims:
            return self.make_result(level, 1, 1, desc, messages)

        out_of = 0
        score = 0
        axis_entries = self.CTcoords.get("axis_entry", {})
        data_out_name = (
            self.var_entry.get("out_name", "") if self.var_entry else ""
        )

        for dim_id in other_dims:
            ce = axis_entries.get(dim_id)
            if ce is None:
                out_of += 1
                messages.append(
                    f"dim_id '{dim_id}' not found in CMIP7 coordinate table."
                )
                continue

            out_name = ce.get("out_name", dim_id)
            coord_type = ce.get("type", "")
            value = ce.get("value", "")
            requested = _as_list(ce.get("requested", []))
            requested_bounds = _as_list(ce.get("requested_bounds", []))
            must_have_bounds = ce.get("must_have_bounds", "no") == "yes"
            is_character = coord_type == "character"
            is_multi = bool(requested)

            if is_multi:
                out_of, score, msgs = self._check_multi_value_coord(
                    dim_id, out_name, ce, requested, requested_bounds,
                    must_have_bounds, is_character, out_of, score
                )
            else:
                out_of, score, msgs = self._check_scalar_coord(
                    dim_id, out_name, ce, value, is_character,
                    data_out_name, out_of, score
                )
            messages.extend(msgs)

        if out_of == 0:
            return self.make_result(level, 1, 1, desc, messages)
        return self.make_result(level, score, out_of, desc, messages)

    # ------------------------------------------------------------------

    def check_dimensions(self, ds):
        """Verify that the data variable's dimensions are in the expected C order."""
        desc = "Variable dimension ordering"
        level = BaseCheck.HIGH
        messages = []

        if self.var_entry is None:
            return self.make_result(
                level, 0, 1, desc,
                ["Cannot check dimensions: CMOR table entry not resolved."],
            )

        data_out_name = self.var_entry.get("out_name", "")
        if not data_out_name or data_out_name not in self.xrds:
            return self.make_result(level, 1, 1, desc, messages)

        axis_entries = self.CTcoords.get("axis_entry", {})
        has_horizontal = any(d in _HORIZONTAL_DIM_IDS for d in self.requested_dims)

        # Build expected C-order dims (reverse of CMOR Fortran order, scalars excluded)
        expected = []
        for dim_id in reversed(self.requested_dims):
            if dim_id in _HORIZONTAL_DIM_IDS:
                continue  # handled as single cell-dim placeholder below
            if dim_id in _VERTICAL_GENERIC_IDS:
                expected.append("lev")
                continue
            if _is_time_dim(dim_id):
                ce = axis_entries.get(dim_id, {})
                expected.append(ce.get("out_name", "time"))
                continue
            ce = axis_entries.get(dim_id, {})
            if ce:
                if _is_scalar_coord(ce):
                    continue  # scalar coords not in variable dims
                expected.append(ce.get("out_name", dim_id))

        # Insert horizontal placeholder at the innermost (last) position
        if has_horizontal:
            expected.append("<ncells>")

        actual = list(self.xrds[data_out_name].dims)
        out_of = 1
        score = 0

        if len(actual) != len(expected):
            messages.append(
                f"Variable '{data_out_name}' has {len(actual)} dimension(s) "
                f"{actual}; expected {len(expected)} based on CMOR dims "
                f"{self.requested_dims} → {expected}."
            )
        else:
            mismatches = []
            for i, (exp, act) in enumerate(zip(expected, actual)):
                if exp == "<ncells>":
                    continue  # any name acceptable for unstructured cell dim
                if exp != act:
                    mismatches.append(f"position {i}: '{act}' (expected '{exp}')")
            if mismatches:
                messages.append(
                    f"Variable '{data_out_name}' dimension order mismatch: "
                    + "; ".join(mismatches)
                    + f". CMOR dims: {self.requested_dims}."
                )
            else:
                score = 1

        return self.make_result(level, score, out_of, desc, messages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_var_by(self, name=None, standard_name=None, axis=None):
        """Return the first file variable matching the given criteria."""
        for v in self._all_file_vars:
            if name and v == name:
                return v
            if standard_name and self.xrds[v].attrs.get("standard_name") == standard_name:
                if axis is None or self.xrds[v].attrs.get("axis") == axis:
                    return v
            if axis and not standard_name and self.xrds[v].attrs.get("axis") == axis:
                return v
        return None

    def _check_scalar_coord(self, dim_id, out_name, ce, value, is_character,
                            data_out_name, out_of, score):
        """Check a scalar coordinate (dimensionless, referenced in 'coordinates')."""
        messages = []

        # Locate the coordinate variable (allow lookup by standard_name as fallback)
        out_of += 1
        coord_var = None
        if out_name in self.xrds:
            coord_var = out_name
        else:
            sn = ce.get("standard_name", "")
            if sn:
                coord_var = self._find_var_by(standard_name=sn)

        if coord_var is None:
            messages.append(
                f"Scalar coordinate '{out_name}' (dim_id='{dim_id}') not found in file."
            )
            return out_of, score, messages
        score += 1

        if is_character:
            # Character scalar: only dimension is 'strlen'
            out_of += 1
            dims = list(self.xrds[coord_var].dims)
            if dims != ["strlen"]:
                messages.append(
                    f"Character scalar coordinate '{coord_var}' must have only "
                    f"dimension 'strlen'; found {dims}."
                )
            else:
                score += 1
        else:
            # Numeric scalar: fully dimensionless
            out_of += 1
            if self.xrds[coord_var].dims:
                messages.append(
                    f"Scalar coordinate '{coord_var}' (dim_id='{dim_id}') must be "
                    f"dimensionless; found dims={list(self.xrds[coord_var].dims)}."
                )
            else:
                score += 1
            # Verify value
            if value:
                out_of += 1
                try:
                    actual = float(np.asarray(self.xrds[coord_var].values).flat[0])
                    expected = float(value)
                    if not np.isclose(actual, expected, rtol=1e-5, atol=0):
                        messages.append(
                            f"Scalar coordinate '{coord_var}' value={actual}, "
                            f"expected {expected}."
                        )
                    else:
                        score += 1
                except (TypeError, ValueError):
                    score += 1  # non-numeric – skip value check

        # Data variable must list this coord in its 'coordinates' attribute
        if data_out_name and data_out_name in self.xrds:
            out_of += 1
            coord_attr = self.xrds[data_out_name].attrs.get("coordinates", "")
            if out_name not in coord_attr.split():
                messages.append(
                    f"Data variable '{data_out_name}' 'coordinates' attribute must "
                    f"include scalar coordinate '{out_name}' (dim_id='{dim_id}'); "
                    f"current value: '{coord_attr}'."
                )
            else:
                score += 1

        return out_of, score, messages

    def _check_multi_value_coord(self, dim_id, out_name, ce, requested,
                                  requested_bounds, must_have_bounds, is_character,
                                  out_of, score):
        """Check a multi-value coordinate (dimension coordinate with requested values)."""
        messages = []

        # Locate coordinate variable
        out_of += 1
        coord_var = None
        if out_name in self.xrds:
            coord_var = out_name
        else:
            sn = ce.get("standard_name", "")
            if sn:
                coord_var = self._find_var_by(standard_name=sn)

        if coord_var is None:
            messages.append(
                f"Coordinate variable '{out_name}' (dim_id='{dim_id}') not found in file."
            )
            return out_of, score, messages
        score += 1

        if is_character:
            # Character dimension coordinate: dims must be (out_name, strlen)
            out_of += 1
            dims = list(self.xrds[coord_var].dims)
            if len(dims) != 2 or dims[1] != "strlen":
                messages.append(
                    f"Character coordinate '{coord_var}' must have dims "
                    f"('{out_name}', 'strlen'); found {dims}."
                )
            else:
                score += 1

            # All requested string values must be present
            if requested:
                out_of += 1
                try:
                    raw = self.xrds[coord_var].values
                    vals = [
                        b"".join(row).decode("utf-8", errors="replace")
                        .rstrip("\x00").strip()
                        for row in raw
                    ]
                    missing = [r for r in requested if r not in vals]
                    if missing:
                        messages.append(
                            f"Character coordinate '{coord_var}' is missing requested "
                            f"value(s) {missing}; found {vals}."
                        )
                    else:
                        score += 1
                except Exception as exc:
                    messages.append(
                        f"Could not decode character coordinate '{coord_var}': {exc}"
                    )
        else:
            # Numeric dimension coordinate
            tol_str = ce.get("tolerance", "")
            tol = float(tol_str) if tol_str else 1e-6

            # Check all requested values are present
            if requested:
                out_of += 1
                try:
                    file_vals = list(np.asarray(self.xrds[coord_var].values).flat)
                    req_floats = [float(r) for r in requested]
                    missing = [
                        r for r in req_floats
                        if not any(abs(r - fv) <= tol for fv in file_vals)
                    ]
                    if missing:
                        messages.append(
                            f"Coordinate '{coord_var}' is missing requested value(s) "
                            f"{missing} (tolerance={tol})."
                        )
                    else:
                        score += 1
                except Exception as exc:
                    messages.append(
                        f"Could not check values of coordinate '{coord_var}': {exc}"
                    )

            # Check bounds if required
            if must_have_bounds:
                # Resolve bounds variable name
                bnds_attr = self.xrds[coord_var].attrs.get("bounds", "")
                bnds_name = bnds_attr if bnds_attr else f"{coord_var}_bnds"

                out_of += 1
                if not bnds_attr:
                    messages.append(
                        f"Coordinate '{coord_var}' must have a 'bounds' attribute "
                        f"set to '{bnds_name}'."
                    )
                else:
                    score += 1

                out_of += 1
                if bnds_name not in self.xrds:
                    messages.append(
                        f"Bounds variable '{bnds_name}' for '{coord_var}' not found."
                    )
                else:
                    score += 1
                    # bounds must have NO attributes
                    out_of += 1
                    if self.xrds[bnds_name].attrs:
                        messages.append(
                            f"Bounds variable '{bnds_name}' must have no attributes; "
                            f"found {list(self.xrds[bnds_name].attrs.keys())}."
                        )
                    else:
                        score += 1

                    # Verify requested bound pairs are present
                    if requested_bounds:
                        out_of += 1
                        try:
                            file_bnds = np.asarray(self.xrds[bnds_name].values).reshape(-1, 2)
                            req_pairs = list(zip(
                                [float(requested_bounds[i]) for i in range(0, len(requested_bounds), 2)],
                                [float(requested_bounds[i]) for i in range(1, len(requested_bounds), 2)],
                            ))
                            missing_pairs = [
                                pair for pair in req_pairs
                                if not any(
                                    abs(pair[0] - fb[0]) <= tol
                                    and abs(pair[1] - fb[1]) <= tol
                                    for fb in file_bnds
                                )
                            ]
                            if missing_pairs:
                                messages.append(
                                    f"Bounds '{bnds_name}' missing requested pair(s) "
                                    f"{missing_pairs} (tolerance={tol})."
                                )
                            else:
                                score += 1
                        except Exception as exc:
                            messages.append(
                                f"Could not check bounds of '{coord_var}': {exc}"
                            )

        return out_of, score, messages


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_formula_terms(ft_str: str) -> dict:
    """Parse a CF formula_terms string like 'ap: ap b: b ps: ps'."""
    result = {}
    parts = ft_str.split()
    i = 0
    while i + 1 < len(parts):
        if parts[i].endswith(":"):
            result[parts[i].rstrip(":")] = parts[i + 1]
            i += 2
        else:
            i += 1
    return result


def _find_formula_entry(formula_entries: dict, var_name: str, generic_id: str) -> dict:
    """Return the formula_terms table entry whose out_name and dimension match."""
    for entry in formula_entries.values():
        if entry.get("out_name") == var_name and generic_id in entry.get("dimensions", ""):
            return entry
    return {}


def _check_formula_var_attrs(xr_var, var_name: str, ft_entry: dict,
                              out_of: int, score: int):
    """Spot-check units and standard_name of a formula-term variable."""
    messages = []
    expected_units = ft_entry.get("units", "")
    expected_sn = ft_entry.get("standard_name", "")

    if expected_units:
        out_of += 1
        if xr_var.attrs.get("units", "") != expected_units:
            messages.append(
                f"Formula term variable '{var_name}' units="
                f"'{xr_var.attrs.get('units', '')}', expected '{expected_units}'."
            )
        else:
            score += 1

    if expected_sn:
        out_of += 1
        if xr_var.attrs.get("standard_name", "") != expected_sn:
            messages.append(
                f"Formula term variable '{var_name}' standard_name="
                f"'{xr_var.attrs.get('standard_name', '')}', expected '{expected_sn}'."
            )
        else:
            score += 1

    return out_of, score, messages


def _as_list(val) -> list:
    """Normalise a CMIP7 table 'requested' / 'requested_bounds' field to a list."""
    if not val:
        return []
    if isinstance(val, list):
        return [v for v in val if v != ""]
    if isinstance(val, str):
        return [val] if val else []
    return list(val)


def _is_scalar_coord(ce: dict) -> bool:
    """Return True if the coordinate entry represents a scalar in the file."""
    requested = _as_list(ce.get("requested", []))
    return not requested  # no requested list → scalar (value may or may not be set)
