"""
aicc.py — AWI ICON Coordinate Checker (AICC) / AI Compliance Checker

Verifies CMIP7 coordinate compliance for AWI and ICON model output.

Variable discovery delegates to compliance_checker.cf.util (cfutil) for
consistent CF-convention application. Configuration lives in config.py;
pure utility functions live in utils.py.
"""

import json
import os
from pathlib import Path

from compliance_checker.base import BaseCheck, BaseNCCheck, Result, TestCtx

from cc_plugin_aicc import __version__
from cc_plugin_aicc.checks import (
    CoordinateChecks,
    GridChecks,
    QuantizationChecks,
    TimeChecks,
    VerticalChecks,
)
from cc_plugin_aicc.config import (
    DEFAULT_TABLES_PATH,
    REALM_TO_TABLE,
    load_grid_config,
    load_vertical_config,
    resolve_grid_type,
    resolve_vertical_config,
)
from cc_plugin_aicc.esgvoc_adapter import load_esgvoc_metadata
from cc_plugin_aicc.utils import _format_attribute, _ncattr

# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


class AICC(
    GridChecks,
    VerticalChecks,
    TimeChecks,
    CoordinateChecks,
    QuantizationChecks,
    BaseNCCheck,
    BaseCheck,
):
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
        "curvilinear": {
            "check": "_check_curvilinear_grid",
            "dimensions": "_expected_curvilinear_horizontal_dimensions",
            "table": "grids",
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

        # Resolve the model-specific vertical mapping.
        source_id = _ncattr(dataset, "source_id")
        vertical_config = load_vertical_config(self.options.get("vertical_config"))

        self._conf_key, vertical_conf = resolve_vertical_config(
            source_id, vertical_config
        )
        if vertical_conf:
            # Support both nested and legacy flat vertical configurations.
            if "vertical" in vertical_conf:
                self._vert_mapping = vertical_conf.get("vertical") or {}
            else:
                self._vert_mapping = vertical_conf
        else:
            self._vert_mapping = None

        # Resolve the globally registered grid_label independently of source_id.
        self._grid_label = _ncattr(dataset, "grid_label")
        grid_config = load_grid_config(self.options.get("grid_config"))
        self._grid_type, self._grid_type_known = resolve_grid_type(
            self._grid_label, grid_config
        )

        # Identify the variable and its metadata definition.
        self.branded_variable = _ncattr(dataset, "branded_variable") or None
        self.var_entry = None
        self.table_name = None
        self.requested_dims = []

        self._metadata_source = "esgvoc" if self._option_enabled("esgvoc") else "cmor"
        if self._metadata_source == "esgvoc":
            self._read_esgvoc_metadata()
        else:
            tables_path = self.options.get("tables", DEFAULT_TABLES_PATH)
            self._read_cmip7_tables(tables_path)
            if self.branded_variable:
                self._resolve_table_and_variable(dataset)

        if self.var_entry is not None:
            self.requested_dims = self.var_entry.get("dimensions", [])

    def _option_enabled(self, name):
        """Interpret a flag-style or explicitly valued checker option."""
        if name not in self.options:
            return False
        value = self.options[name]
        if value is None:
            return True
        return str(value).strip().lower() not in {"", "0", "false", "no", "off"}

    def _read_esgvoc_metadata(self):
        """Populate AICC metadata from ESGVoc rather than CMOR JSON files."""
        metadata = load_esgvoc_metadata(self.branded_variable or "")
        self.CTcoords = metadata["coordinates"]
        self.CTgrids = metadata["grids"]
        self.CTformulas = metadata["formulas"]
        self.CT = {}
        self._table_prefix = None
        self.var_entry = metadata["variable"]
        self.table_name = metadata["table_name"]

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
            source = (
                "the ESGVoc known_branded_variable descriptor"
                if getattr(self, "_metadata_source", "cmor") == "esgvoc"
                else "any CMIP7 table"
            )
            ctx.add_failure(
                f"branded_variable {_format_attribute(self.branded_variable)} "
                f"not found in {source} (file table_id="
                f"{_format_attribute(_ncattr(ds, 'table_id'))}, realm="
                f"{_format_attribute(_ncattr(ds, 'realm'))})."
            )
            return [ctx.to_result()]

        ctx.add_pass()
        return [ctx.to_result()]
