"""Integration tests for checker setup and CMOR-table resolution."""

import json

from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from cc_plugin_aicc.aicc import AICC


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _minimal_tables(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    _write_json(
        tables / "CMIP7_coordinate.json",
        {"axis_entry": {"test_hybrid": {"generic_level_name": "alevel"}}},
    )
    _write_json(tables / "CMIP7_grids.json", {"variable_entry": {}})
    _write_json(tables / "CMIP7_formula_terms.json", {"formula_entry": {}})
    _write_json(
        tables / "CMIP7_atmos.json",
        {"variable_entry": {"ta": {"out_name": "ta", "dimensions": ["alevel"]}}},
    )
    return tables


def _messages(results):
    return [message for result in results for message in result.msgs]


def test_setup_loads_named_config_options_and_resolves_variable(tmp_path):
    tables = _minimal_tables(tmp_path)
    vertical_config = tmp_path / "vertical_config.json"
    grid_config = tmp_path / "grid_config.json"
    _write_json(
        vertical_config,
        {"TestModel": {"vertical": {"alevel": "test_hybrid"}}},
    )
    _write_json(grid_config, {"test-grid": "curvilinear"})

    path = tmp_path / "input.nc"
    with Dataset(path, "w") as nc:
        nc.source_id = "TestModel-1"
        nc.grid_label = "test-grid"
        nc.branded_variable = "ta"
        nc.table_id = "atmos"

    checker = AICC(
        options={
            "tables": str(tables),
            "vertical_config": str(vertical_config),
            "grid_config": str(grid_config),
        }
    )
    with Dataset(path) as nc:
        checker.setup(nc)
        results = checker.check_branded_variable(nc)

    assert checker._conf_key == "TestModel"
    assert checker._vert_mapping == {"alevel": "test_hybrid"}
    assert checker._grid_type == "curvilinear"
    assert checker._grid_type_known is True
    assert checker.table_name == "atmos"
    assert checker.requested_dims == ["alevel"]
    assert _messages(results) == []


def test_branded_variable_reports_missing_and_unresolvable_values():
    checker = AICC()
    checker.branded_variable = None
    checker.var_entry = None
    missing = checker.check_branded_variable(type("Dataset", (), {})())
    assert any("is not set" in message for message in _messages(missing))

    class DatasetStub:
        table_id = "atmos"
        realm = "atmos"

        @staticmethod
        def ncattrs():
            return ["table_id", "realm"]

        @staticmethod
        def getncattr(name):
            return getattr(DatasetStub, name)

    checker.branded_variable = "unknown"
    unresolved = checker.check_branded_variable(DatasetStub())
    assert any(
        "not found in any CMIP7 table" in message for message in _messages(unresolved)
    )
    assert unresolved[0].weight == BaseCheck.HIGH
