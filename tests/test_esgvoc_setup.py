"""Tests for the optional ESGVoc metadata setup path."""

import pytest
from netCDF4 import Dataset

from cc_plugin_aicc.aicc import AICC
from cc_plugin_aicc.esgvoc_adapter import (
    _require_supported_esgvoc_version,
    load_esgvoc_metadata,
)


class FakeESGVocAPI:
    def __init__(self, branded=None, descriptors=None):
        self.branded = branded
        self.descriptors = descriptors or {}
        self.calls = []

    def get_term_in_data_descriptor(self, descriptor, identifier, fields):
        self.calls.append(("one", descriptor, identifier, fields))
        assert descriptor == "known_branded_variable"
        return self.branded

    def get_all_terms_in_collection(self, project, descriptor, fields):
        self.calls.append(("all", project, descriptor, fields))
        assert project == "cmip7"
        return self.descriptors.get(descriptor, [])


def _descriptor_fixture():
    generic = {"id": "alevel"}
    formula_terms = [
        {
            "id": "ap",
            "data_type": "double",
            "out_name": "ap",
            "units": "Pa",
            "dimensions": [generic],
        },
        {
            "id": "ap_bnds",
            "data_type": "double",
            "out_name": "ap_bnds",
            "units": "Pa",
            "dimensions": [generic],
        },
        {
            "id": "b",
            "data_type": "double",
            "out_name": "b",
            "dimensions": [generic],
        },
        {
            "id": "b_bnds",
            "data_type": "double",
            "out_name": "b_bnds",
            "dimensions": [generic],
        },
        {
            "id": "ps",
            "data_type": "real",
            "out_name": "ps",
            "units": "Pa",
            "dimensions": [{"id": "latitude"}, {"id": "longitude"}],
        },
        {
            "id": "ptop",
            "data_type": "double",
            "out_name": "ptop",
            "units": "Pa",
            "dimensions": None,
        },
    ]
    return {
        "data_coordinate": [
            {
                "id": "longitude",
                "coordinate_type": {"id": "generic_horizontal"},
                "axis": "X",
                "data_type": "double",
                "out_name": "lon",
                "units": "degrees_east",
            },
            {
                "id": "latitude",
                "coordinate_type": {"id": "generic_horizontal"},
                "axis": "Y",
                "data_type": "double",
                "out_name": "lat",
                "units": "degrees_north",
            },
            {
                "id": "time",
                "coordinate_type": {"id": "standard_1d"},
                "axis": "T",
                "data_type": "double",
                "out_name": "time",
                "units": "days since ?",
            },
            {
                "id": "height2m",
                "coordinate_type": {"id": "scalar"},
                "axis": "Z",
                "data_type": "double",
                "out_name": "height",
                "units": "m",
                "coordinate_values": [0],
                "coordinate_bounds": [-1, 1],
                "bounds_required": False,
                "valid_min": 0,
            },
            {
                "id": "plev3",
                "coordinate_type": {"id": "standard_1d"},
                "axis": "Z",
                "data_type": "double",
                "out_name": "plev",
                "units": "Pa",
                "coordinate_values": [100000, 85000, 50000],
                "coordinate_bounds": [110000, 92500, 67500, 40000],
                "bounds_required": True,
                "tolerance": 1,
            },
        ],
        "formula_term": formula_terms,
        "model_level_coordinate": [
            {
                "id": "alternate_hybrid_sigma",
                "axis": "Z",
                "data_type": "double",
                "cf_standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
                "out_name": "lev",
                "units": "1",
                "positive": "down",
                "stored_direction": "decreasing",
                "bounds_required": True,
                "formula": "p = ap + b*ps",
                # This deliberately mirrors the aliased relationship returned
                # by the currently installed ESGVoc database.
                "z_factors": [formula_terms[1], formula_terms[3], formula_terms[4]],
                "z_bounds_factors": [
                    formula_terms[1],
                    formula_terms[3],
                    formula_terms[4],
                ],
                "generic_level_name": generic,
            },
            {
                "id": "standard_sigma",
                "axis": "Z",
                "data_type": "double",
                "cf_standard_name": "atmosphere_sigma_coordinate",
                "out_name": "lev",
                "units": "1",
                "positive": "down",
                "stored_direction": "decreasing",
                "bounds_required": True,
                "formula": "p = ptop + sigma*(ps - ptop)",
                "z_factors": [formula_terms[5], formula_terms[4]],
                "z_bounds_factors": [formula_terms[5], formula_terms[4]],
                "generic_level_name": generic,
            },
        ],
        "grid_variable": [
            {
                "id": "latitude",
                "data_type": "double",
                "cf_standard_name": "latitude",
                "out_name": "latitude",
                "units": "degrees_north",
                "dimensions": [{"id": "longitude"}, {"id": "latitude"}],
            }
        ],
        "grid_axis": [
            {
                "id": "grid_latitude",
                "axis": "Y",
                "data_type": "double",
                "cf_standard_name": "grid_latitude",
                "out_name": "rlat",
                "units": "degrees",
            }
        ],
    }


def test_esgvoc_adapter_builds_equivalent_checker_metadata():
    api = FakeESGVocAPI(
        branded={
            "id": "tas_tavg-h2m-hxy-u",
            "out_name": "tas",
            "dimensions": [
                {"id": "longitude"},
                {"id": "latitude"},
                {"id": "height2m"},
                {"id": "time"},
            ],
            "table_id": [{"id": "atmos"}],
        },
        descriptors=_descriptor_fixture(),
    )

    metadata = load_esgvoc_metadata("tas_tavg-h2m-hxy-u", api=api)

    assert metadata["variable"] == {
        "out_name": "tas",
        "dimensions": ["longitude", "latitude", "height2m", "time"],
    }
    assert metadata["table_name"] == "atmos"

    entries = metadata["coordinates"]["axis_entry"]
    assert entries["height2m"]["coordinate_type"] == "scalar"
    assert entries["height2m"]["value"] == "0"
    assert entries["height2m"]["bounds_values"] == "-1 1"
    assert entries["height2m"]["valid_min"] == "0"
    assert entries["plev3"]["requested_bounds"] == [
        110000,
        92500,
        92500,
        67500,
        67500,
        40000,
    ]

    model_level = entries["alternate_hybrid_sigma"]
    assert model_level["generic_level_name"] == "alevel"
    assert model_level["z_factors"] == "ap: ap b: b ps: ps"
    assert model_level["z_bounds_factors"] == "ap: ap_bnds b: b_bnds ps: ps"
    assert entries["standard_sigma"]["z_factors"] == ("sigma: lev ptop: ptop ps: ps")
    assert entries["standard_sigma"]["z_bounds_factors"] == (
        "sigma: lev_bnds ptop: ptop ps: ps"
    )

    assert metadata["grids"]["variable_entry"]["latitude"]["standard_name"] == (
        "latitude"
    )
    assert metadata["grids"]["axis_entry"]["grid_latitude"]["out_name"] == ("rlat")
    assert metadata["formulas"]["formula_entry"]["ap"]["dimensions"] == ["alevel"]


def test_esgvoc_adapter_handles_unknown_branded_variable():
    metadata = load_esgvoc_metadata(
        "missing", api=FakeESGVocAPI(descriptors=_descriptor_fixture())
    )
    assert metadata["variable"] is None
    assert metadata["table_name"] is None


def test_esgvoc_version_must_be_at_least_5_1_0():
    for unsupported in ("4.9.9", "5.0.1", "5.1.0.dev1"):
        try:
            _require_supported_esgvoc_version(unsupported)
        except RuntimeError as exc:
            assert "requires esgvoc>=5.1.0" in str(exc)
            assert f"esgvoc=={unsupported}" in str(exc)
        else:
            raise AssertionError(f"esgvoc=={unsupported} should have been rejected")

    assert _require_supported_esgvoc_version("5.1.0") == "5.1.0"


def test_setup_uses_esgvoc_without_reading_cmor_tables(tmp_path, monkeypatch):
    path = tmp_path / "input.nc"
    with Dataset(path, "w") as nc:
        nc.source_id = "NoConfiguredModel"
        nc.grid_label = "g100"
        nc.branded_variable = "tas_tavg-h2m-hxy-u"

    metadata = {
        "coordinates": {"axis_entry": {"height2m": {"out_name": "height"}}},
        "grids": {"variable_entry": {}, "axis_entry": {}},
        "formulas": {"formula_entry": {}},
        "variable": {"out_name": "tas", "dimensions": ["height2m"]},
        "table_name": "atmos",
    }
    monkeypatch.setattr(
        "cc_plugin_aicc.aicc.load_esgvoc_metadata", lambda branded: metadata
    )

    checker = AICC(options={"esgvoc": None, "tables": "/does/not/exist"})
    monkeypatch.setattr(
        checker,
        "_read_cmip7_tables",
        lambda path: (_ for _ in ()).throw(AssertionError("CMOR path was used")),
    )
    with Dataset(path) as nc:
        checker.setup(nc)
        results = checker.check_branded_variable(nc)

    assert checker._metadata_source == "esgvoc"
    assert checker.requested_dims == ["height2m"]
    assert results[0].value == (1, 1)


def test_esgvoc_adapter_identifies_an_empty_collection():
    descriptors = _descriptor_fixture()
    descriptors["grid_axis"] = []

    with pytest.raises(RuntimeError, match="empty.*'grid_axis'.*incomplete"):
        load_esgvoc_metadata("missing", api=FakeESGVocAPI(descriptors=descriptors))


def test_esgvoc_adapter_identifies_a_failed_collection():
    class FailingAPI(FakeESGVocAPI):
        def get_all_terms_in_collection(self, project, descriptor, fields):
            if descriptor == "grid_variable":
                raise OSError("database unavailable")
            return super().get_all_terms_in_collection(project, descriptor, fields)

    with pytest.raises(
        RuntimeError,
        match="'grid_variable'.*OSError: database unavailable",
    ):
        load_esgvoc_metadata(
            "missing", api=FailingAPI(descriptors=_descriptor_fixture())
        )
