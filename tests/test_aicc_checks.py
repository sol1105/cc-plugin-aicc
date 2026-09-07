"""Focused tests for AICC coordinate attribute and direction checks."""

from contextlib import contextmanager

import numpy as np
import pytest
import xarray as xr
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from cc_plugin_aicc.aicc import AICC
from cc_plugin_aicc.utils import (
    _cmor_bound_tol_vals,
    _cmor_tol_val,
    _compare_units,
    _neutral_dtype,
)


TIME_ENTRY = {
    "out_name": "time",
    "standard_name": "time",
    "long_name": "Time Intervals",
    "units": "days since ?",
    "type": "double",
    "must_have_bounds": "no",
}

HYBRID_ENTRY = {
    "out_name": "lev",
    "generic_level_name": "alevel",
    "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
    "stored_direction": "decreasing",
    "positive": "down",
    "formula": "p = ap + b*ps",
    "z_factors": "ap: ap b: b ps: ps",
    "type": "double",
}

DEPTH_ENTRY = {
    "out_name": "depth",
    "standard_name": "depth",
    "stored_direction": "increasing",
    "positive": "down",
    "formula": "",
    "type": "double",
    "value": "",
    "bounds_values": "",
}

RHO_ENTRY = {
    "out_name": "rho",
    "standard_name": "sea_water_potential_density",
    "stored_direction": "increasing",
    "positive": "",
    "formula": "",
    "type": "double",
    "value": "",
    "bounds_values": "",
}

PLEV_ENTRY = {
    "out_name": "plev",
    "standard_name": "air_pressure",
    "stored_direction": "decreasing",
    "positive": "down",
    "formula": "",
    "type": "double",
    "value": "",
    "bounds_values": "",
}

UNSTRUCTURED_GRID_ENTRIES = {
    "latitude": {
        "out_name": "latitude",
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "type": "double",
    },
    "vertices_latitude": {
        "out_name": "vertices_latitude",
        "standard_name": "",
        "long_name": "",
        "units": "degrees_north",
        "type": "double",
    },
}

CURVILINEAR_GRID_ENTRIES = {
    "latitude": {
        "out_name": "latitude",
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "type": "double",
        "valid_min": "-90",
        "valid_max": "90",
    },
    "longitude": {
        "out_name": "longitude",
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "type": "double",
        "valid_min": "0",
        "valid_max": "360",
    },
    "vertices_latitude": {
        "out_name": "vertices_latitude",
        "units": "degrees_north",
        "type": "double",
        "valid_min": "-90",
        "valid_max": "90",
    },
    "vertices_longitude": {
        "out_name": "vertices_longitude",
        "units": "degrees_east",
        "type": "double",
        "valid_min": "0",
        "valid_max": "360",
    },
}

CURVILINEAR_AXIS_ENTRIES = {
    "grid_latitude": {
        "axis": "Y",
        "out_name": "rlat",
        "standard_name": "grid_latitude",
        "long_name": "latitude in rotated pole grid",
        "units": "degrees",
        "type": "double",
    },
    "grid_longitude": {
        "axis": "X",
        "out_name": "rlon",
        "standard_name": "grid_longitude",
        "long_name": "longitude in rotated pole grid",
        "units": "degrees",
        "type": "double",
    },
    "x": {
        "axis": "X",
        "out_name": "",
        "standard_name": "projection_x_coordinate",
        "long_name": "x coordinate of projection",
        "units": "m",
        "type": "double",
    },
    "y": {
        "axis": "Y",
        "out_name": "",
        "standard_name": "projection_y_coordinate",
        "long_name": "y coordinate of projection",
        "units": "m",
        "type": "double",
    },
    "x_deg": {
        "axis": "X",
        "out_name": "x",
        "standard_name": "projection_x_angular_coordinate",
        "long_name": "x angular coordinate of projection",
        "units": "degrees",
        "type": "double",
    },
    "y_deg": {
        "axis": "Y",
        "out_name": "y",
        "standard_name": "projection_y_angular_coordinate",
        "long_name": "y angular coordinate of projection",
        "units": "degrees",
        "type": "double",
    },
    "i_index": {
        "axis": "",
        "out_name": "i",
        "standard_name": "",
        "long_name": "first spatial index",
        "units": "1",
        "type": "integer",
    },
    "j_index": {
        "axis": "",
        "out_name": "j",
        "standard_name": "",
        "long_name": "second spatial index",
        "units": "1",
        "type": "integer",
    },
}


@contextmanager
def _open_netcdf(tmp_path, name, dataset):
    """Write an xarray dataset and yield it as the checker's netCDF4 input."""
    path = tmp_path / f"{name}.nc"
    dataset.to_netcdf(path, engine="netcdf4")
    with Dataset(path) as nc:
        yield nc


def _checker(requested_dims, axis_entries, *, var_entry=None, vert_mapping=None):
    checker = AICC()
    checker.requested_dims = requested_dims
    checker.CTcoords = {"axis_entry": axis_entries}
    checker.CTformulas = {"formula_entry": {}}
    checker.var_entry = var_entry or {"out_name": "tas"}
    checker._vert_mapping = vert_mapping
    checker._conf_key = "test-model"
    checker._grid_type = None
    checker._grid_type_known = False
    return checker


def _messages(results, severity=None):
    return [
        message
        for result in results
        if severity is None or result.weight == severity
        for message in result.msgs
    ]


@pytest.mark.parametrize(
    ("candidate", "required", "expected_level", "message_fragment"),
    [
        ("m", "m", "ok", ""),
        ("meter", "m", "fail", "convertible"),
        ("", "m", "fail", "units missing"),
        (
            "hours since 2000-01-01",
            "days since ?",
            "fail",
            "base unit 'hours' is convertible",
        ),
    ],
)
def test_compare_units_requires_an_exact_match(
    candidate, required, expected_level, message_fragment
):
    level, message = _compare_units(candidate, required)

    assert level == expected_level
    assert message_fragment in message


@pytest.mark.parametrize(
    ("dtype", "cmor_type"),
    [
        ("float64", "double"),
        ("float32", "real"),
        ("int32", "integer"),
        ("S1", "character"),
    ],
)
def test_neutral_dtype_distinguishes_cmor_storage_types(dtype, cmor_type):
    assert _neutral_dtype(np.asarray([], dtype=dtype)) == cmor_type


def test_time_long_name_is_suggested_but_convertible_units_are_required(tmp_path):
    dataset = xr.Dataset(
        data_vars={"tas": (("time",), [280.0, 281.0])},
        coords={"time": ("time", [0.0, 30.0])},
    )
    dataset["time"].attrs.update(
        {
            "axis": "T",
            "standard_name": "time",
            "long_name": "time",
            "units": "hours since 2000-01-01",
            "calendar": "proleptic_gregorian",
        }
    )
    checker = _checker(
        ["time"],
        {"time": TIME_ENTRY},
        var_entry={"out_name": "tas"},
    )

    with _open_netcdf(tmp_path, "time_attributes", dataset) as nc:
        results = checker.check_time(nc)

    required = _messages(results, BaseCheck.HIGH)
    suggested = _messages(results, BaseCheck.LOW)
    assert any("units" in message and "convertible" in message for message in required)
    assert not any("long_name" in message for message in required)
    assert any("long_name" in message for message in suggested)
    assert not any("units" in message for message in suggested)


def _climatology_dataset(*, climatology="climatology_bounds", bounds=None):
    dataset = xr.Dataset(
        data_vars={
            "climatology_bounds": (("time", "nv"), [[0.0, 30.0], [30.0, 60.0]]),
        },
        coords={"time": ("time", [15.0, 45.0])},
    )
    dataset["time"].attrs.update(
        {
            "axis": "T",
            "standard_name": "time",
            "long_name": "Time Intervals",
            "units": "days since 2000-01-01",
            "calendar": "proleptic_gregorian",
            "climatology": climatology,
        }
    )
    if bounds is not None:
        dataset["time"].attrs["bounds"] = bounds
    return dataset


def test_time4_uses_table_climatology_and_rejects_regular_bounds(tmp_path):
    entry = {**TIME_ENTRY, "must_have_bounds": "yes", "climatology": "yes"}
    dataset = _climatology_dataset(bounds="time_bnds")
    dataset["time_bnds"] = (("time", "nv"), [[0.0, 30.0], [30.0, 60.0]])
    checker = _checker(["time4"], {"time4": entry})

    with _open_netcdf(tmp_path, "time4_climatology", dataset) as nc:
        results = checker.check_time(nc)

    messages = _messages(results, BaseCheck.HIGH)
    assert any("must not have a 'bounds' attribute" in message for message in messages)
    assert any(
        "must not define regular bounds variable 'time_bnds'" in message
        for message in messages
    )


def test_climatology_may_name_time_bnds_itself(tmp_path):
    entry = {**TIME_ENTRY, "must_have_bounds": "yes", "climatology": "yes"}
    dataset = _climatology_dataset(climatology="time_bnds")
    dataset = dataset.drop_vars("climatology_bounds")
    dataset["time_bnds"] = (("time", "nv"), [[0.0, 30.0], [30.0, 60.0]])
    checker = _checker(["time2"], {"time2": entry})

    with _open_netcdf(tmp_path, "climatology_named_time_bnds", dataset) as nc:
        results = checker.check_time(nc)

    assert not any(
        "must not define regular bounds variable" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_time_bounds_follow_declared_nonstandard_name(tmp_path):
    entry = {
        **TIME_ENTRY,
        "must_have_bounds": "yes",
        "stored_direction": "increasing",
    }
    dataset = xr.Dataset(
        data_vars={
            "custom_time_bounds": (
                ("time", "nv"),
                [[0.0, 30.0], [30.0, 60.0]],
            )
        },
        coords={"time": ("time", [15.0, 45.0])},
    )
    dataset["time"].attrs.update(
        {
            "axis": "T",
            "standard_name": "time",
            "long_name": "Time Intervals",
            "units": "days since 2000-01-01",
            "calendar": "proleptic_gregorian",
            "bounds": "custom_time_bounds",
        }
    )
    checker = _checker(["time1"], {"time1": entry})

    with _open_netcdf(tmp_path, "time_custom_bounds", dataset) as nc:
        results = checker.check_time(nc)

    high = _messages(results, BaseCheck.HIGH)
    assert not any(
        "custom_time_bounds" in message and "not found" in message for message in high
    )
    assert any(
        "recommended name is 'time_bnds'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


def _hybrid_dataset(b_values, *, formula_terms="ap: ap b: b ps: ps"):
    dataset = xr.Dataset(
        data_vars={
            "ap": (("lev",), [0.0, 0.0, 0.0]),
            "b": (("lev",), b_values),
            "ps": (("time", "cell"), [[100000.0, 90000.0]]),
        },
        coords={
            "lev": ("lev", [1.0, 0.5, 0.0]),
            "time": ("time", [0]),
            "cell": ("cell", [0, 1]),
        },
    )
    dataset["lev"].attrs.update(
        {
            "axis": "Z",
            "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
            "positive": "down",
            "formula": HYBRID_ENTRY["formula"],
        }
    )
    if formula_terms is not None:
        dataset["lev"].attrs["formula_terms"] = formula_terms
    return dataset


def _vertical_checker():
    return _checker(
        ["alevel"],
        {"hybrid": HYBRID_ENTRY},
        vert_mapping={"alevel": "hybrid"},
    )


def test_vertical_direction_checks_a_formula_profile_at_one_horizontal_point(
    tmp_path,
):
    dataset = _hybrid_dataset([1.0, 0.5, 0.0])

    with _open_netcdf(tmp_path, "hybrid_correct", dataset) as nc:
        results = _vertical_checker().check_vertical_direction(nc)

    assert _messages(results, BaseCheck.HIGH) == []


def test_vertical_direction_reports_a_reversed_formula_profile(tmp_path):
    dataset = _hybrid_dataset([0.0, 0.5, 1.0])

    with _open_netcdf(tmp_path, "hybrid_reversed", dataset) as nc:
        results = _vertical_checker().check_vertical_direction(nc)

    messages = _messages(results, BaseCheck.HIGH)
    assert any(
        "Formula-derived profile" in message and "not strictly decreasing" in message
        for message in messages
    )


def test_vertical_bounds_follow_stored_direction(tmp_path):
    dataset = _hybrid_dataset([1.0, 0.5, 0.0])
    dataset["lev_bnds"] = (
        ("lev", "nv"),
        [[0.9, 1.1], [0.4, 0.6], [-0.1, 0.1]],
    )
    dataset["lev"].attrs["bounds"] = "lev_bnds"

    with _open_netcdf(tmp_path, "vertical_bounds_direction", dataset) as nc:
        results = _vertical_checker().check_vertical_direction(nc)

    assert any(
        "not ordered upper-to-lower" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_vertical_bounds_follow_declared_nonstandard_name(tmp_path):
    entry = {**HYBRID_ENTRY, "must_have_bounds": "yes"}
    dataset = _hybrid_dataset([1.0, 0.5, 0.0])
    dataset["custom_level_bounds"] = (
        ("lev", "nv"),
        [[1.1, 0.9], [0.6, 0.4], [0.1, -0.1]],
    )
    dataset["lev"].attrs["bounds"] = "custom_level_bounds"
    checker = _checker(
        ["alevel"],
        {"hybrid": entry},
        vert_mapping={"alevel": "hybrid"},
    )

    with _open_netcdf(tmp_path, "vertical_custom_bounds", dataset) as nc:
        results = checker.check_vertical(nc)

    high = _messages(results, BaseCheck.HIGH)
    assert not any(
        "custom_level_bounds" in message and "not found" in message for message in high
    )
    assert any(
        "recommended name is 'lev_bnds'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


@pytest.mark.parametrize("file_standard_name", [None, "height"])
def test_vertical_direction_uses_the_table_standard_name(tmp_path, file_standard_name):
    dataset = _hybrid_dataset([1.0, 0.5, 0.0])
    dataset["lev"].attrs["positive"] = "up"
    if file_standard_name is None:
        del dataset["lev"].attrs["standard_name"]
    else:
        dataset["lev"].attrs["standard_name"] = file_standard_name

    with _open_netcdf(tmp_path, "hybrid_file_standard_name", dataset) as nc:
        results = _vertical_checker().check_vertical_direction(nc)

    assert any(
        "standard_name='atmosphere_hybrid_sigma_pressure_coordinate'" in message
        and "implies positive='down'" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


@pytest.mark.parametrize(
    "formula_terms",
    [None, "ap: missing_ap b: missing_b ps: missing_ps"],
)
def test_vertical_direction_silently_skips_unavailable_formula_terms(
    tmp_path, formula_terms
):
    dataset = _hybrid_dataset(
        [1.0, 0.5, 0.0],
        formula_terms=formula_terms,
    ).drop_vars(["ap", "b", "ps"])

    with _open_netcdf(tmp_path, "hybrid_missing_terms", dataset) as nc:
        results = _vertical_checker().check_vertical_direction(nc)

    assert _messages(results, BaseCheck.HIGH) == []


def test_vertical_direction_silently_skips_divergent_formula_terms(tmp_path):
    dataset = _hybrid_dataset(
        [0.0, 0.5, 1.0],
        formula_terms="ap: b b: ap ps: ps",
    )

    with _open_netcdf(tmp_path, "hybrid_divergent_terms", dataset) as nc:
        results = _vertical_checker().check_vertical_direction(nc)

    # Using the divergent mapping would produce a reversed calculated profile.
    # AICC003b must leave the metadata finding to AICC003 and skip this portion.
    assert _messages(results, BaseCheck.HIGH) == []


def test_vertical_coordinate_reports_divergent_formula_terms(tmp_path):
    dataset = _hybrid_dataset(
        [1.0, 0.5, 0.0],
        formula_terms="ap: b b: ap ps: ps",
    )

    with _open_netcdf(tmp_path, "hybrid_divergent_terms_aicc003", dataset) as nc:
        results = _vertical_checker().check_vertical(nc)

    assert any(
        "formula_terms=" in message
        and "expected 'ap: ap b: b ps: ps' from the CMOR table" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


@pytest.mark.parametrize("formula", [None, "p = a*p0 + b*ps"])
def test_vertical_coordinate_requires_the_table_formula(tmp_path, formula):
    dataset = _hybrid_dataset([1.0, 0.5, 0.0])
    if formula is None:
        del dataset["lev"].attrs["formula"]
    else:
        dataset["lev"].attrs["formula"] = formula

    with _open_netcdf(tmp_path, "hybrid_formula", dataset) as nc:
        results = _vertical_checker().check_vertical(nc)

    assert any(
        "formula=" in message and "expected 'p = ap + b*ps'" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_vertical_mapping_uses_table_generic_level_marker_only(tmp_path):
    dataset = _hybrid_dataset([1.0, 0.5, 0.0])
    assert "generic_level_name" not in dataset["lev"].attrs
    wrong_entry = {**HYBRID_ENTRY, "generic_level_name": "olevel"}
    checker = _checker(
        ["alevel"],
        {"hybrid": wrong_entry},
        vert_mapping={"alevel": "hybrid"},
    )

    with _open_netcdf(tmp_path, "hybrid_generic_marker", dataset) as nc:
        results = checker.check_vertical(nc)

    assert any(
        "generic_level_name='olevel'; expected 'alevel'" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_generic_level_name_is_not_required_in_file_metadata(tmp_path):
    dataset = _hybrid_dataset([1.0, 0.5, 0.0])
    assert "generic_level_name" not in dataset["lev"].attrs

    with _open_netcdf(tmp_path, "hybrid_without_generic_attribute", dataset) as nc:
        results = _vertical_checker().check_vertical(nc)

    assert not any(
        "generic_level_name" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def _coordinate_dataset(name, values, standard_name, *, positive=None, dims=None):
    dimensions = dims or (name,)
    dataset = xr.Dataset({name: (dimensions, np.asarray(values, dtype="float64"))})
    dataset[name].attrs["standard_name"] = standard_name
    if positive is not None:
        dataset[name].attrs["positive"] = positive
    return dataset


def test_coordinate_direction_accepts_increasing_depth_positive_down(tmp_path):
    dataset = _coordinate_dataset("depth", [0.0, 10.0, 20.0], "depth", positive="down")
    checker = _checker(["sdepth"], {"sdepth": DEPTH_ENTRY})

    with _open_netcdf(tmp_path, "depth_correct", dataset) as nc:
        results = checker.check_coordinate_direction(nc)

    assert _messages(results, BaseCheck.HIGH) == []


def test_coordinate_direction_reports_nonmonotonic_depth(tmp_path):
    dataset = _coordinate_dataset("depth", [0.0, 20.0, 10.0], "depth", positive="down")
    checker = _checker(["sdepth"], {"sdepth": DEPTH_ENTRY})

    with _open_netcdf(tmp_path, "depth_nonmonotonic", dataset) as nc:
        results = checker.check_coordinate_direction(nc)

    assert any(
        "not strictly increasing" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_coordinate_reports_an_incorrect_storage_type(tmp_path):
    dataset = xr.Dataset(
        coords={"depth": ("depth", np.asarray([0.0, 10.0], dtype="float32"))}
    )
    dataset["depth"].attrs["standard_name"] = "depth"
    checker = _checker(["sdepth"], {"sdepth": DEPTH_ENTRY})

    with _open_netcdf(tmp_path, "depth_float32", dataset) as nc:
        results = checker.check_coord(nc)

    assert any(
        "'depth' has data type 'float32' (CMOR type 'real'); expected CMOR type "
        "'double'" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_coordinate_positive_is_checked_against_table_and_physical_meaning(
    tmp_path,
):
    dataset = xr.Dataset(coords={"plev": ("plev", [100000.0, 50000.0, 10000.0])})
    dataset["plev"].attrs.update({"standard_name": "air_pressure", "positive": "up"})
    checker = _checker(["plev19"], {"plev19": PLEV_ENTRY})

    with _open_netcdf(tmp_path, "plev_positive_up", dataset) as nc:
        table_results = checker.check_coord(nc)
        direction_results = checker.check_coordinate_direction(nc)

    assert any(
        "'plev' positive='up'; expected 'down' from the CMOR table" in message
        for message in _messages(table_results, BaseCheck.HIGH)
    )
    assert any(
        "positive='up'" in message and "implies positive='down'" in message
        for message in _messages(direction_results, BaseCheck.HIGH)
    )


def test_coordinate_direction_reports_incorrect_physical_positive(tmp_path):
    dataset = _coordinate_dataset("depth", [0.0, 10.0, 20.0], "depth", positive="up")
    checker = _checker(["sdepth"], {"sdepth": DEPTH_ENTRY})

    with _open_netcdf(tmp_path, "depth_positive", dataset) as nc:
        results = checker.check_coordinate_direction(nc)

    assert any(
        "positive='up'" in message and "implies positive='down'" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_coordinate_direction_requires_exactly_one_dimension(tmp_path):
    checker = _checker(["sdepth"], {"sdepth": DEPTH_ENTRY})

    # netCDF permits this deliberately invalid coordinate-like variable, while
    # xarray refuses to construct it because its name is also one of its two
    # dimensions. Create it directly so the checker behavior remains tested.
    path = tmp_path / "depth_two_dimensional.nc"
    with Dataset(path, "w") as nc:
        nc.createDimension("depth", 2)
        nc.createDimension("cell", 2)
        depth = nc.createVariable("depth", "f8", ("depth", "cell"))
        depth[:] = [[0.0, 1.0], [10.0, 11.0]]
        depth.standard_name = "depth"
        depth.positive = "down"
    with Dataset(path) as nc:
        results = checker.check_coordinate_direction(nc)

    assert any(
        "must be one-dimensional" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_stored_direction_applies_without_physical_positive_semantics(tmp_path):
    dataset = _coordinate_dataset(
        "rho",
        [1025.0, 1027.0, 1026.0],
        "sea_water_potential_density",
    )
    checker = _checker(["rho"], {"rho": RHO_ENTRY})

    with _open_netcdf(tmp_path, "density_nonmonotonic", dataset) as nc:
        results = checker.check_coordinate_direction(nc)

    messages = _messages(results, BaseCheck.HIGH)
    assert any("not strictly increasing" in message for message in messages)
    assert not any("positive=" in message for message in messages)


def test_numeric_coordinate_must_be_named_and_dimensioned_as_itself(tmp_path):
    dataset = xr.Dataset({"depth": (("level",), [0.0, 10.0])})
    dataset["depth"].attrs.update({"standard_name": "depth", "positive": "down"})
    checker = _checker(["sdepth"], {"sdepth": DEPTH_ENTRY})

    with _open_netcdf(tmp_path, "depth_wrong_dimension", dataset) as nc:
        results = checker.check_coord(nc)

    assert any(
        "must be 'depth(depth)'" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_ordinary_axis_and_scalar_units_are_required(tmp_path):
    entry = {
        **DEPTH_ENTRY,
        "value": "2",
        "axis": "Z",
        "units": "m",
        "valid_min": "1",
        "valid_max": "10",
    }
    dataset = xr.Dataset({"depth": xr.DataArray(2.0)})
    dataset["depth"].attrs.update(
        {"standard_name": "depth", "positive": "down", "axis": "X", "units": "km"}
    )
    checker = _checker(["height2m"], {"height2m": entry})

    with _open_netcdf(tmp_path, "scalar_axis_units", dataset) as nc:
        results = checker.check_coord(nc)

    messages = _messages(results, BaseCheck.HIGH)
    assert any("axis='X'; expected 'Z'" in message for message in messages)
    assert any(
        "units" in message and "required table units 'm'" in message
        for message in messages
    )


def test_long_name_is_required_without_standard_name_and_mismatch_is_medium(tmp_path):
    entry = {
        **RHO_ENTRY,
        "standard_name": "",
        "long_name": "Density class",
    }
    checker = _checker(["rho"], {"rho": entry})
    missing = xr.Dataset(coords={"rho": ("rho", [1.0, 2.0])})

    with _open_netcdf(tmp_path, "missing_long_name", missing) as nc:
        results = checker.check_coord(nc)
    assert any(
        "must have a long_name" in message
        for message in _messages(results, BaseCheck.HIGH)
    )

    mismatch = missing.copy(deep=True)
    mismatch["rho"].attrs["long_name"] = "Wrong label"
    with _open_netcdf(tmp_path, "mismatched_long_name", mismatch) as nc:
        results = checker.check_coord(nc)
    assert any(
        "expected 'Density class'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


def test_unprescribed_attributes_warn_and_optional_bounds_are_allowed(tmp_path):
    entry = {
        **RHO_ENTRY,
        "standard_name": "",
        "long_name": "Density class",
        "axis": "",
        "units": "",
        "positive": "",
        "must_have_bounds": "no",
    }
    dataset = xr.Dataset(
        data_vars={"rho_bnds": (("rho", "nv"), [[0.5, 1.5], [1.5, 2.5]])},
        coords={"rho": ("rho", [1.0, 2.0])},
    )
    dataset["rho"].attrs.update(
        {
            "standard_name": "model_specific_density",
            "long_name": "Density class",
            "axis": "Z",
            "units": "kg m-3",
            "positive": "down",
            "bounds": "rho_bnds",
        }
    )
    checker = _checker(["rho"], {"rho": entry})

    with _open_netcdf(tmp_path, "unprescribed_attributes", dataset) as nc:
        results = checker.check_coord(nc)

    warnings = _messages(results, BaseCheck.LOW)
    assert any("does not prescribe a standard_name" in message for message in warnings)
    assert any("does not prescribe an axis" in message for message in warnings)
    assert any("does not prescribe units" in message for message in warnings)
    assert any("does not prescribe positive" in message for message in warnings)
    assert not any(
        "bounds" in message.lower() for message in _messages(results, BaseCheck.HIGH)
    )
    assert any(
        "recommended name is 'bnds'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


@pytest.mark.parametrize(
    ("limits", "values", "fragment"),
    [
        ({"valid_min": "0", "valid_max": ""}, [-1.0, 2.0], "below valid_min"),
        ({"valid_min": "", "valid_max": "10"}, [1.0, 11.0], "above valid_max"),
    ],
)
def test_valid_limits_are_checked_independently(tmp_path, limits, values, fragment):
    entry = {**RHO_ENTRY, **limits}
    dataset = xr.Dataset(coords={"rho": ("rho", values)})
    dataset["rho"].attrs["standard_name"] = "sea_water_potential_density"
    checker = _checker(["rho"], {"rho": entry})

    with _open_netcdf(tmp_path, "valid_limit", dataset) as nc:
        results = checker.check_coord(nc)
    messages = _messages(results, BaseCheck.HIGH)
    assert any(fragment in message for message in messages)
    assert not any("numerical slack" in message for message in messages)


def test_bounds_structure_type_direction_and_naming(tmp_path):
    entry = {**DEPTH_ENTRY, "must_have_bounds": "yes"}
    dataset = xr.Dataset(
        data_vars={
            "custom_bounds": (
                ("depth", "nv"),
                np.asarray([[1.0, 0.0], [11.0, 9.0]], dtype="float32"),
            )
        },
        coords={"depth": ("depth", [0.5, 10.0])},
    )
    dataset["depth"].attrs.update(
        {
            "standard_name": "depth",
            "positive": "down",
            "bounds": "custom_bounds",
        }
    )
    checker = _checker(["sdepth"], {"sdepth": entry})

    with _open_netcdf(tmp_path, "ordinary_bounds", dataset) as nc:
        results = checker.check_coord(nc)

    high = _messages(results, BaseCheck.HIGH)
    assert not any(
        "custom_bounds" in message and "not found" in message for message in high
    )
    assert any("expected CMOR type 'double'" in message for message in high)
    assert any("not ordered lower-to-upper" in message for message in high)
    assert any(
        "recommended name is 'depth_bnds'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


def test_optional_bounds_attribute_is_followed(tmp_path):
    entry = {**DEPTH_ENTRY, "must_have_bounds": "no"}
    dataset = xr.Dataset(coords={"depth": ("depth", [0.5, 10.0])})
    dataset["depth"].attrs.update(
        {
            "standard_name": "depth",
            "positive": "down",
            "bounds": "provided_but_missing",
        }
    )
    checker = _checker(["sdepth"], {"sdepth": entry})

    with _open_netcdf(tmp_path, "optional_dangling_bounds", dataset) as nc:
        results = checker.check_coord(nc)

    assert any(
        "Bounds variable 'provided_but_missing'" in message and "not found" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_scalar_bounds_follow_declared_nonstandard_name(tmp_path):
    entry = {
        **DEPTH_ENTRY,
        "value": "2",
        "bounds_values": "1 3",
        "must_have_bounds": "yes",
    }
    dataset = xr.Dataset(
        data_vars={
            "depth": xr.DataArray(2.0),
            "custom_scalar_bounds": (("nv",), [1.0, 3.0]),
        }
    )
    dataset["depth"].attrs.update(
        {
            "standard_name": "depth",
            "positive": "down",
            "bounds": "custom_scalar_bounds",
        }
    )
    checker = _checker(["depth2"], {"depth2": entry})

    with _open_netcdf(tmp_path, "scalar_custom_bounds", dataset) as nc:
        results = checker.check_coord(nc)

    high = _messages(results, BaseCheck.HIGH)
    assert not any(
        "custom_scalar_bounds" in message and "not found" in message for message in high
    )
    assert any(
        "recommended name is 'depth_bnds'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


def test_bounds_must_use_coordinate_and_size_two_dimensions(tmp_path):
    entry = {**DEPTH_ENTRY, "must_have_bounds": "yes"}
    dataset = xr.Dataset(
        data_vars={"depth_bnds": (("nv", "depth"), [[0.0, 9.0], [1.0, 11.0]])},
        coords={"depth": ("depth", [0.5, 10.0])},
    )
    dataset["depth"].attrs.update(
        {
            "standard_name": "depth",
            "positive": "down",
            "bounds": "depth_bnds",
        }
    )
    checker = _checker(["sdepth"], {"sdepth": entry})

    with _open_netcdf(tmp_path, "bounds_dimensions", dataset) as nc:
        results = checker.check_coord(nc)

    assert any(
        "expected dimensions ('depth', <size-2>)" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_scalar_bounds_do_not_use_cmor_tolerance(tmp_path):
    entry = {
        **DEPTH_ENTRY,
        "value": "2",
        "bounds_values": "1 3",
        "must_have_bounds": "yes",
        "tolerance": "1",
    }
    dataset = xr.Dataset(
        data_vars={
            "depth": xr.DataArray(2.0),
            "depth_bnds": (("nv",), [1.0005, 3.0]),
        }
    )
    dataset["depth"].attrs.update(
        {
            "standard_name": "depth",
            "positive": "down",
            "bounds": "depth_bnds",
        }
    )
    checker = _checker(["depth2"], {"depth2": entry})

    with _open_netcdf(tmp_path, "scalar_bounds_exact", dataset) as nc:
        results = checker.check_coord(nc)

    assert any(
        "expected exact scalar bounds [1.0, 3.0]" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def _character_sector_dataset(coordinates=""):
    labels = np.asarray([list(b"alpha"), list(b"beta ")], dtype="u1").view("S1")
    dataset = xr.Dataset(
        data_vars={
            "tas": (("basin",), [1.0, 2.0]),
            "sector": (("basin", "strlen"), labels),
        }
    )
    if coordinates:
        dataset["tas"].attrs["coordinates"] = coordinates
    return dataset


def test_text_auxiliary_coordinate_must_be_listed_but_extra_labels_are_allowed(
    tmp_path,
):
    entry = {
        "out_name": "basin",
        "standard_name": "",
        "long_name": "Basin",
        "type": "character",
        "requested": ["beta", "alpha"],
    }
    checker = _checker(["basin"], {"basin": entry}, var_entry={"out_name": "tas"})
    missing = _character_sector_dataset()
    missing["sector"].attrs["long_name"] = "Basin"
    with _open_netcdf(tmp_path, "sector_not_listed", missing) as nc:
        results = checker.check_coord(nc)
    assert any(
        "coordinates' attribute must include text auxiliary coordinate 'sector'"
        in message
        for message in _messages(results, BaseCheck.HIGH)
    )

    listed = _character_sector_dataset("sector")
    listed["sector"].attrs["long_name"] = "Basin"
    with _open_netcdf(tmp_path, "sector_listed", listed) as nc:
        results = checker.check_coord(nc)
    assert not any(
        "missing requested value" in message for message in _messages(results)
    )


def test_site_requires_latitude_and_longitude_auxiliaries(tmp_path):
    entry = {
        "out_name": "site",
        "standard_name": "",
        "long_name": "site index",
        "type": "integer",
        "requested": ["1", "2"],
    }
    dataset = xr.Dataset(
        data_vars={"tas": (("site",), [280.0, 281.0])},
        coords={
            "site": ("site", np.asarray([1, 2], dtype="int32")),
            "station_lat": ("site", [50.0, 51.0]),
            "station_lon": ("site", [10.0, 11.0]),
        },
    )
    dataset["site"].attrs["long_name"] = "site index"
    dataset["station_lat"].attrs.update(
        {"standard_name": "latitude", "units": "degrees_north"}
    )
    dataset["station_lon"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east"}
    )
    dataset["tas"].attrs["coordinates"] = "station_lat station_lon"
    checker = _checker(["site"], {"site": entry}, var_entry={"out_name": "tas"})

    with _open_netcdf(tmp_path, "site_coordinates", dataset) as nc:
        results = checker.check_coord(nc)
        attribute_results = checker.check_coordinates_attribute(nc)
    assert _messages(results, BaseCheck.HIGH) == []
    assert _messages(attribute_results, BaseCheck.LOW) == []

    dataset["tas"].attrs["coordinates"] = "station_lat"
    with _open_netcdf(tmp_path, "site_missing_longitude", dataset) as nc:
        results = checker.check_coord(nc)
    assert any(
        "standard_name='longitude'" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_cmor_tolerance_algorithms_follow_coordinate_recipe():
    assert _cmor_tol_val(0, [100.0, 110.0], [], 1.0) == pytest.approx(0.1)
    assert _cmor_tol_val(1, [100.0, 110.0], [], 1.0) == pytest.approx(0.11)
    assert _cmor_bound_tol_vals(0, [(100.0, 200.0)], 1.0) == pytest.approx((0.1, 0.2))


def _horizontal_dataset(coordinates="latitude longitude"):
    dataset = xr.Dataset(
        data_vars={
            "ta": (
                ("time", "latitude", "longitude"),
                np.zeros((1, 2, 3), dtype="float32"),
            )
        },
        coords={
            "time": ("time", [0.0]),
            "latitude": ("latitude", [-45.0, 45.0]),
            "longitude": ("longitude", [0.0, 120.0, 240.0]),
        },
    )
    dataset["latitude"].attrs.update(
        {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
    )
    dataset["longitude"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    )
    dataset["ta"].attrs["coordinates"] = coordinates
    return dataset


def test_rectilinear_bounds_name_uses_cmor_coordinate_name(tmp_path):
    latitude_entry = {
        "out_name": "lat",
        "standard_name": "latitude",
        "axis": "Y",
        "units": "degrees_north",
        "type": "double",
        "must_have_bounds": "yes",
    }
    dataset = _horizontal_dataset()
    dataset["latitude_bnds"] = (
        ("latitude", "nv"),
        [[-90.0, 0.0], [0.0, 90.0]],
    )
    dataset["latitude"].attrs["bounds"] = "latitude_bnds"
    checker = _checker(
        ["latitude"],
        {"latitude": latitude_entry},
        var_entry={"out_name": "ta"},
    )
    checker._grid_type = "rectilinear"
    checker._grid_type_known = True

    with _open_netcdf(tmp_path, "rectilinear_bounds_name", dataset) as nc:
        results = checker.check_grid(nc)

    assert not any(
        "latitude_bnds" in message and "not found" in message
        for message in _messages(results, BaseCheck.HIGH)
    )
    assert any(
        "'latitude' bounds='latitude_bnds'; recommended name is 'lat_bnds'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


def test_existing_longitude_bounds_is_not_reported_as_missing(tmp_path):
    longitude_entry = {
        "out_name": "lon",
        "standard_name": "longitude",
        "axis": "X",
        "units": "degrees_east",
        "type": "double",
        "must_have_bounds": "yes",
    }
    dataset = _horizontal_dataset()
    dataset["longitude_bnds"] = (
        ("longitude", "nv"),
        [[-60.0, 60.0], [60.0, 180.0], [180.0, 300.0]],
    )
    dataset["longitude"].attrs["bounds"] = "longitude_bnds"
    checker = _checker(
        ["longitude"],
        {"longitude": longitude_entry},
        var_entry={"out_name": "ta"},
    )
    checker._grid_type = "rectilinear"
    checker._grid_type_known = True

    with _open_netcdf(tmp_path, "longitude_declared_bounds", dataset) as nc:
        results = checker.check_grid(nc)

    high = _messages(results, BaseCheck.HIGH)
    assert not any(
        "longitude_bnds" in message and "not found" in message for message in high
    )
    assert any(
        "'longitude' bounds='longitude_bnds'; recommended name is 'lon_bnds'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


def _unknown_grid_checker():
    checker = _checker(
        ["longitude", "latitude", "time"],
        {},
        var_entry={"out_name": "ta"},
    )
    checker._grid_type = None
    checker._grid_type_known = False
    return checker


def test_dimensions_silently_skips_an_unresolved_horizontal_grid(tmp_path):
    with _open_netcdf(tmp_path, "unknown_grid_dimensions", _horizontal_dataset()) as nc:
        results = _unknown_grid_checker().check_dimensions(nc)

    assert _messages(results, BaseCheck.HIGH) == []


def test_coordinates_attribute_allows_horizontal_coordinates_for_unknown_grid(
    tmp_path,
):
    dataset = _horizontal_dataset()

    with _open_netcdf(tmp_path, "unknown_grid_coordinates", dataset) as nc:
        results = _unknown_grid_checker().check_coordinates_attribute(nc)

    assert _messages(results, BaseCheck.LOW) == []


def test_coordinates_attribute_still_reports_unrelated_entries_for_unknown_grid(
    tmp_path,
):
    dataset = _horizontal_dataset("latitude longitude rogue")

    with _open_netcdf(tmp_path, "unknown_grid_extra_coordinate", dataset) as nc:
        results = _unknown_grid_checker().check_coordinates_attribute(nc)

    assert any(
        "not requested auxiliary or scalar coordinates: ['rogue']" in message
        for message in _messages(results, BaseCheck.LOW)
    )


def _curvilinear_dataset(axis_scheme):
    if axis_scheme == "rotated":
        dims = ("rlat", "rlon")
        axis_coords = {
            "rlat": ("rlat", [-1.0, 1.0], CURVILINEAR_AXIS_ENTRIES["grid_latitude"]),
            "rlon": (
                "rlon",
                [-2.0, 0.0, 2.0],
                CURVILINEAR_AXIS_ENTRIES["grid_longitude"],
            ),
        }
    elif axis_scheme == "metric":
        dims = ("y", "x")
        axis_coords = {
            "y": ("y", [0.0, 1000.0], CURVILINEAR_AXIS_ENTRIES["y"]),
            "x": ("x", [0.0, 1000.0, 2000.0], CURVILINEAR_AXIS_ENTRIES["x"]),
        }
    elif axis_scheme == "angular":
        dims = ("y", "x")
        axis_coords = {
            "y": ("y", [-1.0, 1.0], CURVILINEAR_AXIS_ENTRIES["y_deg"]),
            "x": ("x", [-2.0, 0.0, 2.0], CURVILINEAR_AXIS_ENTRIES["x_deg"]),
        }
    elif axis_scheme == "index":
        dims = ("j", "i")
        axis_coords = {
            "j": (
                "j",
                np.asarray([0, 1], dtype="int32"),
                CURVILINEAR_AXIS_ENTRIES["j_index"],
            ),
            "i": (
                "i",
                np.asarray([0, 1, 2], dtype="int32"),
                CURVILINEAR_AXIS_ENTRIES["i_index"],
            ),
        }
    else:
        dims = ("y", "x")
        axis_coords = {}

    latitude = np.asarray([[40.0, 40.5, 41.0], [41.0, 41.5, 42.0]])
    longitude = np.asarray([[10.0, 11.0, 12.0], [10.5, 11.5, 12.5]])
    vertex_offsets = np.asarray([-0.2, -0.1, 0.1, 0.2])
    dataset = xr.Dataset(
        data_vars={
            "tas": (dims, np.zeros((2, 3), dtype="float32")),
            "vertices_latitude": (
                (*dims, "vertices"),
                latitude[..., None] + vertex_offsets,
            ),
            "vertices_longitude": (
                (*dims, "vertices"),
                longitude[..., None] + vertex_offsets,
            ),
        },
        coords={
            **axis_coords,
            "latitude": (dims, latitude),
            "longitude": (dims, longitude),
        },
    )
    for name in ("latitude", "longitude", "vertices_latitude", "vertices_longitude"):
        dataset[name].attrs.update(CURVILINEAR_GRID_ENTRIES[name])
    dataset["latitude"].attrs["bounds"] = "vertices_latitude"
    dataset["longitude"].attrs["bounds"] = "vertices_longitude"
    dataset["tas"].attrs["coordinates"] = "latitude longitude"
    if axis_scheme == "rotated":
        dataset["rotated_pole"] = xr.DataArray(0)
        dataset["rotated_pole"].attrs["grid_mapping_name"] = (
            "rotated_latitude_longitude"
        )
        dataset["tas"].attrs["grid_mapping"] = "rotated_pole"
    elif axis_scheme in {"metric", "angular"}:
        dataset["projection"] = xr.DataArray(0)
        dataset["projection"].attrs["grid_mapping_name"] = "lambert_conformal_conic"
        dataset["tas"].attrs["grid_mapping"] = "projection"
    return dataset


def _curvilinear_checker():
    checker = _checker(
        ["longitude", "latitude"],
        {},
        var_entry={"out_name": "tas"},
    )
    checker.CTgrids = {
        "variable_entry": CURVILINEAR_GRID_ENTRIES,
        "axis_entry": CURVILINEAR_AXIS_ENTRIES,
    }
    checker._grid_type = "curvilinear"
    checker._grid_type_known = True
    return checker


@pytest.mark.parametrize(
    "axis_scheme", ["implicit", "rotated", "metric", "angular", "index"]
)
def test_curvilinear_grid_accepts_supported_axis_representations(tmp_path, axis_scheme):
    dataset = _curvilinear_dataset(axis_scheme)
    checker = _curvilinear_checker()

    with _open_netcdf(tmp_path, f"curvilinear_{axis_scheme}", dataset) as nc:
        grid_results = checker.check_grid(nc)
        dimension_results = checker.check_dimensions(nc)

    assert _messages(grid_results, BaseCheck.HIGH) == []
    assert _messages(dimension_results, BaseCheck.HIGH) == []


def test_curvilinear_grid_rejects_invalid_vertex_dimensions(tmp_path):
    dataset = _curvilinear_dataset("implicit")
    dataset["vertices_latitude"] = (
        ("vertices", "y", "x"),
        np.moveaxis(dataset["vertices_latitude"].values, -1, 0),
    )
    dataset["vertices_latitude"].attrs.update(
        CURVILINEAR_GRID_ENTRIES["vertices_latitude"]
    )

    with _open_netcdf(tmp_path, "curvilinear_bad_vertices", dataset) as nc:
        results = _curvilinear_checker().check_grid(nc)

    assert any(
        "must use the coordinate's two dimensions followed by a vertex dimension"
        in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_curvilinear_vertices_follow_declared_bounds_name(tmp_path):
    dataset = _curvilinear_dataset("implicit").rename(
        {"vertices_longitude": "custom_longitude_vertices"}
    )
    dataset["longitude"].attrs["bounds"] = "custom_longitude_vertices"

    with _open_netcdf(tmp_path, "curvilinear_custom_vertices", dataset) as nc:
        results = _curvilinear_checker().check_grid(nc)

    high = _messages(results, BaseCheck.HIGH)
    assert not any(
        "custom_longitude_vertices" in message and "not found" in message
        for message in high
    )
    assert any(
        "bounds='custom_longitude_vertices'; recommended name is "
        "'vertices_longitude'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


def test_curvilinear_grid_requires_both_horizontal_coordinates(tmp_path):
    dataset = _curvilinear_dataset("implicit").drop_vars(
        ["longitude", "vertices_longitude"]
    )

    with _open_netcdf(tmp_path, "curvilinear_missing_longitude", dataset) as nc:
        results = _curvilinear_checker().check_grid(nc)

    assert any(
        "Curvilinear longitude variable 'longitude' not found" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_curvilinear_latitude_and_longitude_must_share_dimensions(tmp_path):
    dataset = _curvilinear_dataset("implicit").drop_vars(
        ["longitude", "vertices_longitude"]
    )
    longitude = np.asarray([[10.0, 11.0, 12.0], [10.5, 11.5, 12.5]])
    offsets = np.asarray([-0.2, -0.1, 0.1, 0.2])
    dataset = dataset.assign_coords(longitude=(("y", "x2"), longitude))
    dataset["longitude"].attrs.update(CURVILINEAR_GRID_ENTRIES["longitude"])
    dataset["vertices_longitude"] = (
        ("y", "x2", "vertices"),
        longitude[..., None] + offsets,
    )
    dataset["vertices_longitude"].attrs.update(
        CURVILINEAR_GRID_ENTRIES["vertices_longitude"]
    )

    with _open_netcdf(tmp_path, "curvilinear_mismatched_dims", dataset) as nc:
        results = _curvilinear_checker().check_grid(nc)

    assert any(
        "latitude and longitude must share the same two dimensions" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_rotated_mapping_requires_rotated_coordinate_axes(tmp_path):
    dataset = _curvilinear_dataset("metric")
    dataset["projection"].attrs["grid_mapping_name"] = "rotated_latitude_longitude"

    with _open_netcdf(tmp_path, "curvilinear_wrong_mapping_axes", dataset) as nc:
        results = _curvilinear_checker().check_grid(nc)

    assert any(
        "requires the rlat/rlon axis representation" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def _unstructured_latitude_dataset(vertex_units=None):
    dataset = xr.Dataset(
        data_vars={
            "tas": (("cell",), [280.0, 281.0]),
            "vertices_latitude": (
                ("cell", "vertex"),
                [[-50.0, -45.0, -40.0], [40.0, 45.0, 50.0]],
            ),
        },
        coords={"latitude": ("cell", [-45.0, 45.0])},
    )
    dataset["latitude"].attrs.update(
        {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
        }
    )
    dataset["latitude"].attrs["bounds"] = "vertices_latitude"
    dataset["tas"].attrs["coordinates"] = "latitude"
    if vertex_units is not None:
        dataset["vertices_latitude"].attrs["units"] = vertex_units
    return dataset


def _unstructured_grid_checker():
    checker = _checker(
        ["latitude"],
        {},
        var_entry={"out_name": "tas"},
    )
    checker.CTgrids = {"variable_entry": UNSTRUCTURED_GRID_ENTRIES}
    checker._grid_type = "unstructured"
    checker._grid_type_known = True
    return checker


@pytest.mark.parametrize("vertex_units", [None, "degrees_north"])
def test_vertex_bounds_attributes_are_optional(tmp_path, vertex_units):
    dataset = _unstructured_latitude_dataset(vertex_units)

    with _open_netcdf(tmp_path, "optional_vertex_attributes", dataset) as nc:
        results = _unstructured_grid_checker().check_grid(nc)

    assert not any(
        "vertices_latitude" in message for message in _messages(results, BaseCheck.HIGH)
    )
    assert any(
        "trailing dimension 'vertex'; recommended name is 'vertices'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


def test_vertex_bounds_reports_an_incorrect_attribute(tmp_path):
    dataset = _unstructured_latitude_dataset("radians")

    with _open_netcdf(tmp_path, "incorrect_vertex_attributes", dataset) as nc:
        results = _unstructured_grid_checker().check_grid(nc)

    assert any(
        "'vertices_latitude' units:" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_unstructured_vertices_follow_declared_bounds_name(tmp_path):
    dataset = _unstructured_latitude_dataset().rename(
        {"vertices_latitude": "custom_latitude_vertices"}
    )
    dataset["latitude"].attrs["bounds"] = "custom_latitude_vertices"

    with _open_netcdf(tmp_path, "unstructured_custom_vertices", dataset) as nc:
        results = _unstructured_grid_checker().check_grid(nc)

    high = _messages(results, BaseCheck.HIGH)
    assert not any(
        "custom_latitude_vertices" in message and "not found" in message
        for message in high
    )
    assert any(
        "bounds='custom_latitude_vertices'; recommended name is "
        "'vertices_latitude'" in message
        for message in _messages(results, BaseCheck.MEDIUM)
    )


def test_numeric_coordinate_without_stored_direction_must_be_strictly_monotonic(
    tmp_path,
):
    entry = {**RHO_ENTRY, "stored_direction": ""}
    dataset = xr.Dataset(coords={"rho": ("rho", [1.0, 3.0, 2.0])})
    dataset["rho"].attrs["standard_name"] = entry["standard_name"]

    with _open_netcdf(tmp_path, "rho_not_monotonic", dataset) as nc:
        results = _checker(["rho"], {"rho": entry}).check_coordinate_direction(nc)

    assert any(
        "not strictly monotonic (increasing or decreasing)" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_bounds_infer_parent_direction_and_check_inside_and_between_pairs(tmp_path):
    entry = {**RHO_ENTRY, "stored_direction": "", "must_have_bounds": "yes"}
    dataset = xr.Dataset(
        data_vars={
            "rho_bnds": (
                ("rho", "bnds"),
                [[0.5, 1.5], [2.5, 1.8], [1.0, 3.5]],
            )
        },
        coords={"rho": ("rho", [1.0, 2.0, 3.0])},
    )
    dataset["rho"].attrs.update(
        {"standard_name": entry["standard_name"], "bounds": "rho_bnds"}
    )

    with _open_netcdf(tmp_path, "inferred_bounds_direction", dataset) as nc:
        results = _checker(["rho"], {"rho": entry}).check_coord(nc)

    messages = _messages(results, BaseCheck.HIGH)
    assert any("not ordered lower-to-upper" in message for message in messages)
    assert any("between successive cells" in message for message in messages)
    assert all(
        "parent coordinate 'rho'" in message
        for message in messages
        if "ordered" in message or "successive" in message
    )


def test_dimension_order_uses_generic_vertical_out_name(tmp_path):
    generic = {"out_name": "model_lev", "type": "double"}
    configured = {
        **HYBRID_ENTRY,
        "out_name": "model_lev",
        "generic_level_name": "alevel",
    }
    dataset = xr.Dataset(
        data_vars={"ta": (("model_lev",), [250.0, 240.0])},
        coords={"model_lev": ("model_lev", [1.0, 0.0])},
    )
    checker = _checker(
        ["alevel"],
        {"alevel": generic, "configured": configured},
        var_entry={"out_name": "ta"},
        vert_mapping={"alevel": "configured"},
    )

    with _open_netcdf(tmp_path, "generic_vertical_out_name", dataset) as nc:
        results = checker.check_dimensions(nc)

    assert _messages(results, BaseCheck.HIGH) == []


def test_formula_term_dimensions_are_validated(tmp_path):
    dataset = _hybrid_dataset([1.0, 0.5, 0.0]).drop_vars("ap")
    dataset["ap"] = (("wrong_level",), [0.0, 0.0, 0.0])
    checker = _vertical_checker()
    checker.CTformulas = {
        "formula_entry": {
            "ap": {
                "out_name": "ap",
                "dimensions": "alevel",
                "type": "double",
                "units": "",
            }
        }
    }

    with _open_netcdf(tmp_path, "formula_term_dimensions", dataset) as nc:
        results = checker.check_vertical(nc)

    assert any(
        "Formula term 'ap' has dimensions ['wrong_level']; expected ['lev']" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_bounds_formula_terms_and_factor_dimensions_are_validated(tmp_path):
    entry = {
        **HYBRID_ENTRY,
        "must_have_bounds": "yes",
        "z_bounds_factors": "ap: ap_bnds b: b_bnds ps: ps",
    }
    dataset = _hybrid_dataset([1.0, 0.5, 0.0])
    dataset["lev_bnds"] = (
        ("lev", "bnds"),
        [[1.1, 0.9], [0.6, 0.4], [0.1, -0.1]],
    )
    dataset["lev"].attrs["bounds"] = "lev_bnds"
    dataset["lev_bnds"].attrs["formula_terms"] = "ap: wrong b: b_bnds ps: ps"
    dataset["ap_bnds"] = (("wrong_level", "bnds"), np.zeros((3, 2)))
    dataset["b_bnds"] = (("lev", "bnds"), np.zeros((3, 2)))
    checker = _checker(
        ["alevel"],
        {"hybrid": entry},
        vert_mapping={"alevel": "hybrid"},
    )
    checker.CTformulas = {
        "formula_entry": {
            "ap_bnds": {
                "out_name": "ap_bnds",
                "dimensions": "alevel",
                "type": "double",
                "units": "Pa",
            },
            "b_bnds": {
                "out_name": "b_bnds",
                "dimensions": "alevel",
                "type": "double",
                "units": "",
            },
        }
    }

    with _open_netcdf(tmp_path, "bounds_formula_terms", dataset) as nc:
        results = checker.check_vertical(nc)

    messages = _messages(results, BaseCheck.HIGH)
    assert any("'lev_bnds' formula_terms=" in message for message in messages)
    assert any(
        "Bounds formula term 'ap_bnds'" in message
        and "size-2 bounds dimension" in message
        for message in messages
    ), messages


def test_model_level_bounds_formula_must_match_parent(tmp_path):
    entry = {**HYBRID_ENTRY, "must_have_bounds": "yes"}
    dataset = _hybrid_dataset([1.0, 0.5, 0.0])
    dataset["lev_bnds"] = (
        ("lev", "bnds"),
        [[1.1, 0.9], [0.6, 0.4], [0.1, -0.1]],
    )
    dataset["lev"].attrs["bounds"] = "lev_bnds"
    dataset["lev_bnds"].attrs["formula"] = "p = a*p0 + b*ps"
    checker = _checker(
        ["alevel"],
        {"hybrid": entry},
        vert_mapping={"alevel": "hybrid"},
    )

    with _open_netcdf(tmp_path, "bounds_formula", dataset) as nc:
        results = checker.check_vertical(nc)

    assert any(
        "expected the same formula as 'lev': 'p = ap + b*ps'" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_unstructured_vertices_follow_auxiliary_dimension(tmp_path):
    dataset = _unstructured_latitude_dataset()
    dataset["vertices_latitude"] = (
        ("other_cell", "vertices"),
        [[-50.0, -45.0, -40.0], [40.0, 45.0, 50.0]],
    )

    with _open_netcdf(tmp_path, "unstructured_vertex_dimensions", dataset) as nc:
        results = _unstructured_grid_checker().check_grid(nc)

    assert any(
        "must use the auxiliary coordinate dimension" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_explicit_grid_axes_must_be_strictly_monotonic(tmp_path):
    dataset = _curvilinear_dataset("rotated")
    dataset = dataset.assign_coords(
        rlat=(
            "rlat",
            [0.0, 0.0],
            CURVILINEAR_AXIS_ENTRIES["grid_latitude"],
        )
    )

    with _open_netcdf(tmp_path, "nonmonotonic_grid_axis", dataset) as nc:
        results = _curvilinear_checker().check_grid(nc)

    assert any(
        "Grid axis 'rlat' is not strictly monotonic" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_explicit_unstructured_axis_is_validated(tmp_path):
    dataset = _unstructured_latitude_dataset().assign_coords(
        cell=(
            "cell",
            [0, 0],
            {"long_name": "cell index", "units": "1"},
        )
    )
    checker = _unstructured_grid_checker()
    checker.CTgrids["axis_entry"] = {
        "cell_index": {
            "out_name": "cell",
            "long_name": "cell index",
            "units": "1",
            "type": "integer",
        }
    }

    with _open_netcdf(tmp_path, "nonmonotonic_cell_axis", dataset) as nc:
        results = checker.check_grid(nc)

    assert any(
        "Grid axis 'cell' is not strictly monotonic" in message
        for message in _messages(results, BaseCheck.HIGH)
    )


def test_climatology_and_bounds_dimension_names_are_recommended(tmp_path):
    entry = {**TIME_ENTRY, "must_have_bounds": "yes", "climatology": "yes"}
    dataset = _climatology_dataset()
    checker = _checker(["time4"], {"time4": entry})

    with _open_netcdf(tmp_path, "climatology_names", dataset) as nc:
        results = checker.check_time(nc)

    messages = _messages(results, BaseCheck.MEDIUM)
    assert any(
        "recommended name is 'climatology_bnds'" in message for message in messages
    )
    assert any("recommended name is 'bnds'" in message for message in messages)
