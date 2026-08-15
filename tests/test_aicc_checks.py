"""Focused tests for AICC coordinate attribute and direction checks."""

from contextlib import contextmanager

import numpy as np
import pytest
import xarray as xr
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from cc_plugin_aicc.aicc import AICC
from cc_plugin_aicc.utils import _compare_units


TIME_ENTRY = {
    "out_name": "time",
    "standard_name": "time",
    "long_name": "Time Intervals",
    "units": "days since ?",
    "must_have_bounds": "no",
}

HYBRID_ENTRY = {
    "out_name": "lev",
    "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
    "stored_direction": "decreasing",
    "positive": "down",
    "formula": "p = ap + b*ps",
    "z_factors": "ap: ap b: b ps: ps",
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


@pytest.mark.parametrize("file_standard_name", [None, "height"])
def test_vertical_direction_uses_the_table_standard_name(
    tmp_path, file_standard_name
):
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
    dataset = _coordinate_dataset(
        "depth",
        [[0.0, 1.0], [10.0, 11.0]],
        "depth",
        positive="down",
        dims=("depth", "cell"),
    )
    checker = _checker(["sdepth"], {"sdepth": DEPTH_ENTRY})

    with _open_netcdf(tmp_path, "depth_two_dimensional", dataset) as nc:
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
