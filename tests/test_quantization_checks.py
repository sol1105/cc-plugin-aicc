"""Focused tests for CF-1.12 quantization metadata."""

from contextlib import contextmanager

import numpy as np
import xarray as xr
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from cc_plugin_aicc.aicc import AICC


@contextmanager
def _open_netcdf(tmp_path, name, dataset):
    path = tmp_path / f"{name}.nc"
    dataset.to_netcdf(path, engine="netcdf4")
    with Dataset(path) as nc:
        yield nc


def _messages(results):
    return [
        message
        for result in results
        if result.weight == BaseCheck.HIGH
        for message in result.msgs
    ]


def _quantized_dataset():
    dataset = xr.Dataset(
        data_vars={
            "tas": ("x", np.asarray([280.0, 281.0], dtype="float32")),
            "quantization_info": xr.DataArray(np.int8(0)),
        },
        coords={"x": ("x", [0, 1])},
    )
    dataset["quantization_info"].attrs.update(
        {"algorithm": "bitround", "implementation": "netCDF-C version 4.9.2"}
    )
    dataset["tas"].attrs.update(
        {"quantization": "quantization_info", "quantization_nsb": 10}
    )
    return dataset


def test_valid_quantization_metadata_passes(tmp_path):
    with _open_netcdf(tmp_path, "valid_quantization", _quantized_dataset()) as nc:
        results = AICC().check_quantization(nc)

    assert _messages(results) == []


def test_parameter_without_quantization_reference_fails(tmp_path):
    dataset = _quantized_dataset()
    del dataset["tas"].attrs["quantization"]

    with _open_netcdf(tmp_path, "parameter_only", dataset) as nc:
        results = AICC().check_quantization(nc)

    assert any(
        "missing the CF 'quantization' attribute" in msg for msg in _messages(results)
    )


def test_coordinate_variable_must_not_be_quantized(tmp_path):
    dataset = _quantized_dataset()
    dataset = dataset.assign_coords(x=("x", np.asarray([0.0, 1.0], dtype="float32")))
    dataset["x"].attrs.update(
        {"quantization": "quantization_info", "quantization_nsb": 10}
    )

    with _open_netcdf(tmp_path, "quantized_coordinate", dataset) as nc:
        results = AICC().check_quantization(nc)

    assert any(
        "must not be quantized" in msg and "coordinate variable" in msg
        for msg in _messages(results)
    )
