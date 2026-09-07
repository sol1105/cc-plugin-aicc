"""CF quantization metadata checks for AICC."""

import re

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx
from compliance_checker.cf import util as cfutil

from cc_plugin_aicc.utils import _format_attribute, _ncattr, _parse_formula_terms


class QuantizationChecks:
    """CF-1.12 lossy quantization metadata validation."""

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
        implementation_pattern = re.compile(r"\S+ version \S+(?: \([^()]+\))?")
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
