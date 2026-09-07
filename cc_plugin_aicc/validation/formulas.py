"""Safe evaluation and sampling of CMOR vertical-coordinate formulas."""

import ast
import re

import numpy as np

from cc_plugin_aicc.validation.attributes import _numeric_profile


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
            f"formula term '{var.name}' did not reduce to a scalar at the sampled point"
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
            ast.Pow: lambda: left**right,
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
        expression = re.sub(rf"\b{re.escape(term)}\s*\([^()]*\)", term, expression)
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
            "sampled point " + ", ".join(f"{dim}={point[dim]}" for dim in sample_dims)
            if sample_dims
            else "vertical-only formula terms"
        )
        return profile, sample

    detail = f": {last_error}" if last_error else ""
    raise ValueError(f"no usable sampled point was found{detail}")
