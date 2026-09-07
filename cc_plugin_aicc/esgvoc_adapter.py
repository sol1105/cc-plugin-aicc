"""Adapt ESGVoc coordinate descriptors to AICC's internal metadata shape.

The checks intentionally consume one small, source-independent dictionary
format.  This module is the only place that knows about ESGVoc's Pydantic
models and field names; the default CMOR JSON path remains unchanged.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packaging.version import InvalidVersion, Version


_COMMON_FIELDS = {
    "axis": "axis",
    "data_type": "type",
    "long_name": "long_name",
    "cf_standard_name": "standard_name",
    "out_name": "out_name",
    "units": "units",
    "positive": "positive",
    "stored_direction": "stored_direction",
    "tolerance": "tolerance",
    "valid_min": "valid_min",
    "valid_max": "valid_max",
}

_DATA_COORDINATE_FIELDS = [
    "id",
    "coordinate_type",
    "axis",
    "data_type",
    "long_name",
    "cf_standard_name",
    "out_name",
    "units",
    "positive",
    "stored_direction",
    "coordinate_values",
    "coordinate_bounds",
    "bounds_required",
    "tolerance",
    "valid_min",
    "valid_max",
    "is_climatology",
    "is_generic_model_level_coordinate",
]

_MODEL_LEVEL_FIELDS = [
    "id",
    "axis",
    "data_type",
    "long_name",
    "cf_standard_name",
    "out_name",
    "units",
    "positive",
    "stored_direction",
    "bounds_required",
    "valid_min",
    "valid_max",
    "formula",
    "z_factors",
    "z_bounds_factors",
    "generic_level_name",
]

_FORMULA_TERM_FIELDS = [
    "id",
    "data_type",
    "long_name",
    "cf_standard_name",
    "out_name",
    "units",
    "dimensions",
]

_GRID_VARIABLE_FIELDS = [
    "id",
    "data_type",
    "long_name",
    "cf_standard_name",
    "out_name",
    "units",
    "dimensions",
    "valid_min",
    "valid_max",
]

_GRID_AXIS_FIELDS = [
    "id",
    "axis",
    "data_type",
    "long_name",
    "cf_standard_name",
    "out_name",
    "units",
]

_PROJECT_ID = "cmip7"
_MINIMUM_ESGVOC_VERSION = Version("5.1.0")


def _as_dict(record: Any) -> dict:
    """Return a plain dictionary for a Pydantic model, subset, or test double."""
    if isinstance(record, dict):
        return record
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="python")
    return dict(record)


def _reference_id(value: Any) -> str:
    """Extract an ESGVoc term ID from a resolved or unresolved reference."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("id", ""))
    return str(getattr(value, "id", ""))


def _reference_ids(values: Any) -> list[str]:
    if not values:
        return []
    return [_reference_id(value) for value in values if _reference_id(value)]


def _common_entry(record: Any) -> dict:
    source = _as_dict(record)
    entry = {}
    for origin, target in _COMMON_FIELDS.items():
        value = source.get(origin)
        if value is None:
            value = ""
        elif origin in {"tolerance", "valid_min", "valid_max"}:
            # The existing CMOR adapter receives these as strings.  Retaining
            # that representation also keeps zero distinct from an unset value
            # in validation code inherited from the JSON-table path.
            value = str(value)
        entry[target] = value
    if "dimensions" in source:
        entry["dimensions"] = _reference_ids(source.get("dimensions"))
    return entry


def _flatten_bound_edges(edges: list) -> list:
    """Convert ESGVoc's m+1 edge vector to CMOR's flattened bound pairs."""
    return [edge for pair in zip(edges[:-1], edges[1:]) for edge in pair]


def _data_coordinate_entry(record: Any) -> dict:
    source = _as_dict(record)
    entry = _common_entry(source)
    coordinate_type = _reference_id(source.get("coordinate_type"))
    entry["coordinate_type"] = coordinate_type
    entry["must_have_bounds"] = "yes" if source.get("bounds_required") is True else "no"
    entry["climatology"] = "yes" if source.get("is_climatology") else ""

    values = source.get("coordinate_values")
    bounds = source.get("coordinate_bounds") or []
    if coordinate_type == "scalar":
        if isinstance(values, list):
            value = values[0] if values else ""
        else:
            value = values if values is not None else ""
        entry["value"] = str(value) if value != "" else ""
        entry["bounds_values"] = " ".join(str(value) for value in bounds)
    else:
        if values is None:
            entry["requested"] = []
        elif isinstance(values, list):
            entry["requested"] = values
        else:
            entry["requested"] = [values]
        entry["requested_bounds"] = _flatten_bound_edges(bounds) if bounds else []
    return entry


def _formula_symbol(term_id: str, out_name: str) -> str:
    """Return the CF formula-term label represented by a descriptor."""
    identifier = term_id or out_name
    return re.sub(r"_(?:bnds|half)$", "", identifier)


def _factor_ids(factors: Any) -> list[str]:
    """Return factor IDs without expanding resolved ESGVoc models."""
    return [
        _reference_id(_as_dict(factor)) if not isinstance(factor, str) else factor
        for factor in factors or []
    ]


def _coordinate_formula_terms(
    formula: str, declared_symbols: set[str], coordinate_out_name: str
) -> list[str]:
    """Find formula terms represented by the model-level coordinate itself.

    ESGVoc's ``FormulaTerm`` descriptors cover separate variables.  In several
    CF parametric coordinates, however, the coordinate variable supplies one
    term itself (for example ``sigma: lev`` or ``a: lev``).  Such a term occurs
    on a formula right-hand side but is absent from ``z_factors``.  A literal
    use of the coordinate's own name, as in ``p0*lev``, needs no mapping.
    """
    if not formula or "=" not in formula:
        return []
    expressions = " ".join(re.findall(r"=\s*([^;]+)", formula))
    excluded = {
        coordinate_out_name,
        "and",
        "defined",
        "exp",
        "for",
        "is",
        "levels",
        "log",
        "max",
        "min",
        "not",
        "or",
        "where",
    }
    # n, k, j, and i are indices in CF's expanded ocean-coordinate formulas.
    excluded.update({"n", "k", "j", "i"})
    result = []
    for symbol in re.findall(r"\b[A-Za-z_]\w*\b", expressions):
        if (
            symbol not in declared_symbols
            and symbol not in excluded
            and symbol not in result
        ):
            result.append(symbol)
    return result


def _formula_mapping(
    factors: Any,
    all_terms: dict[str, dict],
    generic_id: str,
    *,
    bounds: bool,
) -> str:
    """Build a CF ``formula_terms`` string from structured ESGVoc terms.

    Formula-term IDs distinguish level, half-level, and bounds variants while
    ``out_name`` holds the variable name.  Selecting by ``generic_id`` also
    repairs databases in which the two factor relationships were combined by
    an older JSON-LD context.
    """
    mappings = []
    for factor in factors or []:
        item = _as_dict(factor) if not isinstance(factor, str) else {"id": factor}
        item_id = _reference_id(item)
        item_out_name = str(item.get("out_name") or item_id)
        symbol = _formula_symbol(item_id, item_out_name)

        candidates = []
        for candidate_id, candidate in all_terms.items():
            candidate_out = str(candidate.get("out_name") or candidate_id)
            if _formula_symbol(candidate_id, candidate_out) != symbol:
                continue
            dimensions = candidate.get("dimensions", [])
            has_generic_dimension = generic_id in dimensions
            is_bounds_term = candidate_id.endswith("_bnds") or candidate_out.endswith(
                "_bnds"
            )
            if has_generic_dimension:
                desired = is_bounds_term if bounds else not is_bounds_term
                candidates.append((0 if desired else 2, candidate_id, candidate))
            elif candidate_id == item_id:
                # Horizontal or scalar factors such as ps, orog, depth_c.
                candidates.append((1, candidate_id, candidate))

        if candidates:
            _, _, selected = min(candidates, key=lambda value: (value[0], value[1]))
            variable_name = str(selected.get("out_name") or selected.get("id"))
        else:
            variable_name = item_out_name
        mappings.append(f"{symbol}: {variable_name}")
    return " ".join(mappings)


def _model_level_entry(record: Any, formula_terms: dict[str, dict]) -> dict:
    source = _as_dict(record)
    entry = _common_entry(source)
    generic_id = _reference_id(source.get("generic_level_name"))
    bounds_required = source.get("bounds_required") is True
    z_factors = source.get("z_factors")
    z_bounds_factors = source.get("z_bounds_factors")
    has_distinct_bounds_factors = bool(z_bounds_factors) and _factor_ids(
        z_bounds_factors
    ) != _factor_ids(z_factors)
    main_mapping = _formula_mapping(z_factors, formula_terms, generic_id, bounds=False)
    declared_symbols = set(re.findall(r"(\w+)\s*:", main_mapping))
    self_symbols = _coordinate_formula_terms(
        source.get("formula") or "",
        declared_symbols,
        source.get("out_name") or "lev",
    )
    self_mapping = " ".join(
        f"{symbol}: {source.get('out_name') or 'lev'}" for symbol in self_symbols
    )
    if self_mapping:
        main_mapping = f"{self_mapping} {main_mapping}".strip()

    bounds_mapping = ""
    if bounds_required or has_distinct_bounds_factors:
        bounds_mapping = _formula_mapping(
            z_bounds_factors,
            formula_terms,
            generic_id,
            bounds=True,
        )
        if self_symbols:
            coordinate_bounds_name = f"{source.get('out_name') or 'lev'}_bnds"
            self_bounds_mapping = " ".join(
                f"{symbol}: {coordinate_bounds_name}" for symbol in self_symbols
            )
            bounds_mapping = f"{self_bounds_mapping} {bounds_mapping}".strip()

    entry.update(
        {
            "generic_level_name": generic_id,
            "must_have_bounds": "yes" if bounds_required else "no",
            "formula": source.get("formula") or "",
            "z_factors": main_mapping,
            # An older ESGVoc JSON-LD context aliases z_factors onto an otherwise
            # absent z_bounds_factors field.  Equal optional lists are therefore
            # treated as the alias; genuinely distinct optional bounds terms are
            # retained.
            "z_bounds_factors": bounds_mapping,
        }
    )
    return entry


def _plain_entry(record: Any) -> dict:
    return _common_entry(record)


def _records_by_id(records: list, converter) -> dict[str, dict]:
    result = {}
    for record in records:
        identifier = _reference_id(_as_dict(record))
        if identifier:
            result[identifier] = converter(record)
    return result


def _require_supported_esgvoc_version(installed_version: str | None = None) -> str:
    """Require the first ESGVoc release containing coordinate descriptors."""
    if installed_version is None:
        try:
            installed_version = version("esgvoc")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"The 'aicc:esgvoc' setup path requires "
                f"esgvoc>={_MINIMUM_ESGVOC_VERSION}. "
                "No ESGVoc installation was found."
            ) from exc
    try:
        parsed = Version(installed_version)
    except InvalidVersion as exc:
        raise RuntimeError(
            f"The 'aicc:esgvoc' setup path requires "
            f"esgvoc>={_MINIMUM_ESGVOC_VERSION}, but the "
            f"installed version {installed_version!r} is not a valid version."
        ) from exc
    if parsed < _MINIMUM_ESGVOC_VERSION:
        raise RuntimeError(
            f"The 'aicc:esgvoc' setup path requires "
            f"esgvoc>={_MINIMUM_ESGVOC_VERSION} because earlier releases do not "
            "provide the required coordinate descriptor models; "
            f"found esgvoc=={installed_version}."
        )
    return installed_version


def _get_api():
    try:
        import esgvoc.api as esgvoc_api
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            f"The 'aicc:esgvoc' setup path requires "
            f"esgvoc>={_MINIMUM_ESGVOC_VERSION}, but ESGVoc could not be "
            f"imported: {type(exc).__name__}: {exc}"
        ) from exc
    _require_supported_esgvoc_version()
    return esgvoc_api


def _all(api, descriptor: str, fields: list[str]):
    """Read one required ESGVoc collection with an actionable error."""
    try:
        records = api.get_all_terms_in_collection(_PROJECT_ID, descriptor, fields)
    except Exception as exc:
        raise RuntimeError(
            f"The 'aicc:esgvoc' setup path could not read project "
            f"{_PROJECT_ID!r} {descriptor!r} metadata: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not records:
        raise RuntimeError(
            f"The 'aicc:esgvoc' setup path received an empty project "
            f"{_PROJECT_ID!r} {descriptor!r} collection; the coordinate "
            "catalogue is incomplete."
        )
    return records


def load_esgvoc_metadata(branded_variable: str, api=None) -> dict:
    """Load the ESGVoc terms needed by AICC for one branded variable."""
    api = api or _get_api()

    try:
        branded = api.get_term_in_data_descriptor(
            "known_branded_variable",
            branded_variable,
            ["id", "out_name", "dimensions", "table_id"],
        )
    except Exception as exc:
        raise RuntimeError(
            "The 'aicc:esgvoc' setup path failed while reading "
            f"known_branded_variable {branded_variable!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    data_coordinates = _all(api, "data_coordinate", _DATA_COORDINATE_FIELDS)
    formula_records = _all(api, "formula_term", _FORMULA_TERM_FIELDS)
    formula_terms = _records_by_id(formula_records, _plain_entry)
    model_levels = _all(api, "model_level_coordinate", _MODEL_LEVEL_FIELDS)
    grid_variables = _all(api, "grid_variable", _GRID_VARIABLE_FIELDS)
    grid_axes = _all(api, "grid_axis", _GRID_AXIS_FIELDS)

    axis_entries = _records_by_id(data_coordinates, _data_coordinate_entry)
    axis_entries.update(
        _records_by_id(
            model_levels,
            lambda record: _model_level_entry(record, formula_terms),
        )
    )

    var_entry = None
    table_name = None
    if branded is not None:
        branded_data = _as_dict(branded)
        var_entry = {
            "out_name": branded_data.get("out_name") or "",
            "dimensions": _reference_ids(branded_data.get("dimensions")),
        }
        missing = [
            identifier
            for identifier in var_entry["dimensions"]
            if identifier not in axis_entries
        ]
        if missing:
            raise RuntimeError(
                f"The 'aicc:esgvoc' setup path found coordinate ID(s) {missing} "
                f"referenced by known_branded_variable {branded_variable!r}, but "
                "they are absent from the project 'cmip7' coordinate catalogue."
            )
        table_ids = _reference_ids(branded_data.get("table_id"))
        table_name = table_ids[0] if table_ids else "esgvoc"

    return {
        "coordinates": {"axis_entry": axis_entries},
        "grids": {
            "variable_entry": _records_by_id(grid_variables, _plain_entry),
            "axis_entry": _records_by_id(grid_axes, _plain_entry),
        },
        "formulas": {"formula_entry": formula_terms},
        "variable": var_entry,
        "table_name": table_name,
    }
