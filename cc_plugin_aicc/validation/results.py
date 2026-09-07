"""Helpers for constructing and emitting check results."""

from compliance_checker.base import BaseCheck, TestCtx

from cc_plugin_aicc.utils import _format_attribute, _ncattr


def append_context_results(results: list, *contexts) -> None:
    """Append results only for contexts which recorded messages."""
    results.extend(ctx.to_result() for ctx in contexts if ctx.messages)


def vertical_config_context(ds, name: str) -> TestCtx:
    """Return the common prerequisite failure for model-specific Z checks."""
    ctx = TestCtx(BaseCheck.HIGH, name)
    source_id = _ncattr(ds, "source_id")
    ctx.add_failure(
        f"Vertical checks cannot run because source_id "
        f"{_format_attribute(source_id)} is not registered in the model "
        f"configuration. Add a matching source_id key to the default "
        f"configuration or pass it through the 'vertical_config' option."
    )
    return ctx
