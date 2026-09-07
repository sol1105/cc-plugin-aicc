"""Time-coordinate checks for AICC."""

from compliance_checker.base import BaseCheck, TestCtx
from compliance_checker.cf import util as cfutil

from cc_plugin_aicc.utils import (
    _compare_units,
    _format_attribute,
    _is_time_dim,
    _ncattr,
)
from cc_plugin_aicc.validation.attributes import _check_coord_attrs, _check_coord_type
from cc_plugin_aicc.validation.bounds import (
    _check_bounds_reference,
    _check_bounds_structure,
)
from cc_plugin_aicc.validation.results import append_context_results


class TimeChecks:
    """Time coordinate and climatology validation."""

    def check_time(self, ds):
        """Verify time coordinate(s) against the CMIP7 coordinate table."""
        time_dims = [d for d in self.requested_dims if _is_time_dim(d)]
        if not time_dims:
            ctx = TestCtx(BaseCheck.HIGH, "[AICC004] Time coordinate")
            ctx.add_pass()
            return [ctx.to_result()]

        results = []
        axis_entries = self.CTcoords.get("axis_entry", {})

        # Use cfutil to find the time variable (CF-aware, not name-based)
        t_var_name = cfutil.get_time_variable(ds)

        for time_dim_id in time_dims:
            ce = axis_entries.get(time_dim_id, {})
            out_name = ce.get("out_name", "time")
            must_have_bounds = ce.get("must_have_bounds", "no") == "yes"
            climatology_setting = ce.get("climatology", "")
            is_climatology = (
                climatology_setting is True or str(climatology_setting).lower() == "yes"
            )

            ctx = TestCtx(BaseCheck.HIGH, f"[AICC004] Time coordinate ({time_dim_id})")
            low_ctx = TestCtx(
                BaseCheck.LOW,
                f"[AICC004] Time coordinate ({time_dim_id}, advisory)",
            )
            medium_ctx = TestCtx(
                BaseCheck.MEDIUM,
                f"[AICC004] Time coordinate ({time_dim_id}, recommended)",
            )

            data_out_name = self.var_entry.get("out_name", "") if self.var_entry else ""
            if out_name not in ds.dimensions:
                ctx.add_failure(
                    f"Required time dimension '{out_name}' "
                    f"(dim_id='{time_dim_id}') not found in file."
                )
            elif (
                data_out_name
                and data_out_name in ds.variables
                and out_name not in ds.variables[data_out_name].dimensions
            ):
                ctx.add_failure(
                    f"Time-dependent variable '{data_out_name}' must use "
                    f"dimension '{out_name}'; found dimensions "
                    f"{list(ds.variables[data_out_name].dimensions)}."
                )
            else:
                ctx.add_pass()

            resolved_t = t_var_name or (out_name if out_name in ds.variables else None)
            if resolved_t is None:
                ctx.add_failure(
                    f"Time coordinate variable '{out_name}' (dim_id='{time_dim_id}') "
                    f"not found in file."
                )
                results.append(ctx.to_result())
                continue
            ctx.add_pass()

            t_var = ds.variables[resolved_t]

            if resolved_t != out_name or t_var.dimensions != (out_name,):
                ctx.add_failure(
                    f"Time coordinate must be '{out_name}({out_name})'; found "
                    f"'{resolved_t}({', '.join(t_var.dimensions)})'."
                )
            else:
                ctx.add_pass()

            # axis=T
            if _ncattr(t_var, "axis") != "T":
                ctx.add_failure(f"'{resolved_t}' must have attribute axis='T'.")
            else:
                ctx.add_pass()

            # Attributes from CMIP7_coordinate.json
            _check_coord_attrs(
                ctx, low_ctx, t_var, resolved_t, ce, medium_ctx=medium_ctx
            )
            _check_coord_type(ctx, t_var, resolved_t, ce)

            # Units must match the CMOR template exactly (reference date is free).
            units = _ncattr(t_var, "units")
            level, msg = _compare_units(units, ce.get("units", ""))
            if level != "ok":
                ctx.add_failure(f"'{resolved_t}' units: {msg}")
            else:
                ctx.add_pass()

            # calendar attribute
            if not _ncattr(t_var, "calendar"):
                ctx.add_failure(f"'{resolved_t}' is missing 'calendar' attribute.")
            else:
                ctx.add_pass()

            if is_climatology:
                # CF §7.4: time must carry a 'climatology' attribute
                clim_attr = _ncattr(t_var, "climatology")
                if not clim_attr:
                    ctx.add_failure(
                        f"Climatology time variable '{resolved_t}' must have a "
                        f"'climatology' attribute pointing to the bounds variable."
                    )
                else:
                    ctx.add_pass()
                    # Verify the referenced climatology bounds variable exists.
                    if clim_attr not in ds.variables:
                        ctx.add_failure(
                            f"Climatology bounds variable "
                            f"{_format_attribute(clim_attr)} "
                            f"(referenced by '{resolved_t}:climatology') not found."
                        )
                    else:
                        ctx.add_pass()
                        if clim_attr != "climatology_bnds":
                            medium_ctx.add_failure(
                                f"'{resolved_t}' climatology="
                                f"{_format_attribute(clim_attr)}; recommended name "
                                "is 'climatology_bnds'."
                            )
                        _check_bounds_structure(
                            ctx,
                            medium_ctx,
                            ds.variables[clim_attr],
                            clim_attr,
                            t_var,
                            ce,
                        )
                regular_bounds = _ncattr(t_var, "bounds")
                if regular_bounds:
                    ctx.add_failure(
                        f"Climatological time coordinate '{resolved_t}' must not "
                        f"have a 'bounds' attribute; found "
                        f"{_format_attribute(regular_bounds)}."
                    )
                else:
                    ctx.add_pass()
                regular_name = f"{out_name}_bnds"
                if regular_name in ds.variables and clim_attr != regular_name:
                    ctx.add_failure(
                        f"Climatological time coordinate '{resolved_t}' must not "
                        f"define regular bounds variable '{regular_name}' unless "
                        f"its 'climatology' attribute names that same variable."
                    )
                else:
                    ctx.add_pass()
            elif must_have_bounds or bool(_ncattr(t_var, "bounds")):
                # Regular time bounds
                bnds_name, bnds_var = _check_bounds_reference(
                    ctx, medium_ctx, ds, t_var, resolved_t, out_name
                )
                if bnds_var is not None:
                    if bnds_var.ncattrs():
                        ctx.add_failure(
                            f"'{bnds_name}' must have no attributes; "
                            f"found: {list(bnds_var.ncattrs())}."
                        )
                    else:
                        ctx.add_pass()
                    _check_bounds_structure(
                        ctx, medium_ctx, bnds_var, bnds_name, t_var, ce
                    )

            results.append(ctx.to_result())
            append_context_results(results, medium_ctx, low_ctx)

        return results

    # ------------------------------------------------------------------
