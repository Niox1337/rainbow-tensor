"""Describe the installed teaching API without initializing notebook controls.

The operation inventory comes from the public view exports. Both interactive
dispatch and the machine-readable reference use that inventory, so an exported
operation cannot quietly appear in one list while remaining absent from another.
"""

from inspect import signature


def explorable_operations():
    """Return public operation views with an output that can be selected.

    Shape inspection has no operation result to trace. Storage inspection is not a
    member of the view registry. Imports stay local to avoid initializing a widget
    or introducing a dependency cycle while the public package is being imported.
    """
    from . import views

    return tuple(getattr(views, name) for name in views.__all__ if name != "shape")


def walkthrough_operations():
    """Return standalone calculations that the walkthrough factory can record.

    Other calculations remain available through an explicitly recorded Flow. This
    list describes factory input support, not the operations allowed inside a Flow.
    """
    from .views import matmul, mean, sum

    return sum, mean, matmul


def capabilities():
    """Return a fresh JSON-serializable description of this installed version.

    ``schema_version`` identifies the capability document format, independently of
    the package version. Signatures are inspected from the actual public callables.
    Consumers should use this inventory to choose an operation, then read its
    docstring for value and axis restrictions. A capability is not a promise that a
    notebook host has a working widget manager.

    The returned dictionaries and lists belong to the caller. Editing them cannot
    change public operations, controller dispatch, or a later capability document.
    """
    from . import __version__, views
    from .memory import memory
    from .provenance import Flow

    interactive = frozenset(explorable_operations())
    guided = frozenset(walkthrough_operations())
    functions = [*(getattr(views, name) for name in views.__all__), memory]
    operations = {}
    for function in sorted(functions, key=lambda item: item.__name__):
        name = function.__name__
        method = getattr(Flow, name, None)
        operations[name] = {
            "call": f"rainbow_tensor.{name}",
            "signature": str(signature(function)),
            "flow_signature": str(signature(method)) if callable(method) else None,
            "focus": "focus" in signature(function).parameters,
            "explore": function in interactive,
            "walkthrough": function in guided,
        }
    return {
        "schema_version": 1,
        "package": {"name": "rainbow-tensor", "version": __version__},
        "operations": operations,
        "controllers": {
            "explore": {
                "tracked_tensor": True,
                "operations": sorted(name for name, item in operations.items() if item["explore"]),
                "requires_live_kernel": True,
            },
            "walkthrough": {
                "tracked_tensor": True,
                "operations": sorted(
                    name for name, item in operations.items() if item["walkthrough"]
                ),
                "requires_live_kernel": True,
            },
            "reduction_playground": {
                "operations": ["sum", "mean"],
                "prediction": "result_shape",
                "requires_live_kernel": True,
            },
        },
        "semantics": {
            "arithmetic": "python_scalars",
            "backend_kernels": False,
            "divide_by_zero": "raises_ZeroDivisionError",
            "shape_tuples": "generated_row_major_values",
            "flow_history": "explicitly_recorded_operations_only",
            "structural_trace_reads_values": False,
            "occurrence_counts": "participation_paths_not_coefficients_or_derivatives",
            "scalar_coordinate": [],
            "empty_output_coordinate": None,
            "input_values": "live_per_request",
            "input_shapes": "fixed_after_recording",
            "truncated_trace": "reached_paths_only",
            "over_budget_values": "unknown_never_partial_results",
        },
    }
