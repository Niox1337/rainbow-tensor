"""Reductions and math views.

Public functions ``matmul``, ``sum``, and ``mean``. Matmul draws both operands
and the output and highlights the contraction; the reductions mark the elements
that fold into each result group.
"""

import builtins
from functools import cache

from ..evaluation import DEFAULT_MAX_TERMS, evaluation_explanation, value_evaluation
from ..explanations import t
from ..numerics import numeric_explanation, numeric_semantics
from ..ops import (
    matmul_result_shape,
    reduce_result_shape,
    reduce_source_coords,
)
from ..ops.reductions import iter_matmul_source_terms
from ..renderers import resolve_renderer
from ..selection import BasicSelection
from ..shape import extract_shape, flat_index, format_shape
from ..theme import resolve_theme
from ..tracing import _build_trace, _normalize_focus, _trace_explanation
from ..visual import (
    _operand_tint,
    _preview_explanation,
    _shape_caption_parts,
    _source_value,
    _value_fn_for,
    _visual,
)

# The public ``sum`` below shadows the builtin in this module, so keep a handle
# to the real one for the reduction maths.
_py_sum = builtins.sum


def matmul(
    a, b, theme=None, precision=2, renderer=None, *, focus=None, max_terms=DEFAULT_MAX_TERMS
):
    """Visualise a matrix multiplication ``a @ b``.

    ``a`` and ``b`` are array-likes or shape tuples. Vector, matrix, and batched
    matrix multiplication follow the shape rules of ``numpy.matmul``. The two
    operands and the output are drawn side by side. The shared inner axis is
    marked in the accent colour, and the row of the first operand and the column
    of the second operand that combine into the first output element are
    highlighted, so the contraction reads directly off the figure.

    Numerical previews use Python scalar arithmetic on values read from the
    inputs, not the backend's matrix multiplication kernel. Accumulation dtype,
    rounding, and overflow may differ. Shape tuples supply generated row-major
    placeholder values. ``visual.metadata["numeric_semantics"]`` records this
    model and each operand's value source, even when evaluation is skipped.

    ``focus`` selects an output coordinate instead of the first one. Negative
    indices are supported and a scalar output uses ``()``. ``visual.trace``
    records up to eight ordered source terms without reading array values.

    ``max_terms`` limits contraction terms per output cell, defaulting to
    10,000. Outputs that exceed it display ``?`` without partial evaluation.
    Pass ``None`` to evaluate every term. Source previews and coordinate traces
    remain available when numerical output is skipped. Evaluation details are
    stored in ``visual.metadata["value_evaluation"]``.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    a_shape = extract_shape(a)
    b_shape = extract_shape(b)
    result = matmul_result_shape(a_shape, b_shape)
    focused = _normalize_focus(focus, result)
    evaluation = value_evaluation(a_shape[-1], max_terms)
    semantics = numeric_semantics((a, b))
    display_result = result or (1,)
    a_value = _source_value(a, a_shape)
    b_value = _source_value(b, b_shape)

    @cache
    def result_value(coord):
        if evaluation["status"] == "skipped":
            return "?"
        out_coord = () if not result else coord
        terms = iter_matmul_source_terms(out_coord, a_shape, b_shape)
        return _py_sum(a_value(ac) * b_value(bc) for ac, bc in terms)

    trace = _build_trace(
        "matmul", focused, a_shape[-1], iter_matmul_source_terms(focused, a_shape, b_shape)
    )
    if evaluation["status"] == "skipped":
        terms0 = [tuple(ref.coordinate for ref in term) for term in trace.terms]
    else:
        terms0 = list(iter_matmul_source_terms(focused, a_shape, b_shape))
    a_selected = sorted({ac for ac, _ in terms0})
    b_selected = sorted({bc for _, bc in terms0})
    result_selected = [focused or (0,)]

    a_inner = len(a_shape) - 1
    b_inner = len(b_shape) - 2 if len(b_shape) >= 2 else 0

    def a_color(axis):
        if axis == a_inner:
            return theme.surface_selected
        return theme.axis_color(axis) if axis < len(a_shape) - 1 else theme.text_muted

    def b_color(axis):
        if axis == b_inner:
            return theme.surface_selected
        return theme.axis_color(axis) if axis < len(b_shape) - 1 else theme.text_muted

    b_theme = theme.variant(
        axis_colors=tuple(b_color(axis) for axis in range(len(b_shape)))
    )

    inner = a_shape[a_inner]
    rows = "n/a" if len(a_shape) == 1 else a_shape[-2]
    cols = "n/a" if len(b_shape) == 1 else b_shape[-1]
    explanation = [
        t("matmul.operands", a=format_shape(a_shape), b=format_shape(b_shape)),
        t("matmul.dims", rows=rows, cols=cols, inner=inner),
        t("common.result_shape", shape=format_shape(result)),
        t("matmul.combine") if focus is None else (
            "The highlighted row and column combine into the focused output element."
        ),
    ] + _preview_explanation([a_shape, b_shape, display_result], theme)
    explanation.extend(evaluation_explanation(evaluation, len(trace.terms)))
    explanation.extend(numeric_explanation(semantics))
    if focus is not None:
        explanation.extend(_trace_explanation(trace))
    panels = [
        {
            "shape": a_shape,
            "value_fn": _value_fn_for(a),
            "selected": a_selected,
            "caption_parts": _shape_caption_parts("A", a_shape, theme, color_for=a_color),
        },
        {
            "shape": b_shape,
            "value_fn": _value_fn_for(b),
            "selected": b_selected,
            "theme": b_theme,
            "caption_parts": _shape_caption_parts("B", b_shape, theme, color_for=b_color),
        },
        {
            "shape": display_result,
            "value_fn": result_value,
            "selected": result_selected,
            "caption_parts": _shape_caption_parts("matmul", display_result, theme),
        },
    ]
    content = renderer.render_panels(
        panels=panels,
        connectors=["@", "->"],
        explanation=explanation,
        theme=theme,
        precision=precision,
    )
    visual = _visual(
        content, a_shape, renderer, selected=a_selected, result=result, explanation=explanation
    )
    visual.trace = trace
    visual.metadata["value_evaluation"] = evaluation
    visual.metadata["numeric_semantics"] = semantics
    return visual


def _reduce(array, axis, op_name, theme, precision, renderer, focus, max_terms):
    """Render a reduction without evaluating hidden output groups.

    Each requested output coordinate is evaluated once per visual. A second
    layout pass may request it again when a wide value grows the cells, so the
    local cache avoids repeating backend reads. The cache never reuses values
    across calls on a mutable array.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shape = extract_shape(array)
    result = reduce_result_shape(shape, axis)
    focused = _normalize_focus(focus, result)
    source_value = _source_value(array, shape)
    axis = axis + len(shape) if axis < 0 else axis
    evaluation = value_evaluation(shape[axis], max_terms)
    semantics = numeric_semantics((array,))

    # Mark the source elements that collapse into the focused result element, and
    # tint every other group so values that fold into the same result share one
    # background while the focused group stays clearly highlighted.
    source_index = focused[:axis] + (slice(None),) + focused[axis:]
    selected = BasicSelection(shape, source_index)
    focused_group = flat_index(focused, result) if result else 0
    trace = _build_trace(
        op_name, focused, shape[axis],
        ((coord,) for coord in reduce_source_coords(focused, shape, axis)),
        divisor=shape[axis] if op_name == "mean" else 1,
    )

    def source_tint(coord):
        rc = coord[:axis] + coord[axis + 1:]
        group = flat_index(rc, result) if result else 0
        if group == focused_group:
            return None  # this group is shown through the selected highlight
        return _operand_tint(theme, group - 1)

    disp = result or (1,)

    # The surviving axes keep their original source colours instead of being
    # recoloured by their new position, so reducing a non-last axis does not
    # swap an axis frame to a different colour.
    surviving = [i for i in range(len(shape)) if i != axis]
    if result:
        result_theme = theme.variant(
            axis_colors=tuple(theme.axis_color(surviving[r]) for r in range(len(result)))
        )
    else:
        result_theme = theme

    def result_color(ax):
        return result_theme.axis_color(ax) if ax < len(result) - 1 else theme.text_muted

    # The result element of each group takes the same background as its source
    # group, so a result cell and the values that fold into it read as one unit.
    def result_tint(coord):
        group = flat_index(coord, disp)
        if group == focused_group:
            return None
        return _operand_tint(theme, group - 1)

    selected_result = [focused or (0,)]

    @cache
    def result_value(coord):
        """Evaluate one visible group using a streaming sum of source values."""
        if evaluation["status"] == "skipped":
            return "?"
        rc = () if not result else coord
        total = _py_sum(source_value(sc) for sc in reduce_source_coords(rc, shape, axis))
        return total / shape[axis] if op_name == "mean" else total

    explanation = [
        t("common.original_shape", shape=format_shape(shape)),
        t("reduce.reducing", axis=axis, op=op_name),
        t("common.result_shape", shape=format_shape(result)),
        t("reduce.combines", count=shape[axis], axis=axis),
        t("reduce.share_background") if focus is None else (
            "Values that fold into the same result share its background, "
            "and the focused group is highlighted."
        ),
    ] + _preview_explanation([shape, disp], theme)
    explanation.extend(evaluation_explanation(evaluation))
    explanation.extend(numeric_explanation(semantics))
    if focus is not None:
        explanation.extend(_trace_explanation(trace))
    panels = [
        {
            "shape": shape,
            "value_fn": _value_fn_for(array),
            "selected": selected,
            "cell_tint": source_tint,
            "caption_parts": _shape_caption_parts("source", shape, theme),
        },
        {
            "shape": disp,
            "value_fn": result_value,
            "selected": selected_result,
            "cell_tint": result_tint,
            "theme": result_theme,
            "caption_parts": _shape_caption_parts(
                op_name, disp, theme, color_for=result_color
            ),
        },
    ]
    content = renderer.render_panels(
        panels=panels, connectors=["->"], explanation=explanation, theme=theme, precision=precision
    )
    visual = _visual(
        content, shape, renderer, selected=selected, result=result, explanation=explanation
    )
    visual.trace = trace
    visual.metadata["value_evaluation"] = evaluation
    visual.metadata["numeric_semantics"] = semantics
    return visual


def sum(
    array, axis, theme=None, precision=2, renderer=None, *, focus=None, max_terms=DEFAULT_MAX_TERMS
):
    """Visualise a sum reduction over ``axis``.

    The source panel marks the elements that collapse into the first result
    element, and the result panel holds the per group sums with the reduced
    axis gone.

    Numerical previews use Python scalar arithmetic, not a native backend
    reduction. Accumulation dtype, rounding, and overflow may differ. Shape
    tuples supply generated row-major placeholder values. The model and value
    source are recorded in ``visual.metadata["numeric_semantics"]`` even when
    evaluation is skipped.

    Pass ``focus`` as an output coordinate tuple to explain another group.
    Negative coordinates are supported and a scalar result uses ``()``.
    ``visual.trace`` retains up to eight ordered source terms for that output.

    ``max_terms`` limits source terms per output cell, defaulting to 10,000.
    Longer reductions show ``?`` instead of evaluating a partial sum. Set it
    to ``None`` to evaluate every term. The coordinate selection stays compact,
    and ``visual.metadata["value_evaluation"]`` records the evaluation decision.
    """
    return _reduce(array, axis, "sum", theme, precision, renderer, focus, max_terms)


def mean(
    array, axis, theme=None, precision=2, renderer=None, *, focus=None, max_terms=DEFAULT_MAX_TERMS
):
    """Visualise a mean reduction over ``axis``.

    The source panel marks the elements that collapse into the first result
    element, and the result panel holds the per group means with the reduced
    axis gone.

    Numerical previews sum input values with Python scalar arithmetic and
    divide by the axis size, rather than calling a native backend reduction.
    Accumulation dtype, rounding, and overflow may differ. Shape tuples supply
    generated row-major placeholder values. The model and value source are
    recorded in ``visual.metadata["numeric_semantics"]`` even when evaluation
    is skipped.

    Pass ``focus`` as an output coordinate tuple to explain another group.
    Negative coordinates are supported and a scalar result uses ``()``.
    ``visual.trace`` retains up to eight terms and the mean's divisor.

    ``max_terms`` limits source terms per output cell, defaulting to 10,000.
    Longer reductions show ``?`` instead of evaluating a partial mean. Set it
    to ``None`` to evaluate every term. The coordinate selection stays compact,
    and ``visual.metadata["value_evaluation"]`` records the evaluation decision.
    """
    return _reduce(array, axis, "mean", theme, precision, renderer, focus, max_terms)
