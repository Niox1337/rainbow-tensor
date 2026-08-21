"""Reductions and math views.

Public functions ``matmul``, ``sum``, and ``mean``. Matmul draws both operands
and the output and highlights the contraction; the reductions mark the elements
that fold into each result group.
"""

import builtins

from ..evaluation import (
    DEFAULT_MAX_TERMS,
    DEFAULT_MAX_TOTAL_TERMS,
    budgeted_values,
    evaluation_explanation,
    evaluation_plan,
)
from ..explanations import t
from ..numerics import numeric_explanation, numeric_semantics
from ..ops import (
    matmul_result_shape,
    reduce_result_shape,
    reduce_source_coords,
)
from ..ops.reductions import (
    iter_matmul_source_terms,
    normalize_reduction_axes,
    reduce_source_index,
    reduce_term_count,
    validate_keepdims,
)
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
    a, b, theme=None, precision=2, renderer=None, *, focus=None, max_terms=DEFAULT_MAX_TERMS,
    max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
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
    ``max_total_terms`` additionally caps all visible output terms at 100,000.
    If either limit is exceeded, all output values are skipped. Set both limits
    to ``None`` to remove them. Source previews and coordinate traces
    remain available when numerical output is skipped. Evaluation details are
    stored in ``visual.metadata["value_evaluation"]``.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    a_shape = extract_shape(a)
    b_shape = extract_shape(b)
    result = matmul_result_shape(a_shape, b_shape)
    focused = _normalize_focus(focus, result)
    semantics = numeric_semantics((a, b))
    display_result = result or (1,)
    result_selected = [focused or (0,)]
    evaluation = evaluation_plan(
        a_shape[-1], max_terms, max_total_terms, display_result, result_selected, theme
    )
    a_value = _source_value(a, a_shape)
    b_value = _source_value(b, b_shape)

    @budgeted_values(evaluation)
    def result_value(coord):
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


def _reduce(
    array, axis, op_name, theme, precision, renderer, keepdims, focus, max_terms, max_total_terms
):
    """Render a reduction without evaluating hidden output groups.

    Each requested output coordinate is evaluated once per visual. A second
    layout pass may request it again when a wide value grows the cells, so the
    local cache avoids repeating backend reads. The cache never reuses values
    across calls on a mutable array.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shape = extract_shape(array)
    axes = normalize_reduction_axes(axis, len(shape))
    keepdims = validate_keepdims(keepdims)
    result = reduce_result_shape(shape, axes, keepdims=keepdims)
    focused = _normalize_focus(focus, result)
    source_value = _source_value(array, shape)
    semantics = numeric_semantics((array,))
    term_count = reduce_term_count(shape, axes)
    reduced = set(axes)

    # Mark the source elements that collapse into the focused result element, and
    # tint every other group so values that fold into the same result share one
    # background while the focused group stays clearly highlighted.
    source_index = reduce_source_index(focused, shape, axes, keepdims=keepdims)
    selected = BasicSelection(shape, source_index)
    focused_group = flat_index(focused, result) if result else 0
    trace = _build_trace(
        op_name, focused, term_count,
        ((coord,) for coord in reduce_source_coords(focused, shape, axes, keepdims=keepdims)),
        divisor=term_count if op_name == "mean" else 1,
    )

    def source_tint(coord):
        rc = (
            tuple(0 if i in reduced else c for i, c in enumerate(coord))
            if keepdims else tuple(c for i, c in enumerate(coord) if i not in reduced)
        )
        group = flat_index(rc, result) if result else 0
        if group == focused_group:
            return None  # this group is shown through the selected highlight
        return _operand_tint(theme, group - 1)

    disp = result or (1,)

    # The surviving axes keep their original source colours instead of being
    # recoloured by their new position, so reducing a non-last axis does not
    # swap an axis frame to a different colour.
    surviving = [i for i in range(len(shape)) if i not in reduced]
    if result:
        colors = (
            tuple(
                theme.surface_selected if i in reduced else theme.axis_color(i)
                for i in range(len(shape))
            ) if keepdims else tuple(theme.axis_color(i) for i in surviving)
        )
        result_theme = theme.variant(
            axis_colors=colors
        )
    else:
        result_theme = theme

    def result_color(ax):
        if keepdims and ax in reduced:
            return theme.surface_selected
        return result_theme.axis_color(ax) if ax < len(result) - 1 else theme.text_muted

    # The result element of each group takes the same background as its source
    # group, so a result cell and the values that fold into it read as one unit.
    def result_tint(coord):
        group = flat_index(coord, disp)
        if group == focused_group:
            return None
        return _operand_tint(theme, group - 1)

    selected_result = [focused or (0,)]
    evaluation = evaluation_plan(
        term_count, max_terms, max_total_terms, disp, selected_result, result_theme
    )

    @budgeted_values(evaluation)
    def result_value(coord):
        """Evaluate one visible group using a streaming sum of source values."""
        rc = () if not result else coord
        total = _py_sum(
            source_value(sc) for sc in reduce_source_coords(rc, shape, axes, keepdims=keepdims)
        )
        return total / term_count if op_name == "mean" else total

    if len(axes) == 1:
        reducing = t("reduce.reducing", axis=axes[0], op=op_name)
        combines = t("reduce.combines", count=term_count, axis=axes[0])
    elif axes:
        reducing = f"Reducing axes {axes} with {op_name}."
        combines = f"Each result element combines {term_count} values from axes {axes}."
    else:
        reducing = "No axes are reduced because axis=()."
        combines = "Each result element uses the source value at the same coordinate."
    explanation = [
        t("common.original_shape", shape=format_shape(shape)),
        reducing,
        t("common.result_shape", shape=format_shape(result)),
        combines,
        t("reduce.share_background") if focus is None else (
            "Values that fold into the same result share its background, "
            "and the focused group is highlighted."
        ),
    ] + _preview_explanation([shape, disp], theme)
    if keepdims and axes:
        explanation.append(
            f"keepdims=True retains axes {axes} at length 1 for broadcasting. "
            "Their result dimensions use the highlight colour."
        )
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
    visual.metadata["reduction"] = {
        "axes": axes, "keepdims": keepdims, "term_count": term_count,
    }
    return visual


def sum(
    array, axis=None, theme=None, precision=2, renderer=None, *, keepdims=False, focus=None,
    max_terms=DEFAULT_MAX_TERMS,
    max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualise a sum over all axes, one axis, or a tuple of distinct axes.

    The source panel marks the elements that collapse into the first result
    element. ``axis=None`` reduces every axis, an integer reduces one axis,
    and a tuple reduces its named axes. Negative axes count from the end.
    ``axis=()`` leaves each value in its original position. Boolean and
    floating-point axes are rejected rather than coerced.

    ``keepdims=True`` retains every reduced axis at length one so the result
    can broadcast against the source. Those dimensions use the highlight
    colour. The default removes reduced axes, preserving surviving colours.
    ``visual.metadata["reduction"]`` records the normalized axes and term count.

    Numerical previews use Python scalar arithmetic, not a native backend
    reduction. Accumulation dtype, rounding, and overflow may differ. Shape
    tuples supply generated row-major placeholder values. The model and value
    source are recorded in ``visual.metadata["numeric_semantics"]`` even when
    evaluation is skipped.

    Pass ``focus`` as an output coordinate tuple to explain another group.
    Negative coordinates are supported and a scalar result uses ``()``.
    ``visual.trace`` retains up to eight terms in source row-major order,
    regardless of the order of axes in the tuple. Retained axes need a zero
    in the focus coordinate, for example ``(0, 1, 0)`` for shape ``(1, 3, 1)``.

    ``max_terms`` limits source terms per output cell, defaulting to 10,000.
    ``max_total_terms`` caps the sum of terms across visible outputs at 100,000.
    Exceeding either limit shows ``?`` for all outputs without partial sums.
    Set both limits to ``None`` to remove them. The selection stays compact,
    and ``visual.metadata["value_evaluation"]`` records the evaluation decision.
    """
    return _reduce(
        array, axis, "sum", theme, precision, renderer, keepdims, focus, max_terms, max_total_terms
    )


def mean(
    array, axis=None, theme=None, precision=2, renderer=None, *, keepdims=False, focus=None,
    max_terms=DEFAULT_MAX_TERMS,
    max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualise a mean over all axes, one axis, or a tuple of distinct axes.

    The source panel marks the elements that collapse into the first result
    element. ``axis=None`` reduces every axis, an integer reduces one axis,
    and a tuple reduces its named axes. Negative axes count from the end.
    ``axis=()`` leaves each value in its original position. Boolean and
    floating-point axes are rejected rather than coerced.

    ``keepdims=True`` retains every reduced axis at length one so the result
    can broadcast against the source. Those dimensions use the highlight
    colour. The default removes reduced axes, preserving surviving colours.
    ``visual.metadata["reduction"]`` records the normalized axes and term count.

    Numerical previews sum input values with Python scalar arithmetic and
    divide by the product of the reduced axis sizes, rather than calling a
    native backend reduction. An empty axis tuple has a divisor of one.
    Accumulation dtype, rounding, and overflow may differ. Shape tuples supply
    generated row-major placeholder values. The model and value source are
    recorded in ``visual.metadata["numeric_semantics"]`` even when evaluation
    is skipped.

    Pass ``focus`` as an output coordinate tuple to explain another group.
    Negative coordinates are supported and a scalar result uses ``()``.
    ``visual.trace`` retains up to eight terms in source row-major order and
    the mean's divisor. Axis tuple order does not affect the trace. Retained
    axes need a zero in the focus coordinate.

    ``max_terms`` limits source terms per output cell, defaulting to 10,000.
    ``max_total_terms`` caps the sum of terms across visible outputs at 100,000.
    Exceeding either limit shows ``?`` for all outputs without partial means.
    Set both limits to ``None`` to remove them. The selection stays compact,
    and ``visual.metadata["value_evaluation"]`` records the evaluation decision.
    """
    return _reduce(
        array, axis, "mean", theme, precision, renderer, keepdims, focus, max_terms, max_total_terms
    )
