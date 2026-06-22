"""Reductions and math views.

Public functions ``matmul``, ``sum``, and ``mean``. Matmul draws both operands
and the output and highlights the contraction; the reductions mark the elements
that fold into each result group.
"""

import builtins

from ..explanations import t
from ..ops import (
    matmul_result_shape,
    matmul_source_terms,
    reduce_result_shape,
    reduce_source_coords,
)
from ..renderers import resolve_renderer
from ..shape import coordinates, extract_shape, flat_index, format_shape
from ..theme import resolve_theme
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


def matmul(a, b, theme=None, precision=2, renderer=None):
    """Visualise a matrix multiplication ``a @ b``.

    ``a`` and ``b`` are array-likes or shape tuples. Vector, matrix, and batched
    matrix multiplication are all supported, matching ``numpy.matmul``. The two
    operands and the output are drawn side by side. The shared inner axis is
    marked in the accent colour, and the row of the first operand and the column
    of the second operand that combine into the first output element are
    highlighted, so the contraction reads directly off the figure.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    a_shape = extract_shape(a)
    b_shape = extract_shape(b)
    result = matmul_result_shape(a_shape, b_shape)
    display_result = result or (1,)
    a_value = _source_value(a, a_shape)
    b_value = _source_value(b, b_shape)

    def result_value(coord):
        out_coord = () if not result else coord
        terms = matmul_source_terms(out_coord, a_shape, b_shape)
        return _py_sum(a_value(ac) * b_value(bc) for ac, bc in terms)

    first = (0,) * len(result)
    terms0 = matmul_source_terms(first, a_shape, b_shape)
    a_selected = sorted({ac for ac, _ in terms0})
    b_selected = sorted({bc for _, bc in terms0})
    result_selected = [(0,) * len(display_result)]

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

    inner = a_shape[a_inner]
    rows = "n/a" if len(a_shape) == 1 else a_shape[-2]
    cols = "n/a" if len(b_shape) == 1 else b_shape[-1]
    explanation = [
        t("matmul.operands", a=format_shape(a_shape), b=format_shape(b_shape)),
        t("matmul.dims", rows=rows, cols=cols, inner=inner),
        t("common.result_shape", shape=format_shape(result)),
        t("matmul.combine"),
    ] + _preview_explanation([a_shape, b_shape, display_result], theme)
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
    return _visual(
        content, a_shape, renderer, selected=a_selected, result=result, explanation=explanation
    )


def _reduce(array, axis, op_name, combine, theme, precision, renderer):
    """Shared body for axis reductions such as sum and mean."""
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shape = extract_shape(array)
    result = reduce_result_shape(shape, axis)
    source_value = _source_value(array, shape)
    axis = axis + len(shape) if axis < 0 else axis

    values = {}
    for rc in coordinates(result):
        contributing = list(reduce_source_coords(rc, shape, axis))
        values[rc] = combine([source_value(sc) for sc in contributing])

    # Mark the source elements that collapse into the first result element, and
    # tint every other group so values that fold into the same result share one
    # background while the first group stays clearly highlighted.
    first = next(iter(coordinates(result)))
    selected = list(reduce_source_coords(first, shape, axis))

    def source_tint(coord):
        rc = coord[:axis] + coord[axis + 1:]
        group = flat_index(rc, result) if result else 0
        if group == 0:
            return None  # the first group is shown through the selected highlight
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
        if group == 0:
            return None
        return _operand_tint(theme, group - 1)

    selected_result = [next(iter(coordinates(disp)))]

    def result_value(coord):
        return values[() if not result else coord]

    explanation = [
        t("common.original_shape", shape=format_shape(shape)),
        t("reduce.reducing", axis=axis, op=op_name),
        t("common.result_shape", shape=format_shape(result)),
        t("reduce.combines", count=shape[axis], axis=axis),
        t("reduce.share_background"),
    ] + _preview_explanation([shape, disp], theme)
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
    return _visual(
        content, shape, renderer, selected=selected, result=result, explanation=explanation
    )


def sum(array, axis, theme=None, precision=2, renderer=None):
    """Visualise a sum reduction over ``axis``.

    The source panel marks the elements that collapse into the first result
    element, and the result panel holds the per group sums with the reduced
    axis gone.
    """
    return _reduce(array, axis, "sum", _py_sum, theme, precision, renderer)


def mean(array, axis, theme=None, precision=2, renderer=None):
    """Visualise a mean reduction over ``axis``.

    The source panel marks the elements that collapse into the first result
    element, and the result panel holds the per group means with the reduced
    axis gone.
    """
    return _reduce(
        array, axis, "mean", lambda vs: _py_sum(vs) / len(vs), theme, precision, renderer
    )
