"""Combining tensors views.

Public functions ``repeat``, ``take``, ``concatenate``, ``stack``, and
``broadcast``. Each tints the source elements so the copies, gathers, seams, or
stretches that produce the result stay visible.
"""

from ..explanations import t
from ..ops import (
    broadcast_result_shape,
    broadcast_source_coord,
    broadcast_stretched_axes,
    concatenate_result_shape,
    concatenate_source,
    repeat_result_shape,
    repeat_source_coord,
    repeat_source_positions,
    stack_result_shape,
    stack_source,
    take_axis_and_indices,
    take_result_shape,
    take_source_coord,
)
from ..renderers import resolve_renderer
from ..shape import extract_shape, format_shape
from ..theme import resolve_theme
from ..visual import (
    _operand_tint,
    _preview_explanation,
    _shape_caption_parts,
    _source_value,
    _value_fn_for,
    _visual,
)


def repeat(array, repeats, axis=0, theme=None, precision=2, renderer=None):
    """Visualise repeating elements along one axis with ``repeat``.

    ``repeats`` is a single count applied to every element, or one count per
    element along ``axis``, matching ``numpy.repeat``. Each source element and
    the adjacent run of copies it produces share one tint, so the result reads as
    materialised copies. This is the contrast with ``broadcast``, which stretches
    a size one axis virtually without copying any values.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shape = extract_shape(array)
    axis = axis + len(shape) if axis < 0 else axis
    source_positions = repeat_source_positions(shape[axis], repeats)
    result = repeat_result_shape(shape, repeats, axis)
    source_value = _source_value(array, shape)

    def result_value(coord):
        return source_value(repeat_source_coord(coord, source_positions, axis))

    def source_tint(coord):
        return _operand_tint(theme, coord[axis])

    def result_tint(coord):
        return _operand_tint(theme, source_positions[coord[axis]])

    explanation = [
        t("common.original_shape", shape=format_shape(shape)),
        t("repeat.repeating", repeats=repeats, axis=axis),
        t("common.result_shape", shape=format_shape(result)),
        t("repeat.copies"),
    ] + _preview_explanation([shape, result], theme)
    panels = [
        {
            "shape": shape,
            "value_fn": _value_fn_for(array),
            "cell_tint": source_tint,
            "caption_parts": _shape_caption_parts("source", shape, theme),
        },
        {
            "shape": result,
            "value_fn": result_value,
            "cell_tint": result_tint,
            "caption_parts": _shape_caption_parts("repeat", result, theme),
        },
    ]
    content = renderer.render_panels(
        panels=panels, connectors=["->"], explanation=explanation, theme=theme, precision=precision
    )
    return _visual(content, shape, renderer, result=result, explanation=explanation)


def take(array, indices, axis=0, theme=None, precision=2, renderer=None):
    """Visualise gathering values along one axis with ``take``.

    ``indices`` is a 1-D list of positions along ``axis``. The chosen axis is
    replaced by the gathered positions, matching ``numpy.take`` with an axis.
    Negative axes and negative indices both work. Each gathered source slice and
    the result slice it feeds share one tint, so repeated indices repeat a tint
    and reordered indices reorder them, making the gather visible.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shape = extract_shape(array)
    axis, resolved = take_axis_and_indices(shape, indices, axis)
    result = take_result_shape(shape, indices, axis)
    source_value = _source_value(array, shape)
    gathered = set(resolved)

    def result_value(coord):
        return source_value(take_source_coord(coord, resolved, axis))

    def source_tint(coord):
        pos = coord[axis]
        if pos in gathered:
            return _operand_tint(theme, pos)
        return None

    def result_tint(coord):
        return _operand_tint(theme, resolved[coord[axis]])

    explanation = [
        t("common.original_shape", shape=format_shape(shape)),
        t("take.taking", indices=list(indices), axis=axis),
        t("common.result_shape", shape=format_shape(result)),
        t("take.copies"),
    ] + _preview_explanation([shape, result], theme)
    panels = [
        {
            "shape": shape,
            "value_fn": _value_fn_for(array),
            "cell_tint": source_tint,
            "caption_parts": _shape_caption_parts("source", shape, theme),
        },
        {
            "shape": result,
            "value_fn": result_value,
            "cell_tint": result_tint,
            "caption_parts": _shape_caption_parts("take", result, theme),
        },
    ]
    content = renderer.render_panels(
        panels=panels, connectors=["->"], explanation=explanation, theme=theme, precision=precision
    )
    return _visual(content, shape, renderer, result=result, explanation=explanation)


def _combine(
    arrays, shapes, result, origin_fn, name, explanation, theme, precision, renderer
):
    """Render several operands and their combined result in one figure.

    ``origin_fn`` maps a result coordinate to ``(operand_index, source_coord)``,
    which drives both the result values and the per cell tint. Each operand is
    tinted, and the result colours every cell by the operand it came from, so
    the seam between operands stays clear.
    """
    renderer = resolve_renderer(renderer)
    source_values = [_source_value(a, s) for a, s in zip(arrays, shapes)]

    def result_value(coord):
        i, sc = origin_fn(coord)
        return source_values[i](sc)

    def result_tint(coord):
        i, _ = origin_fn(coord)
        return _operand_tint(theme, i)

    panels = []
    for i, (a, s) in enumerate(zip(arrays, shapes)):
        fill, border = _operand_tint(theme, i)
        panels.append(
            {
                "shape": s,
                "value_fn": _value_fn_for(a),
                "theme": theme.variant(surface=fill, cell_border=border),
                "caption_parts": _shape_caption_parts(f"operand {i}", s, theme),
            }
        )
    panels.append(
        {
            "shape": result,
            "value_fn": result_value,
            "cell_tint": result_tint,
            "caption_parts": _shape_caption_parts(name, result, theme),
        }
    )
    connectors = ["+"] * (len(arrays) - 1) + ["->"]
    explanation = explanation + _preview_explanation([*shapes, result], theme)
    content = renderer.render_panels(
        panels=panels,
        connectors=connectors,
        explanation=explanation,
        theme=theme,
        precision=precision,
    )
    return _visual(content, shapes[0], renderer, result=result, explanation=explanation)


def concatenate(arrays, axis=0, theme=None, precision=2, renderer=None):
    """Visualise concatenating several tensors along an existing axis.

    ``arrays`` is a sequence of array-likes or shape tuples. They must match on
    every axis except ``axis``. Each operand is tinted and the result colours
    each cell by its operand, so the seam along the joined axis is clear.
    """
    theme = resolve_theme(theme)
    shapes = [extract_shape(a) for a in arrays]
    result = concatenate_result_shape(shapes, axis)
    axis_r = axis + len(shapes[0]) if axis < 0 else axis

    def origin_fn(coord):
        return concatenate_source(coord, shapes, axis_r)

    sizes = ", ".join(format_shape(s) for s in shapes)
    explanation = [
        t("concatenate.operands", shapes=sizes),
        t("concatenate.joining", axis=axis_r),
        t("common.result_shape", shape=format_shape(result)),
        t("concatenate.seam"),
    ]
    return _combine(
        arrays, shapes, result, origin_fn, "concatenate", explanation, theme, precision, renderer
    )


def stack(arrays, axis=0, theme=None, precision=2, renderer=None):
    """Visualise stacking several tensors onto a brand new axis.

    ``arrays`` is a sequence of array-likes or shape tuples that all share the
    same shape. The new axis of length ``len(arrays)`` is inserted at ``axis``,
    and each operand is tinted so the added dimension is visible.
    """
    theme = resolve_theme(theme)
    shapes = [extract_shape(a) for a in arrays]
    result = stack_result_shape(shapes, axis)
    ndim = len(shapes[0])
    axis_r = axis + ndim + 1 if axis < 0 else axis

    def origin_fn(coord):
        return stack_source(coord, axis_r, ndim)

    explanation = [
        t("stack.operand", shape=format_shape(shapes[0])),
        t("stack.stacking", count=len(arrays), axis=axis_r),
        t("common.result_shape", shape=format_shape(result)),
        t("stack.new_axis"),
    ]
    return _combine(
        arrays, shapes, result, origin_fn, "stack", explanation, theme, precision, renderer
    )


def broadcast(a, b, theme=None, precision=2, renderer=None):
    """Visualise broadcasting two tensors to a common shape.

    Each operand is drawn in its own shape and again stretched to the broadcast
    shape, so the repeated values along a stretched axis are visible. Every
    stretched axis is marked in the accent colour in the stretched caption.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    arrays = [a, b]
    shapes = [extract_shape(x) for x in arrays]
    result = broadcast_result_shape(shapes)

    panels = []
    for i, (arr, s) in enumerate(zip(arrays, shapes)):
        source_value = _source_value(arr, s)
        stretched = set(broadcast_stretched_axes(s, result))

        def stretched_value(coord, _sv=source_value, _s=s):
            return _sv(broadcast_source_coord(coord, _s))

        def color_for(axis, _stretched=stretched):
            if axis in _stretched:
                return theme.surface_selected
            return theme.axis_color(axis) if axis < len(result) - 1 else theme.text_muted

        panels.append(
            {
                "shape": s,
                "value_fn": _value_fn_for(arr),
                "caption_parts": _shape_caption_parts(f"operand {i}", s, theme),
            }
        )
        panels.append(
            {
                "shape": result,
                "value_fn": stretched_value,
                "caption_parts": _shape_caption_parts(
                    "stretched", result, theme, color_for=color_for
                ),
            }
        )

    def axes_line(i):
        ax = broadcast_stretched_axes(shapes[i], result)
        if ax:
            return t("broadcast.stretches", i=i, shape=format_shape(shapes[i]), axes=ax)
        return t("broadcast.matches", i=i, shape=format_shape(shapes[i]))

    explanation = [
        t("broadcast.operands", a=format_shape(shapes[0]), b=format_shape(shapes[1])),
        axes_line(0),
        axes_line(1),
        t("common.result_shape", shape=format_shape(result)),
        t("broadcast.fill"),
    ] + _preview_explanation([*shapes, result], theme)
    content = renderer.render_panels(
        panels=panels,
        connectors=["->", "&", "->"],
        explanation=explanation,
        theme=theme,
        precision=precision,
    )
    return _visual(content, shapes[0], renderer, result=result, explanation=explanation)
