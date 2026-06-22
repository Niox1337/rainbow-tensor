"""Reshaping and moving axes views.

Public functions ``reshape``, ``transpose``, ``swapaxes``, ``moveaxis``,
``squeeze``, and ``expand_dims``. Each draws the source and result side by side
and traces axis colours across the move.
"""

from ..explanations import t
from ..ops import (
    expand_dims_axes,
    expand_dims_result_shape,
    expand_dims_source_coord,
    moveaxis_axes,
    reshape_result_shape,
    reshape_source_coord,
    squeeze_axes,
    squeeze_result_shape,
    squeeze_source_coord,
    swapaxes_axes,
    transpose_axes,
    transpose_result_shape,
    transpose_source_coord,
)
from ..renderers import resolve_renderer
from ..shape import extract_shape, format_shape
from ..theme import resolve_theme
from ..visual import (
    _preview_explanation,
    _shape_caption_parts,
    _source_value,
    _value_fn_for,
    _visual,
)


def reshape(array, new_shape, theme=None, precision=2, renderer=None):
    """Visualise a reshape from the source layout into a new one.

    ``array`` is an array-like with a ``.shape`` attribute or a shape tuple.
    ``new_shape`` is the target shape and may use a single ``-1`` to infer one
    axis. The source and the result are drawn side by side. Reshape keeps the
    row major order, so the element at a flat position stays at that flat
    position, which the figure makes visible by carrying the same values into
    the new layout.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    old = extract_shape(array)
    new = reshape_result_shape(old, new_shape)
    source_value = _source_value(array, old)

    def result_value(coord):
        return source_value(reshape_source_coord(coord, old, new))

    explanation = [
        t("common.original_shape", shape=format_shape(old)),
        t("common.new_shape", shape=format_shape(new)),
        t("reshape.row_major"),
    ] + _preview_explanation([old, new], theme)
    panels = [
        {
            "shape": old,
            "value_fn": _value_fn_for(array),
            "caption_parts": _shape_caption_parts("source", old, theme),
        },
        {
            "shape": new,
            "value_fn": result_value,
            "caption_parts": _shape_caption_parts("reshape", new, theme),
        },
    ]
    content = renderer.render_panels(
        panels=panels, connectors=["->"], explanation=explanation, theme=theme, precision=precision
    )
    return _visual(content, old, renderer, result=new, explanation=explanation)


def _permute_view(array, perm, name, prefix_lines, keep_colour, theme, precision, renderer):
    """Shared body for transpose, swapaxes, and moveaxis.

    ``perm`` is the resolved axis order. The result panel colours each axis by
    the source axis ``perm[r]`` it came from, so a colour traces across the
    move. ``prefix_lines`` are the op specific explanation lines inserted after
    the original shape, and ``keep_colour`` is the closing colour note.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shape = extract_shape(array)
    result = transpose_result_shape(shape, perm)
    source_value = _source_value(array, shape)

    def result_value(coord):
        return source_value(transpose_source_coord(coord, perm))

    result_theme = theme.variant(
        axis_colors=tuple(theme.axis_color(perm[r]) for r in range(len(perm)))
    )

    def result_color(axis):
        return result_theme.axis_color(axis) if axis < len(result) - 1 else theme.text_muted

    explanation = [
        t("common.original_shape", shape=format_shape(shape)),
        *prefix_lines,
        t("common.axes_order", order=perm),
        t("common.result_shape", shape=format_shape(result)),
        keep_colour,
    ] + _preview_explanation([shape, result], theme)
    panels = [
        {
            "shape": shape,
            "value_fn": _value_fn_for(array),
            "caption_parts": _shape_caption_parts("source", shape, theme),
        },
        {
            "shape": result,
            "value_fn": result_value,
            "theme": result_theme,
            "caption_parts": _shape_caption_parts(name, result, theme, color_for=result_color),
        },
    ]
    content = renderer.render_panels(
        panels=panels, connectors=["->"], explanation=explanation, theme=theme, precision=precision
    )
    return _visual(content, shape, renderer, result=result, explanation=explanation)


def transpose(array, axes=None, theme=None, precision=2, renderer=None):
    """Visualise a transpose or permute, with axis colours following the move.

    ``axes`` is a permutation of the source axes, or ``None`` to reverse them
    like ``numpy.transpose``. The result panel colours each axis by the source
    axis it came from, so a colour can be traced across the swap.
    """
    perm = transpose_axes(len(extract_shape(array)), axes)
    return _permute_view(
        array, perm, "transpose", [], t("transpose.keep_colour"), theme, precision, renderer
    )


def swapaxes(array, axis1, axis2, theme=None, precision=2, renderer=None):
    """Visualise swapping two axes of a tensor.

    ``swapaxes`` is the small two axis form of transpose. The source and result
    are drawn side by side, and the two moved axes keep their colours so the
    swap is traceable.
    """
    ndim = len(extract_shape(array))
    perm = swapaxes_axes(ndim, axis1, axis2)
    axis1_r = axis1 + ndim if axis1 < 0 else axis1
    axis2_r = axis2 + ndim if axis2 < 0 else axis2
    return _permute_view(
        array,
        perm,
        "swapaxes",
        [t("swapaxes.swap", a=axis1_r, b=axis2_r)],
        t("swapaxes.keep_colour"),
        theme,
        precision,
        renderer,
    )


def moveaxis(array, source, destination, theme=None, precision=2, renderer=None):
    """Visualise moving one or more axes to new positions.

    ``source`` and ``destination`` are an axis or a sequence of axes of equal
    length, matching ``numpy.moveaxis``. ``moveaxis`` sits between ``transpose``
    and ``swapaxes``: the moved axes go to their destinations and the rest keep
    their order. The source and result are drawn side by side, and each result
    axis keeps the colour of the source axis it came from, so the move is
    traceable.
    """
    order = moveaxis_axes(len(extract_shape(array)), source, destination)
    return _permute_view(
        array,
        order,
        "moveaxis",
        [t("moveaxis.move", source=source, destination=destination)],
        t("moveaxis.keep_colour"),
        theme,
        precision,
        renderer,
    )


def squeeze(array, axis=None, theme=None, precision=2, renderer=None):
    """Visualise squeezing size one axes out of a tensor.

    ``axis`` is ``None`` to remove every size one axis, or an axis or tuple of
    axes to remove only those, which must each have size one, matching
    ``numpy.squeeze``. The source panel marks the removed size one axes in the
    accent colour, and each surviving axis keeps its source colour in the result.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shape = extract_shape(array)
    removed = squeeze_axes(shape, axis)
    removed_set = set(removed)
    result = squeeze_result_shape(shape, axis)
    source_value = _source_value(array, shape)
    surviving = [i for i in range(len(shape)) if i not in removed_set]
    display_result = result or (1,)

    def result_value(coord):
        sc = (0,) * len(shape) if not result else squeeze_source_coord(coord, removed)
        return source_value(sc)

    if result:
        result_theme = theme.variant(
            axis_colors=tuple(theme.axis_color(surviving[r]) for r in range(len(result)))
        )
    else:
        result_theme = theme

    def result_color(axis):
        return result_theme.axis_color(axis) if axis < len(result) - 1 else theme.text_muted

    def source_color(axis):
        if axis in removed_set:
            return theme.surface_selected
        return theme.axis_color(axis) if axis < len(shape) - 1 else theme.text_muted

    # The source frames mark the removed axes in the accent colour too, so a
    # removed axis frame and its caption number read as one, matching how
    # expand_dims marks its inserted axes.
    source_theme = theme.variant(
        axis_colors=tuple(
            theme.surface_selected if a in removed_set else theme.axis_color(a)
            for a in range(len(shape))
        )
    )

    removed_line = (
        t("squeeze.removing", axes=list(removed)) if removed else t("squeeze.none")
    )
    explanation = [
        t("common.original_shape", shape=format_shape(shape)),
        removed_line,
        t("common.result_shape", shape=format_shape(result)),
        t("squeeze.keep_colour"),
    ] + _preview_explanation([shape, display_result], theme)
    panels = [
        {
            "shape": shape,
            "value_fn": _value_fn_for(array),
            "theme": source_theme,
            "caption_parts": _shape_caption_parts("source", shape, theme, color_for=source_color),
        },
        {
            "shape": display_result,
            "value_fn": result_value,
            "theme": result_theme,
            "caption_parts": _shape_caption_parts(
                "squeeze", display_result, theme, color_for=result_color
            ),
        },
    ]
    content = renderer.render_panels(
        panels=panels, connectors=["->"], explanation=explanation, theme=theme, precision=precision
    )
    return _visual(content, shape, renderer, result=result, explanation=explanation)


def expand_dims(array, axis, theme=None, precision=2, renderer=None):
    """Visualise inserting size one axes into a tensor.

    ``axis`` is a position or tuple of positions in the result where a new size
    one axis is inserted, with negative positions counting from the end, matching
    ``numpy.expand_dims``. Each existing axis keeps its source colour, and the
    inserted size one axes are marked in the accent colour.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shape = extract_shape(array)
    inserted = expand_dims_axes(len(shape), axis)
    inserted_set = set(inserted)
    result = expand_dims_result_shape(shape, axis)
    source_value = _source_value(array, shape)

    def result_value(coord):
        return source_value(expand_dims_source_coord(coord, inserted))

    source_axis = {}
    src = iter(range(len(shape)))
    for i in range(len(result)):
        if i not in inserted_set:
            source_axis[i] = next(src)
    result_theme = theme.variant(
        axis_colors=tuple(
            theme.surface_selected if i in inserted_set else theme.axis_color(source_axis[i])
            for i in range(len(result))
        )
    )

    def result_color(axis):
        if axis in inserted_set:
            return theme.surface_selected
        return result_theme.axis_color(axis) if axis < len(result) - 1 else theme.text_muted

    explanation = [
        t("common.original_shape", shape=format_shape(shape)),
        t("expand_dims.inserting", axes=list(inserted)),
        t("common.result_shape", shape=format_shape(result)),
        t("expand_dims.keep_colour"),
    ] + _preview_explanation([shape, result], theme)
    panels = [
        {
            "shape": shape,
            "value_fn": _value_fn_for(array),
            "caption_parts": _shape_caption_parts("source", shape, theme),
        },
        {
            "shape": result,
            "value_fn": result_value,
            "theme": result_theme,
            "caption_parts": _shape_caption_parts(
                "expand_dims", result, theme, color_for=result_color
            ),
        },
    ]
    content = renderer.render_panels(
        panels=panels, connectors=["->"], explanation=explanation, theme=theme, precision=precision
    )
    return _visual(content, shape, renderer, result=result, explanation=explanation)
