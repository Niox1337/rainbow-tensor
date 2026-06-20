"""Notebook display layer.

This module exposes the public functions ``shape`` and ``index``.
Each returns a small object that renders as SVG in a notebook and stays
inspectable in plain Python, so the package is testable outside a notebook.
"""

import builtins
from itertools import product
from math import prod

from .backends import value_at_coordinate
from .indexing import (
    advanced_index,
    explain_index,
    format_index,
    format_token,
    is_advanced,
    result_shape,
    selected_coordinates,
    validate_index,
)
from .layout import axis_visible_limits
from .ops import (
    broadcast_result_shape,
    broadcast_source_coord,
    broadcast_stretched_axes,
    concatenate_result_shape,
    concatenate_source,
    einsum_contracted_labels,
    einsum_index_sizes,
    einsum_result_shape,
    einsum_selected_coords,
    expand_dims_axes,
    expand_dims_result_shape,
    expand_dims_source_coord,
    matmul_result_shape,
    matmul_source_terms,
    parse_einsum_subscripts,
    reduce_result_shape,
    reduce_source_coords,
    reshape_result_shape,
    reshape_source_coord,
    squeeze_axes,
    squeeze_result_shape,
    squeeze_source_coord,
    stack_result_shape,
    stack_source,
    swapaxes_axes,
    take_axis_and_indices,
    take_result_shape,
    take_source_coord,
    transpose_axes,
    transpose_result_shape,
    transpose_source_coord,
)
from .renderers import resolve_renderer
from .shape import coordinates, extract_shape, flat_index, format_shape
from .theme import resolve_theme

# The public ``sum`` below shadows the builtin in this module, so keep a handle
# to the real one for the reduction maths.
_py_sum = builtins.sum


class TensorVisual:
    """A rendered tensor visualisation.

    The ``svg`` attribute holds the SVG string, and ``text`` holds the plain
    explanation. In a notebook the image is displayed first and the explanation
    is printed as standard output underneath it. Outside a notebook both values
    stay available for inspection and testing.
    """

    def __init__(
        self,
        svg,
        shape,
        selected=None,
        result=None,
        explanation=None,
        renderer="svg",
        mime_type="image/svg+xml",
    ):
        self.svg = svg
        self.content = svg
        self.shape = shape
        self.selected = selected
        self.result_shape = result
        self.explanation = explanation or []
        self.text = "\n".join(self.explanation)
        self.renderer = renderer
        self.mime_type = mime_type

    def _repr_svg_(self):
        if self.mime_type == "image/svg+xml":
            return self.svg
        return None

    def __str__(self):
        return self.svg

    def __repr__(self):
        # Stable text repr so a re-run notebook's text/plain output never churns
        # on the object's memory address.
        return f"TensorVisual(shape={self.shape}, result_shape={self.result_shape})"

    def _ipython_display_(self):
        """Display the figure and print explanation text as standard output."""
        from IPython.display import SVG, display

        if self.mime_type == "image/svg+xml":
            display(SVG(self.svg))
        else:
            display({self.mime_type: self.content}, raw=True)
        if self.text:
            print(self.text)

    def save(self, path):
        """Write the SVG to ``path`` and return the path.

        The text is written as UTF-8 so the ellipsis and legend glyphs survive
        the round trip. ``path`` may be a string or a path-like object.
        """
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.svg)
        return path


def _visual(content, shape, renderer, selected=None, result=None, explanation=None):
    """Build the visual result object for a renderer output."""
    return TensorVisual(
        content,
        shape,
        selected=selected,
        result=result,
        explanation=explanation,
        renderer=getattr(renderer, "name", "custom"),
        mime_type=getattr(renderer, "mime_type", "image/svg+xml"),
    )


def _preview_explanation(shapes, theme):
    """Return standard output lines for large tensor previews."""
    lines = []
    for i, shape in enumerate(shapes):
        limits = axis_visible_limits(shape, theme.max_cells, theme.max_visible_cells)
        hidden = [axis for axis, (size, limit) in enumerate(zip(shape, limits)) if size > limit]
        if not hidden:
            continue
        name = "tensor" if len(shapes) == 1 else f"tensor {i}"
        lines.append(
            f"Large preview for {name} {format_shape(shape)} draws at most "
            f"{theme.max_visible_cells} real cells from {prod(shape)} total."
        )
        lines.append(f"Hidden axes {hidden} keep the head, selected positions, and tail.")
    return lines


def _value_fn_for(obj):
    """Build a coordinate to value function for array-like input.

    A tuple carries no values, so generated values are used. Any object with
    a ``.shape`` attribute is read by coordinate, which covers NumPy arrays
    and any compatible array-like without importing a tensor library.
    """
    if isinstance(obj, tuple):
        return None
    if not hasattr(obj, "shape"):
        return None

    def value_fn(coord):
        return value_at_coordinate(obj, coord)

    return value_fn


def _shape_label_parts(shape, theme):
    """Build coloured label parts for a shape.

    Each dimension is coloured to match its frame. Frame axes use their axis
    colour and the leaf axis stays neutral, since it has no frame.
    """
    ndim = len(shape)
    neutral = theme.heading
    parts = [("Shape (", neutral)]
    for axis, dim in enumerate(shape):
        if axis < ndim - 1:
            color = theme.axis_color(axis)
        else:
            color = theme.text_muted
        parts.append((str(dim), color))
        if axis < ndim - 1:
            parts.append((", ", neutral))
    parts.append((")", neutral))
    return parts


def _index_label_parts(index, theme):
    """Build coloured label parts for an index.

    Each token is coloured to match its axis. Frame axes use their axis
    colour and the leaf axis uses the selected value colour, so the label
    mirrors the colours drawn on the tensor.
    """
    ndim = len(index)
    neutral = theme.heading
    parts = [("Index (", neutral)]
    for axis, entry in enumerate(index):
        token = format_token(entry)
        if axis < ndim - 1:
            color = theme.axis_color(axis)
        else:
            color = theme.surface_selected
        parts.append((token, color))
        if axis < ndim - 1:
            parts.append((", ", neutral))
    parts.append((")", neutral))
    return parts


def shape(array, theme=None, precision=2, renderer=None):
    """Visualise the structure of a tensor shape.

    ``array`` may be an array-like object with a ``.shape`` attribute such as
    a NumPy array, in which case its own values are rendered, or a tuple such
    as ``(2, 2, 2)``, in which case placeholder values are generated. ``theme``
    chooses the palette and geometry, accepting a
    :class:`~rainbow_tensor.theme.Theme`, a theme name, or ``None`` for the
    module default. ``precision`` controls how floats are formatted. The
    returned :class:`TensorVisual` renders as SVG in a notebook through
    ``_repr_svg_`` and also exposes the SVG string for inspection and testing.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    normalized = extract_shape(array)
    value_fn = _value_fn_for(array)
    label_parts = _shape_label_parts(normalized, theme)
    explanation = _preview_explanation([normalized], theme)
    content = renderer.render_tensor(
        shape=normalized,
        value_fn=value_fn,
        label_parts=label_parts,
        theme=theme,
        precision=precision,
    )
    return _visual(content, normalized, renderer, explanation=explanation)


def index(array, index, theme=None, precision=2, renderer=None):
    """Visualise how an index selects elements from a tensor.

    ``array`` may be an array-like object with a ``.shape`` attribute, in
    which case its own values are rendered, or a shape tuple. ``index`` is a
    tuple of integers and slices for basic indexing, a full-shape boolean mask,
    or a tuple holding integer index arrays for advanced indexing. Selected
    elements are highlighted, unselected elements stay visible but
    de-emphasised, and an explanation of the result shape is drawn below the
    tensor. ``theme`` and ``precision`` behave as in :func:`shape`. The
    returned :class:`TensorVisual` renders as SVG in a notebook through
    ``_repr_svg_`` and also exposes the SVG string for inspection and testing.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    normalized = extract_shape(array)
    value_fn = _value_fn_for(array)

    if is_advanced(index, normalized):
        selected, result, explanation = advanced_index(normalized, index)
        explanation = explanation + _preview_explanation([normalized], theme)
        label = "mask" if not isinstance(index, tuple) else f"Index ({format_index(index)})"
        content = renderer.render_tensor(
            shape=normalized,
            selected=selected,
            value_fn=value_fn,
            label=label,
            explanation=explanation,
            theme=theme,
            precision=precision,
        )
        return _visual(
            content, normalized, renderer, selected=selected, result=result, explanation=explanation
        )

    validate_index(index, normalized)
    selected = selected_coordinates(normalized, index)
    result = result_shape(normalized, index)
    explanation = explain_index(normalized, index) + _preview_explanation([normalized], theme)
    label_parts = _index_label_parts(index, theme)
    content = renderer.render_tensor(
        shape=normalized,
        selected=selected,
        value_fn=value_fn,
        label_parts=label_parts,
        explanation=explanation,
        theme=theme,
        precision=precision,
    )
    return _visual(
        content, normalized, renderer, selected=selected, result=result, explanation=explanation
    )


def _source_value(array, shape):
    """Return a coordinate to value function for ``array``.

    An array-like is read by coordinate. A bare shape tuple carries no values,
    so the row major flat index stands in, matching :func:`shape`.
    """
    value_fn = _value_fn_for(array)
    if value_fn is None:
        return lambda coord: flat_index(coord, shape)
    return value_fn


def _shape_caption_parts(name, shape, theme, color_for=None):
    """Build a coloured caption naming a panel and its shape.

    ``color_for`` maps an axis to its colour, defaulting to the axis ramp with
    a muted leaf axis. Passing a custom map lets a transpose colour each axis
    by where it came from.
    """
    ndim = len(shape)

    def default_color(axis):
        return theme.axis_color(axis) if axis < ndim - 1 else theme.text_muted

    color_for = color_for or default_color
    neutral = theme.heading
    parts = [(f"{name} (", neutral)]
    for axis, dim in enumerate(shape):
        parts.append((str(dim), color_for(axis)))
        if axis < ndim - 1:
            parts.append((", ", neutral))
    parts.append((")", neutral))
    return parts


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
        f"Original shape: {format_shape(old)}",
        f"New shape: {format_shape(new)}",
        "Values keep their row major order, so element k stays element k.",
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


def transpose(array, axes=None, theme=None, precision=2, renderer=None):
    """Visualise a transpose or permute, with axis colours following the move.

    ``axes`` is a permutation of the source axes, or ``None`` to reverse them
    like ``numpy.transpose``. The result panel colours each axis by the source
    axis it came from, so a colour can be traced across the swap.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shape = extract_shape(array)
    perm = transpose_axes(len(shape), axes)
    result = transpose_result_shape(shape, perm)
    source_value = _source_value(array, shape)

    def result_value(coord):
        return source_value(transpose_source_coord(coord, perm))

    # Result axis r came from source axis perm[r], so it borrows that colour.
    result_theme = theme.variant(
        axis_colors=tuple(theme.axis_color(perm[r]) for r in range(len(perm)))
    )

    def result_color(axis):
        return result_theme.axis_color(axis) if axis < len(result) - 1 else theme.text_muted

    explanation = [
        f"Original shape: {format_shape(shape)}",
        f"Axes order: {perm}",
        f"Result shape: {format_shape(result)}",
        "Each axis keeps its colour as it moves to its new position.",
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
                "transpose", result, theme, color_for=result_color
            ),
        },
    ]
    content = renderer.render_panels(
        panels=panels, connectors=["->"], explanation=explanation, theme=theme, precision=precision
    )
    return _visual(content, shape, renderer, result=result, explanation=explanation)


def swapaxes(array, axis1, axis2, theme=None, precision=2, renderer=None):
    """Visualise swapping two axes of a tensor.

    ``swapaxes`` is the small two axis form of transpose. The source and result
    are drawn side by side, and the two moved axes keep their colours so the
    swap is traceable.
    """
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shape = extract_shape(array)
    perm = swapaxes_axes(len(shape), axis1, axis2)
    result = transpose_result_shape(shape, perm)
    source_value = _source_value(array, shape)
    axis1_r = axis1 + len(shape) if axis1 < 0 else axis1
    axis2_r = axis2 + len(shape) if axis2 < 0 else axis2

    def result_value(coord):
        return source_value(transpose_source_coord(coord, perm))

    result_theme = theme.variant(
        axis_colors=tuple(theme.axis_color(perm[r]) for r in range(len(perm)))
    )

    def result_color(axis):
        return result_theme.axis_color(axis) if axis < len(result) - 1 else theme.text_muted

    explanation = [
        f"Original shape: {format_shape(shape)}",
        f"Swapping axes {axis1_r} and {axis2_r}.",
        f"Axes order: {perm}",
        f"Result shape: {format_shape(result)}",
        "The two moved axes keep their source colours in the result.",
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
                "swapaxes", result, theme, color_for=result_color
            ),
        },
    ]
    content = renderer.render_panels(
        panels=panels, connectors=["->"], explanation=explanation, theme=theme, precision=precision
    )
    return _visual(content, shape, renderer, result=result, explanation=explanation)


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

    removed_line = (
        f"Removing size one axes {list(removed)}." if removed else "No size one axes to remove."
    )
    explanation = [
        f"Original shape: {format_shape(shape)}",
        removed_line,
        f"Result shape: {format_shape(result)}",
        "Each surviving axis keeps its colour, and the removed size one axes are marked.",
    ] + _preview_explanation([shape, display_result], theme)
    panels = [
        {
            "shape": shape,
            "value_fn": _value_fn_for(array),
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
        f"Original shape: {format_shape(shape)}",
        f"Inserting size one axes at {list(inserted)}.",
        f"Result shape: {format_shape(result)}",
        "Each existing axis keeps its colour, and the inserted size one axes are marked.",
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
        f"Operand shapes: {format_shape(a_shape)} @ {format_shape(b_shape)}",
        f"Left rows: {rows}, right columns: {cols}, shared inner axis: {inner}.",
        f"Result shape: {format_shape(result)}",
        "The shared inner axis is marked, and the highlighted row and column "
        "combine into the first output element.",
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
        f"Original shape: {format_shape(shape)}",
        f"Taking indices {list(indices)} along axis {axis}.",
        f"Result shape: {format_shape(result)}",
        "Each result slice copies a source slice, so repeated indices repeat a "
        "slice and reordered indices reorder them.",
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


# Role colour families, chosen so free, shared, and contracted labels read as
# three distinct groups while every label keeps one colour everywhere it appears.
_EINSUM_FREE = ("#2563eb", "#0891b2", "#4f46e5", "#0d9488", "#0284c7")
_EINSUM_SHARED = ("#db2777", "#9333ea", "#c026d3", "#e11d48")
_EINSUM_CONTRACTED = ("#ea580c", "#ca8a04", "#dc2626", "#b45309")


def _einsum_roles(input_axes, output_axes):
    """Classify each einsum label as free, shared, or contracted.

    A label that is not in the output is contracted. Otherwise a label that
    appears in more than one operand is shared, and the rest are free. Each
    label gets exactly one role, so the three roles partition the labels.
    """
    output = set(output_axes)
    operand_count = {}
    for labels in input_axes:
        for label in set(labels):
            operand_count[label] = operand_count.get(label, 0) + 1
    roles = {}
    for labels in (*input_axes, output_axes):
        for label in labels:
            if label in roles:
                continue
            if label not in output:
                roles[label] = "contracted"
            elif operand_count.get(label, 0) >= 2:
                roles[label] = "shared"
            else:
                roles[label] = "free"
    return roles


def _einsum_label_colors(input_axes, output_axes):
    """Map every einsum label to one colour drawn from its role family.

    The colour is stable across every operand and the output, so a label keeps
    one colour in every caption and figure, and the three roles stay visually
    distinct.
    """
    roles = _einsum_roles(input_axes, output_axes)
    ramps = {"free": _EINSUM_FREE, "shared": _EINSUM_SHARED, "contracted": _EINSUM_CONTRACTED}
    counters = {"free": 0, "shared": 0, "contracted": 0}
    colors = {}
    for labels in (*input_axes, output_axes):
        for label in labels:
            if label in colors:
                continue
            role = roles[label]
            colors[label] = ramps[role][counters[role] % len(ramps[role])]
            counters[role] += 1
    return colors


def _einsum_theme(labels, theme, label_colors):
    """Return a theme whose axis ramp follows the subscript labels."""
    if not labels:
        return theme
    return theme.variant(axis_colors=tuple(label_colors[label] for label in labels))


def _einsum_leaf_tint(labels, theme, label_colors):
    """Border a panel's leaf cells with the leaf label colour.

    A non-leaf axis shows its label colour as a frame, but the leaf axis is
    drawn as plain cells, so without this the leaf label colour would appear in
    the caption yet never in the figure. Bordering each cell with the leaf label
    colour keeps the figure and the caption consistent for every label.
    """
    if not labels:
        return None
    border = label_colors[labels[-1]]
    fill = theme.surface
    return lambda coord: (fill, border)


def _einsum_caption_parts(name, labels, shape, theme, label_colors):
    """Build a coloured caption for an einsum panel."""
    neutral = theme.heading
    parts = [(f"{name} ", neutral)]
    if labels:
        for label in labels:
            parts.append((label, label_colors[label]))
    else:
        parts.append(("scalar", theme.text_muted))
    parts.append((" (", neutral))
    for axis, dim in enumerate(shape):
        color = label_colors[labels[axis]] if labels else theme.text_muted
        parts.append((str(dim), color))
        if axis < len(shape) - 1:
            parts.append((", ", neutral))
    parts.append((")", neutral))
    return parts


def _einsum_result_value_fn(input_axes, output_axes, shapes, source_values):
    """Build a value function for the einsum output panel."""
    sizes = einsum_index_sizes(input_axes, shapes)
    contracted = einsum_contracted_labels(input_axes, output_axes)
    ranges = [range(sizes[label]) for label in contracted]

    def result_value(coord):
        output_coord = () if not output_axes else coord
        fixed = dict(zip(output_axes, output_coord))
        total = 0
        for values in product(*ranges):
            assignment = dict(fixed)
            assignment.update(zip(contracted, values))
            term = 1
            for labels, value_fn in zip(input_axes, source_values):
                source_coord = tuple(assignment[label] for label in labels)
                term *= value_fn(source_coord)
            total += term
        return total

    return result_value


def einsum(subscripts, *arrays, theme=None, precision=2, renderer=None):
    """Visualise an einsum expression.

    Each operand is drawn with its subscript labels. Every label keeps one
    colour across all operand captions, operand figures, and the output, drawn
    from a colour family for its role, so free, shared, and contracted labels
    stay visually distinct. The output panel shows the derived free index shape.
    """
    if not arrays:
        raise ValueError("einsum needs at least one operand")

    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shapes = [extract_shape(array) for array in arrays]
    input_axes, output_axes = parse_einsum_subscripts(subscripts, len(arrays), shapes)
    result = einsum_result_shape(subscripts, shapes)
    display_result = result or (1,)
    source_values = [_source_value(array, shape) for array, shape in zip(arrays, shapes)]
    label_colors = _einsum_label_colors(input_axes, output_axes)
    selected = einsum_selected_coords(input_axes, output_axes, shapes)
    result_value = _einsum_result_value_fn(input_axes, output_axes, shapes, source_values)
    contracted = einsum_contracted_labels(input_axes, output_axes)

    panels = []
    for i, (array, labels, shape) in enumerate(zip(arrays, input_axes, shapes)):
        panels.append(
            {
                "shape": shape,
                "value_fn": _value_fn_for(array),
                "theme": _einsum_theme(labels, theme, label_colors),
                "cell_tint": _einsum_leaf_tint(labels, theme, label_colors),
                "caption_parts": _einsum_caption_parts(
                    f"operand {i}", labels, shape, theme, label_colors
                ),
            }
        )

    panels.append(
        {
            "shape": display_result,
            "value_fn": result_value,
            "theme": _einsum_theme(output_axes, theme, label_colors),
            "cell_tint": _einsum_leaf_tint(output_axes, theme, label_colors),
            "caption_parts": _einsum_caption_parts(
                "output", output_axes, display_result, theme, label_colors
            ),
        }
    )
    connectors = ["*"] * (len(arrays) - 1) + ["->"]
    input_text = ", ".join("".join(labels) or "scalar" for labels in input_axes)
    output_text = "".join(output_axes) or "scalar"
    if contracted:
        contracted_line = f"Contracted labels: {', '.join(contracted)}"
    else:
        contracted_line = "No labels are contracted."
    explanation = [
        f"Operand subscripts: {input_text}",
        f"Output subscript: {output_text}",
        contracted_line,
        f"Result shape: {format_shape(result)}",
    ] + _preview_explanation([*shapes, display_result], theme)
    content = renderer.render_panels(
        panels=panels,
        connectors=connectors,
        explanation=explanation,
        theme=theme,
        precision=precision,
    )
    return _visual(
        content, shapes[0], renderer, selected=selected, result=result, explanation=explanation
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
        f"Original shape: {format_shape(shape)}",
        f"Reducing axis {axis} with {op_name}.",
        f"Result shape: {format_shape(result)}",
        f"Each result element combines {shape[axis]} values from axis {axis}.",
        "Values that fold into the same result share one background with that "
        "result element, and the first group is highlighted.",
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
            "caption_parts": _shape_caption_parts(op_name, disp, theme),
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


# Soft operand tints, keyed by operand index, so a combined result can colour
# each cell by where it came from. The first entry is the fill, the second the
# matching border.
_LIGHT_TINTS = [
    ("#dbeafe", "#93c5fd"),  # blue
    ("#fef3c7", "#fcd34d"),  # amber
    ("#dcfce7", "#86efac"),  # green
    ("#fce7f3", "#f9a8d4"),  # pink
    ("#ede9fe", "#c4b5fd"),  # violet
]
_DARK_TINTS = [
    ("#1e3a5f", "#3b82f6"),
    ("#3f2d10", "#d97706"),
    ("#14352a", "#22c55e"),
    ("#3b1d2e", "#db2777"),
    ("#2a2150", "#7c3aed"),
]


def _operand_tint(theme, i):
    """Return the ``(fill, border)`` tint for operand ``i`` under ``theme``."""
    ramp = _DARK_TINTS if theme.name == "dark" else _LIGHT_TINTS
    return ramp[i % len(ramp)]


def _combine(arrays, shapes, result, origin_fn, name, explanation, theme, precision, renderer):
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
        f"Operand shapes: {sizes}",
        f"Joining along axis {axis_r}.",
        f"Result shape: {format_shape(result)}",
        "Each operand keeps its tint, so the seam along the joined axis is clear.",
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
        f"Operand shape: {format_shape(shapes[0])}",
        f"Stacking {len(arrays)} tensors on a new axis {axis_r}.",
        f"Result shape: {format_shape(result)}",
        "The new axis indexes the operands, each kept in its own tint.",
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
            return f"operand {i} {format_shape(shapes[i])} stretches axes {ax}"
        return f"operand {i} {format_shape(shapes[i])} already matches"

    explanation = [
        f"Operand shapes: {format_shape(shapes[0])} and {format_shape(shapes[1])}",
        axes_line(0),
        axes_line(1),
        f"Result shape: {format_shape(result)}",
        "A stretched axis repeats one value, so both operands fill the result shape.",
    ] + _preview_explanation([*shapes, result], theme)
    content = renderer.render_panels(
        panels=panels,
        connectors=["->", "&", "->"],
        explanation=explanation,
        theme=theme,
        precision=precision,
    )
    return _visual(content, shapes[0], renderer, result=result, explanation=explanation)
