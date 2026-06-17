"""Notebook display layer.

This module exposes the public functions ``shape`` and ``index``.
Each returns a small object that renders as SVG in a notebook and stays
inspectable in plain Python, so the package is testable outside a notebook.
"""

import builtins

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
from .ops import (
    concatenate_result_shape,
    concatenate_source,
    reduce_result_shape,
    reduce_source_coords,
    reshape_result_shape,
    reshape_source_coord,
    stack_result_shape,
    stack_source,
    transpose_axes,
    transpose_result_shape,
    transpose_source_coord,
)
from .render_svg import render_panels, render_svg
from .shape import coordinates, extract_shape, flat_index, format_shape
from .theme import resolve_theme

# The public ``sum`` below shadows the builtin in this module, so keep a handle
# to the real one for the reduction maths.
_py_sum = builtins.sum


class TensorVisual:
    """A rendered tensor visualisation.

    The ``svg`` attribute holds the SVG string. In a notebook the object is
    shown as an image through ``_repr_svg_``. Outside a notebook the same
    string is available for inspection and testing.
    """

    def __init__(self, svg, shape, selected=None, result=None, explanation=None):
        self.svg = svg
        self.shape = shape
        self.selected = selected
        self.result_shape = result
        self.explanation = explanation or []

    def _repr_svg_(self):
        return self.svg

    def __str__(self):
        return self.svg

    def __repr__(self):
        # Stable text repr so a re-run notebook's text/plain output never churns
        # on the object's memory address.
        return f"TensorVisual(shape={self.shape}, result_shape={self.result_shape})"

    def save(self, path):
        """Write the SVG to ``path`` and return the path.

        The text is written as UTF-8 so the ellipsis and legend glyphs survive
        the round trip. ``path`` may be a string or a path-like object.
        """
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.svg)
        return path


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
        value = obj[coord]
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        return value

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


def shape(array, theme=None, precision=2):
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
    normalized = extract_shape(array)
    value_fn = _value_fn_for(array)
    label_parts = _shape_label_parts(normalized, theme)
    svg = render_svg(
        normalized,
        value_fn=value_fn,
        label_parts=label_parts,
        theme=theme,
        precision=precision,
    )
    return TensorVisual(svg, normalized)


def index(array, index, theme=None, precision=2):
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
    normalized = extract_shape(array)
    value_fn = _value_fn_for(array)

    if is_advanced(index, normalized):
        selected, result, explanation = advanced_index(normalized, index)
        label = "mask" if not isinstance(index, tuple) else f"Index ({format_index(index)})"
        svg = render_svg(
            normalized,
            selected=selected,
            value_fn=value_fn,
            label=label,
            explanation=explanation,
            theme=theme,
            precision=precision,
        )
        return TensorVisual(
            svg, normalized, selected=selected, result=result, explanation=explanation
        )

    validate_index(index, normalized)
    selected = selected_coordinates(normalized, index)
    result = result_shape(normalized, index)
    explanation = explain_index(normalized, index)
    label_parts = _index_label_parts(index, theme)
    svg = render_svg(
        normalized,
        selected=selected,
        value_fn=value_fn,
        label_parts=label_parts,
        explanation=explanation,
        theme=theme,
        precision=precision,
    )
    return TensorVisual(
        svg, normalized, selected=selected, result=result, explanation=explanation
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


def reshape(array, new_shape, theme=None, precision=2):
    """Visualise a reshape from the source layout into a new one.

    ``array`` is an array-like with a ``.shape`` attribute or a shape tuple.
    ``new_shape`` is the target shape and may use a single ``-1`` to infer one
    axis. The source and the result are drawn side by side. Reshape keeps the
    row major order, so the element at a flat position stays at that flat
    position, which the figure makes visible by carrying the same values into
    the new layout.
    """
    theme = resolve_theme(theme)
    old = extract_shape(array)
    new = reshape_result_shape(old, new_shape)
    source_value = _source_value(array, old)

    def result_value(coord):
        return source_value(reshape_source_coord(coord, old, new))

    explanation = [
        f"Original shape: {format_shape(old)}",
        f"New shape: {format_shape(new)}",
        "Values keep their row major order, so element k stays element k.",
    ]
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
    svg = render_panels(panels, ["->"], explanation, theme=theme, precision=precision)
    return TensorVisual(svg, old, result=new, explanation=explanation)


def transpose(array, axes=None, theme=None, precision=2):
    """Visualise a transpose or permute, with axis colours following the move.

    ``axes`` is a permutation of the source axes, or ``None`` to reverse them
    like ``numpy.transpose``. The result panel colours each axis by the source
    axis it came from, so a colour can be traced across the swap.
    """
    theme = resolve_theme(theme)
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
    ]
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
    svg = render_panels(panels, ["->"], explanation, theme=theme, precision=precision)
    return TensorVisual(svg, shape, result=result, explanation=explanation)


def _reduce(array, axis, op_name, combine, theme, precision):
    """Shared body for axis reductions such as sum and mean."""
    theme = resolve_theme(theme)
    shape = extract_shape(array)
    result = reduce_result_shape(shape, axis)
    source_value = _source_value(array, shape)
    axis = axis + len(shape) if axis < 0 else axis

    values = {}
    for rc in coordinates(result):
        contributing = list(reduce_source_coords(rc, shape, axis))
        values[rc] = combine([source_value(sc) for sc in contributing])

    # Mark the source elements that collapse into the first result element.
    first = next(iter(coordinates(result)))
    selected = list(reduce_source_coords(first, shape, axis))

    disp = result or (1,)

    def result_value(coord):
        return values[() if not result else coord]

    explanation = [
        f"Original shape: {format_shape(shape)}",
        f"Reducing axis {axis} with {op_name}.",
        f"Result shape: {format_shape(result)}",
        f"Each result element combines {shape[axis]} values from axis {axis}.",
    ]
    panels = [
        {
            "shape": shape,
            "value_fn": _value_fn_for(array),
            "selected": selected,
            "caption_parts": _shape_caption_parts("source", shape, theme),
        },
        {
            "shape": disp,
            "value_fn": result_value,
            "caption_parts": _shape_caption_parts(op_name, disp, theme),
        },
    ]
    svg = render_panels(panels, ["->"], explanation, theme=theme, precision=precision)
    return TensorVisual(svg, shape, selected=selected, result=result, explanation=explanation)


def sum(array, axis, theme=None, precision=2):
    """Visualise a sum reduction over ``axis``.

    The source panel marks the elements that collapse into the first result
    element, and the result panel holds the per group sums with the reduced
    axis gone.
    """
    return _reduce(array, axis, "sum", _py_sum, theme, precision)


def mean(array, axis, theme=None, precision=2):
    """Visualise a mean reduction over ``axis``.

    The source panel marks the elements that collapse into the first result
    element, and the result panel holds the per group means with the reduced
    axis gone.
    """
    return _reduce(array, axis, "mean", lambda vs: _py_sum(vs) / len(vs), theme, precision)


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


def _combine(arrays, shapes, result, origin_fn, name, explanation, theme, precision):
    """Render several operands and their combined result in one figure.

    ``origin_fn`` maps a result coordinate to ``(operand_index, source_coord)``,
    which drives both the result values and the per cell tint. Each operand is
    tinted, and the result colours every cell by the operand it came from, so
    the seam between operands stays clear.
    """
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
    svg = render_panels(panels, connectors, explanation, theme=theme, precision=precision)
    return TensorVisual(svg, shapes[0], result=result, explanation=explanation)


def concatenate(arrays, axis=0, theme=None, precision=2):
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
    return _combine(arrays, shapes, result, origin_fn, "concatenate", explanation, theme, precision)


def stack(arrays, axis=0, theme=None, precision=2):
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
    return _combine(arrays, shapes, result, origin_fn, "stack", explanation, theme, precision)
