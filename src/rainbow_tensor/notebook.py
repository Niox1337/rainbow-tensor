"""Notebook display layer.

This module exposes the public functions ``show_shape`` and ``show_index``.
Each returns a small object that renders as SVG in a notebook and stays
inspectable in plain Python, so the package is testable outside a notebook.
"""

from .indexing import (
    explain_index,
    format_slice,
    result_shape,
    selected_coordinates,
    validate_index,
)
from .render_svg import (
    AXIS_FRAME_COLORS,
    NEUTRAL_COLOR,
    SELECT_VALUE_COLOR,
    render_svg,
)
from .shape import extract_shape


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


def _shape_label_parts(shape):
    """Build coloured label parts for a shape.

    Each dimension is coloured to match its frame. Frame axes use their axis
    colour and the leaf axis stays neutral, since it has no frame.
    """
    ndim = len(shape)
    parts = [("Shape (", NEUTRAL_COLOR)]
    for axis, dim in enumerate(shape):
        if axis < ndim - 1:
            color = AXIS_FRAME_COLORS.get(axis, NEUTRAL_COLOR)
        else:
            color = NEUTRAL_COLOR
        parts.append((str(dim), color))
        if axis < ndim - 1:
            parts.append((", ", NEUTRAL_COLOR))
    parts.append((")", NEUTRAL_COLOR))
    return parts


def _index_label_parts(index):
    """Build coloured label parts for an index.

    Each token is coloured to match its axis. Frame axes use their axis
    colour and the leaf axis uses the selected value colour, so the label
    mirrors the colours drawn on the tensor.
    """
    ndim = len(index)
    parts = [("Index (", NEUTRAL_COLOR)]
    for axis, entry in enumerate(index):
        token = format_slice(entry) if isinstance(entry, slice) else str(entry)
        if axis < ndim - 1:
            color = AXIS_FRAME_COLORS.get(axis, NEUTRAL_COLOR)
        else:
            color = SELECT_VALUE_COLOR
        parts.append((token, color))
        if axis < ndim - 1:
            parts.append((", ", NEUTRAL_COLOR))
    parts.append((")", NEUTRAL_COLOR))
    return parts


def _display(visual):
    """Display a visual in IPython when available, otherwise do nothing."""
    try:
        from IPython.display import display
    except Exception:
        return
    display(visual)


def show_shape(shape):
    """Visualise the structure of a tensor shape.

    ``shape`` may be a tuple such as ``(2, 2, 2)`` or an array-like object
    with a ``.shape`` attribute such as a NumPy array. The tensor is rendered
    as SVG and displayed in a notebook. The returned :class:`TensorVisual`
    also exposes the SVG string.
    """
    normalized = extract_shape(shape)
    value_fn = _value_fn_for(shape)
    label_parts = _shape_label_parts(normalized)
    svg = render_svg(normalized, value_fn=value_fn, label_parts=label_parts)
    visual = TensorVisual(svg, normalized)
    _display(visual)
    return visual


def show_index(tensor_or_shape, index):
    """Visualise how an index selects elements from a tensor.

    ``tensor_or_shape`` may be a shape tuple or an array-like object with a
    ``.shape`` attribute. ``index`` must be a tuple of integers and slices
    whose length matches the tensor rank. Selected elements are highlighted,
    unselected elements stay visible but de-emphasised, and an explanation of
    the result shape is drawn below the tensor.
    """
    normalized = extract_shape(tensor_or_shape)
    validate_index(index, normalized)
    selected = selected_coordinates(normalized, index)
    result = result_shape(normalized, index)
    explanation = explain_index(normalized, index)
    value_fn = _value_fn_for(tensor_or_shape)
    label_parts = _index_label_parts(index)
    svg = render_svg(
        normalized,
        selected=selected,
        value_fn=value_fn,
        label_parts=label_parts,
        explanation=explanation,
    )
    visual = TensorVisual(
        svg, normalized, selected=selected, result=result, explanation=explanation
    )
    _display(visual)
    return visual
