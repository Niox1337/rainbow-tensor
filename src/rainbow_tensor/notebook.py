"""Notebook display layer.

This module exposes the public functions ``shape`` and ``index``.
Each returns a small object that renders as SVG in a notebook and stays
inspectable in plain Python, so the package is testable outside a notebook.
"""

from .indexing import (
    explain_index,
    format_token,
    result_shape,
    selected_coordinates,
    validate_index,
)
from .render_svg import render_svg
from .shape import extract_shape
from .theme import resolve_theme


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
    which case its own values are rendered, or a shape tuple. ``index`` must
    be a tuple of integers and slices whose length matches the tensor rank.
    Selected elements are highlighted, unselected elements stay visible but
    de-emphasised, and an explanation of the result shape is drawn below the
    tensor. ``theme`` and ``precision`` behave as in :func:`shape`. The
    returned :class:`TensorVisual` renders as SVG in a notebook through
    ``_repr_svg_`` and also exposes the SVG string for inspection and testing.
    """
    theme = resolve_theme(theme)
    normalized = extract_shape(array)
    validate_index(index, normalized)
    selected = selected_coordinates(normalized, index)
    result = result_shape(normalized, index)
    explanation = explain_index(normalized, index)
    value_fn = _value_fn_for(array)
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
