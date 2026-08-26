"""Shape and indexing views.

This module exposes the public functions ``shape`` and ``index``. Each returns
a small object that renders as SVG in a notebook and stays inspectable in plain
Python, so the package is testable outside a notebook.
"""

from ..index_mapping import IndexMapping
from ..indexing import format_index, format_token
from ..renderers import resolve_renderer
from ..selection import BasicSelection
from ..shape import extract_shape
from ..theme import resolve_theme
from ..visual import (
    _preview_explanation,
    _shape_caption_parts,
    _source_value,
    _value_fn_for,
    _visual,
)


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
    A scalar has shape ``()`` and one value. A shape containing zero has no
    elements, so it displays an empty marker without reading array values.
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


def index(array, index, theme=None, precision=2, renderer=None, *, show_result=False):
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

    Set ``show_result=True`` to draw an unhighlighted source, an arrow, and the
    indexed result. Result values retain the order and repeated positions of
    the index. A scalar result occupies one display cell, and empty results
    contain no value cells. The default remains the original source highlight.

    ``visual.index_mapping.source_coord(coordinate)`` maps an output coordinate
    back to its source. ``result_count`` includes repeated output positions.
    ``visual.selected`` is a compact source-highlight iterable, not the output
    order. Basic selections also expose ``count``. Convert a selection to a list
    only when all unique source coordinates are explicitly needed.
    """
    if not isinstance(show_result, bool):
        raise TypeError("show_result must be a boolean")
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    normalized = extract_shape(array)
    value_fn = _value_fn_for(array)
    mapping = IndexMapping(normalized, index)
    selected = mapping.selection
    result = mapping.result_shape
    preview_shapes = [normalized, result] if show_result else [normalized]
    explanation = mapping.explanation + _preview_explanation(preview_shapes, theme)

    if show_result:
        source_value = _source_value(array, normalized)

        def result_value(coordinate):
            """Read one displayed output through its ordered source mapping."""
            return source_value(mapping.source_coord(coordinate))

        panels = [
            {
                "shape": normalized,
                "value_fn": value_fn,
                "caption_parts": _shape_caption_parts("source", normalized, theme),
            },
            {
                "shape": result,
                "value_fn": result_value,
                "selected": BasicSelection(result, (Ellipsis,)),
                "caption_parts": _shape_caption_parts("result", result, theme),
            },
        ]
        explanation.append("The result preserves the index order, including repeated elements.")
        if mapping.result_count == 0:
            explanation.append("The index selects no elements. The result is empty.")
        content = renderer.render_panels(
            panels=panels, connectors=["->"], explanation=explanation,
            theme=theme, precision=precision,
        )
    elif mapping.is_advanced:
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
    else:
        content = renderer.render_tensor(
            shape=normalized,
            selected=selected,
            value_fn=value_fn,
            label_parts=_index_label_parts(index, theme),
            explanation=explanation,
            theme=theme,
            precision=precision,
        )
    visual = _visual(
        content, normalized, renderer, selected=selected, result=result, explanation=explanation
    )
    visual.index_mapping = mapping
    return visual
