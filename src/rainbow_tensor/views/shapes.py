"""Shape and indexing views.

This module exposes the public functions ``shape`` and ``index``. Each returns
a small object that renders as SVG in a notebook and stays inspectable in plain
Python, so the package is testable outside a notebook.
"""

from ..indexing import (
    advanced_index,
    explain_index,
    format_index,
    format_token,
    is_advanced,
    result_shape,
    selected_coordinates,
    validate_index,
)
from ..renderers import resolve_renderer
from ..shape import extract_shape
from ..theme import resolve_theme
from ..visual import _preview_explanation, _value_fn_for, _visual


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
