"""Render one tensor body and its legend from the bounded layout model."""

import math

from ..explanations import t as translate
from ..index_mapping import CompactSelection
from ..layout import build_layout
from .elements import (
    LEGEND_FONT_SIZE,
    LEGEND_GAP,
    MONO_CHAR_RATIO,
    SANS_CHAR_RATIO,
    SWATCH,
    TEXT_MARGIN,
    VALUE_FONT_SIZE,
    VALUE_PAD,
    _is_numeric,
    _paint_attributes,
    _text_width,
    escape,
    format_value,
)
from .interaction import capturing_cells, record_panel


def _frame_color(frame, has_selection, theme):
    """Pick the stroke colour for a frame.

    A neutral container frame (an ellipsis gap) is always drawn muted. In a
    shape view every axis frame uses its colour. In an index view only
    selected frames keep their colour, the rest are dimmed.
    """
    if frame.axis is None:
        return theme.neutral
    axis_color = theme.axis_color(frame.axis)
    if not has_selection:
        return axis_color
    return axis_color if frame.selected else theme.neutral


def _fit_cell_width(layout, theme, precision):
    """Return a cell width wide enough for the widest formatted value.

    The base width comes from the theme. A wider value, for example a float
    drawn at a higher precision, grows every cell uniformly so the number
    never spills past its frame, while a short value leaves the default size
    untouched.
    """
    widest = 0
    for cell in layout.cells:
        if cell.ellipsis:
            continue
        widest = max(widest, len(format_value(cell.value, precision)))
    needed = widest * VALUE_FONT_SIZE * MONO_CHAR_RATIO + 2 * VALUE_PAD
    return max(theme.cell_w, math.ceil(needed))


def _legend_items(shape, theme, has_selection):
    """Build the legend entries: one swatch and label per axis.

    The leaf axis has no frame, so it borrows the colour used for its label
    token: the selected green in an index view, a muted tone in a shape view.
    This keeps the legend swatch consistent with the figure and the label.
    """
    ndim = len(shape)
    leaf_color = theme.surface_selected if has_selection else theme.text_muted
    items = []
    for axis, size in enumerate(shape):
        color = theme.axis_color(axis) if axis < ndim - 1 else leaf_color
        items.append((color, translate("svg.axis", axis=axis), f"· {size}"))
    return items


def _legend_width(items):
    """Estimate the width the legend row needs."""
    total = TEXT_MARGIN
    for _, name, size in items:
        label = f"{name} {size}"
        total += SWATCH + 7 + _text_width(label, LEGEND_FONT_SIZE, SANS_CHAR_RATIO)
        total += LEGEND_GAP
    return total - LEGEND_GAP + TEXT_MARGIN


def _render_legend(items, x, y, theme):
    """Render the legend row at ``(x, y)``."""
    parts = []
    cursor = x
    sy = y - SWATCH + 2
    ty = y + 2
    for color, name, size in items:
        parts.append(
            f'<rect x="{cursor:.0f}" y="{sy:.0f}" width="{SWATCH}" height="{SWATCH}" '
            f'rx="3" {_paint_attributes(fill=color)}/>'
        )
        cursor += SWATCH + 7
        label = f"{name} {size}"
        parts.append(
            f'<text x="{cursor:.0f}" y="{ty:.0f}" font-size="{LEGEND_FONT_SIZE}" '
            f"{_paint_attributes(fill=theme.text_muted)}>"
            f'<tspan {_paint_attributes(fill=color)} font-weight="600">{escape(name)}</tspan>'
            f" {escape(size)}</text>"
        )
        cursor += _text_width(label, LEGEND_FONT_SIZE, SANS_CHAR_RATIO) + LEGEND_GAP
    return "".join(parts)


def _render_cell(cell, has_selection, theme, precision, hover, tint=None):
    """Render one cell, its surface, value, and optional hover title.

    ``tint`` is an optional ``(fill, border)`` pair that colours a resting cell,
    used to mark which operand a result cell came from.
    """
    if cell.ellipsis:
        cx = cell.x + cell.width / 2
        cy = cell.y + cell.height / 2
        return (
            f'<text x="{cx:.0f}" y="{cy:.0f}" text-anchor="middle" '
            f'dominant-baseline="central" font-size="{VALUE_FONT_SIZE}" '
            f'font-family="{theme.mono_family}" {_paint_attributes(fill=theme.text_muted)}>'
            f"{escape(cell.value)}</text>"
        )

    selected = has_selection and cell.selected
    if selected:
        fill = theme.surface_selected
        border = theme.selected_border
        text_color = theme.text_selected
        weight = "700"
    elif tint is not None:
        # A group tint marks context cells, so it takes precedence over the
        # muted fill used for unselected cells in a selection view.
        fill, border = tint
        text_color = theme.text
        weight = "500"
    elif has_selection:
        fill = theme.surface_muted
        border = theme.cell_border
        text_color = theme.text_muted
        weight = "400"
    else:
        fill = theme.surface
        border = theme.cell_border
        text_color = theme.text
        weight = "500"

    display = format_value(cell.value, precision)
    if _is_numeric(cell.value):
        tx = cell.x + cell.width - VALUE_PAD
        anchor = "end"
    else:
        tx = cell.x + cell.width / 2
        anchor = "middle"
    ty = cell.y + cell.height / 2

    rect = (
        f'<rect x="{cell.x:.0f}" y="{cell.y:.0f}" '
        f'width="{cell.width:.0f}" height="{cell.height:.0f}" '
        f'rx="{theme.cell_radius:.0f}" {_paint_attributes(fill=fill, stroke=border)} '
        f'stroke-width="1"/>'
    )
    text = (
        f'<text x="{tx:.0f}" y="{ty:.0f}" text-anchor="{anchor}" '
        f'dominant-baseline="central" font-size="{VALUE_FONT_SIZE}" '
        f'font-family="{theme.mono_family}" font-weight="{weight}" '
        f"{_paint_attributes(fill=text_color)}>{escape(display)}</text>"
    )

    if hover and cell.coord is not None:
        coord = ", ".join(str(c) for c in cell.coord)
        title = translate("svg.cell", coord=coord, value=display, flat=cell.flat)
        detail = escape(title)
        return (
            f'<g aria-label="{detail}" style="cursor:help"><title>{detail}</title>{rect}{text}</g>'
        )
    return f"{rect}{text}"


def _selection_for_render(selected):
    """Keep compact selections lazy and make explicit iterables reusable."""
    return selected if isinstance(selected, CompactSelection) else list(selected or [])


def _render_body(shape, selected_list, value_fn, theme, precision, hover, cell_tint=None):
    """Render the frames and cells of one tensor in local coordinates.

    Returns ``(body_svg, width, height, theme)``. The cells may grow to fit a
    wide value, which yields a theme variant, so the caller uses the returned
    theme for any later measurement. Coordinates start at the theme padding,
    so a panel renderer can translate the whole body by an offset. ``cell_tint``
    is an optional function mapping a coordinate to a ``(fill, border)`` pair,
    used to colour each result cell by the operand it came from.
    """
    layout = build_layout(shape, selected=selected_list, value_fn=value_fn, theme=theme)
    if layout.empty:
        rendered = _render_empty_body(layout, theme)
        record_panel(rendered[0], ())
        return rendered
    has_selection = bool(selected_list)

    # Grow the cells if any value is wider than the default, then lay out again
    # so the widened cells sit on the correct grid.
    fitted = _fit_cell_width(layout, theme, precision)
    if fitted > theme.cell_w:
        theme = theme.variant(cell_w=fitted)
        layout = build_layout(shape, selected=selected_list, value_fn=value_fn, theme=theme)

    parts = []
    for frame in layout.frames:
        color = _frame_color(frame, has_selection, theme)
        parts.append(
            f'<rect x="{frame.x:.0f}" y="{frame.y:.0f}" '
            f'width="{frame.width:.0f}" height="{frame.height:.0f}" '
            f'rx="{theme.frame_radius:.0f}" fill="none" {_paint_attributes(stroke=color)} '
            f'stroke-width="{theme.frame_width}"/>'
        )
    captured = [] if capturing_cells() else None
    for cell in layout.cells:
        tint = cell_tint(cell.coord) if cell_tint and cell.coord is not None else None
        fragment = _render_cell(cell, has_selection, theme, precision, hover, tint)
        parts.append(fragment)
        if captured is not None and not cell.ellipsis and cell.coord is not None:
            captured.append((fragment, cell.coord))
    body = "".join(parts)
    if captured is not None:
        record_panel(body, captured)
    return body, layout.width, layout.height, theme


def _render_empty_body(layout, theme):
    """Render a zero-element layout without inventing a value or coordinate."""
    padding = theme.padding
    width, height = layout.width, layout.height
    marker = translate("svg.empty")
    width = max(width, _text_width(marker, VALUE_FONT_SIZE, SANS_CHAR_RATIO) + 2 * padding)
    description = translate("svg.empty_description", shape=layout.shape)
    body = (
        f'<g role="note" aria-label="{escape(description)}">'
        f'<rect x="{padding:.0f}" y="{padding:.0f}" '
        f'width="{width - 2 * padding:.0f}" height="{height - 2 * padding:.0f}" '
        f'rx="{theme.frame_radius:.0f}" fill="none" {_paint_attributes(stroke=theme.neutral)} '
        f'stroke-width="{theme.frame_width}" stroke-dasharray="4 4"/>'
        f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle" '
        f'dominant-baseline="central" font-size="{VALUE_FONT_SIZE}" '
        f"{_paint_attributes(fill=theme.text_muted)}>{escape(marker)}</text></g>"
    )
    return body, width, height, theme
