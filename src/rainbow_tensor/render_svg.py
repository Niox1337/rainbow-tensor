"""SVG rendering.

This module turns a :class:`~rainbow_tensor.layout.Layout` into an SVG
string. Every visual choice comes from the active
:class:`~rainbow_tensor.theme.Theme`, so the same tensor renders in the light
or the dark preset without any change here.

Each non-leaf axis draws a frame in its rainbow colour. Leaf elements sit in
rounded cells. In a shape view every frame keeps its axis colour. In an index
view the selected elements fill green, the selected frames keep their colour,
and the rest are dimmed so the selection stands out. An axis legend below the
tensor names each axis with its size in the matching colour, and each cell
carries an optional hover title with its coordinate and flat index.
"""

import math

from .layout import build_layout
from .theme import LIGHT, resolve_theme

# Back-compatible colour constants, sourced from the light preset so existing
# imports and the notebook layer keep working.
AXIS_FRAME_COLORS = {0: LIGHT.axis_colors[0], 1: LIGHT.axis_colors[1]}
SELECT_VALUE_COLOR = LIGHT.surface_selected
NEUTRAL_COLOR = LIGHT.neutral
TEXT_COLOR = LIGHT.text
LABEL_COLOR = LIGHT.heading

LABEL_HEIGHT = 34
LINE_HEIGHT = 20
LEGEND_HEIGHT = 30
LEGEND_GAP = 18
SWATCH = 13
TEXT_MARGIN = 20
LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 13
EXPLANATION_FONT_SIZE = 13
VALUE_FONT_SIZE = 15
VALUE_PAD = 10
# Approximate glyph width relative to the font size, per family.
MONO_CHAR_RATIO = 0.6
SANS_CHAR_RATIO = 0.55


def escape(text):
    """Escape a value so it is safe inside SVG text and attributes."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_document(content, width, height, theme=None):
    """Wrap rendered elements in a complete SVG document.

    A background rectangle in the theme colour fills the whole canvas so the
    dark preset reads as dark even on a transparent host page.
    """
    t = theme or LIGHT
    background = (
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" '
        f'rx="14" fill="{escape(t.background)}" stroke="{escape(t.card_border)}"/>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" '
        f'font-family="{t.sans_family}">'
        f"{background}{content}"
        f"</svg>"
    )


def format_value(value, precision):
    """Format a cell value for display.

    Floats use a fixed number of decimals so a column lines up. Integers and
    other values are shown as their plain string.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{precision}f}"
    return str(value)


def _is_numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def _text_width(char_count, font_size, ratio):
    """Estimate the rendered width of a string."""
    return char_count * font_size * ratio


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
        items.append((color, f"axis {axis}", f"· {size}"))
    return items


def _legend_width(items):
    """Estimate the width the legend row needs."""
    total = TEXT_MARGIN
    for _, name, size in items:
        label = f"{name} {size}"
        total += SWATCH + 7 + _text_width(len(label), LEGEND_FONT_SIZE, SANS_CHAR_RATIO)
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
            f'rx="3" fill="{escape(color)}"/>'
        )
        cursor += SWATCH + 7
        label = f"{name} {size}"
        parts.append(
            f'<text x="{cursor:.0f}" y="{ty:.0f}" font-size="{LEGEND_FONT_SIZE}" '
            f'fill="{escape(theme.text_muted)}">'
            f'<tspan fill="{escape(color)}" font-weight="600">{escape(name)}</tspan>'
            f" {escape(size)}</text>"
        )
        cursor += _text_width(len(label), LEGEND_FONT_SIZE, SANS_CHAR_RATIO) + LEGEND_GAP
    return "".join(parts)


def _render_cell(cell, has_selection, theme, precision, hover):
    """Render one cell, its surface, value, and optional hover title."""
    if cell.ellipsis:
        cx = cell.x + cell.width / 2
        cy = cell.y + cell.height / 2
        return (
            f'<text x="{cx:.0f}" y="{cy:.0f}" text-anchor="middle" '
            f'dominant-baseline="central" font-size="{VALUE_FONT_SIZE}" '
            f'font-family="{theme.mono_family}" fill="{escape(theme.text_muted)}">'
            f"{escape(cell.value)}</text>"
        )

    selected = has_selection and cell.selected
    if selected:
        fill = theme.surface_selected
        border = theme.selected_border
        text_color = theme.text_selected
        weight = "700"
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
        f'rx="{theme.cell_radius:.0f}" fill="{escape(fill)}" '
        f'stroke="{escape(border)}" stroke-width="1"/>'
    )
    text = (
        f'<text x="{tx:.0f}" y="{ty:.0f}" text-anchor="{anchor}" '
        f'dominant-baseline="central" font-size="{VALUE_FONT_SIZE}" '
        f'font-family="{theme.mono_family}" font-weight="{weight}" '
        f'fill="{escape(text_color)}">{escape(display)}</text>'
    )

    if hover and cell.coord is not None:
        coord = ", ".join(str(c) for c in cell.coord)
        title = f"[{coord}]  value {display}  ·  flat {cell.flat}"
        return f"<g><title>{escape(title)}</title>{rect}{text}</g>"
    return f"{rect}{text}"


def _label_element(x, y, parts):
    """Build a label text element from coloured parts."""
    spans = "".join(
        f'<tspan fill="{escape(color)}">{escape(text)}</tspan>' for text, color in parts
    )
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" font-size="{LABEL_FONT_SIZE}" '
        f'font-weight="700">{spans}</text>'
    )


def _render_body(shape, selected_list, value_fn, theme, precision, hover):
    """Render the frames and cells of one tensor in local coordinates.

    Returns ``(body_svg, width, height, theme)``. The cells may grow to fit a
    wide value, which yields a theme variant, so the caller uses the returned
    theme for any later measurement. Coordinates start at the theme padding,
    so a panel renderer can translate the whole body by an offset.
    """
    has_selection = len(selected_list) > 0
    layout = build_layout(shape, selected=selected_list, value_fn=value_fn, theme=theme)

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
            f'rx="{theme.frame_radius:.0f}" fill="none" stroke="{color}" '
            f'stroke-width="{theme.frame_width}"/>'
        )
    for cell in layout.cells:
        parts.append(_render_cell(cell, has_selection, theme, precision, hover))
    return "".join(parts), layout.width, layout.height, theme


def render_svg(
    shape,
    selected=None,
    value_fn=None,
    label="",
    label_parts=None,
    explanation=None,
    theme=None,
    precision=2,
    legend=True,
    hover=True,
):
    """Render a tensor as an SVG string.

    ``selected`` is an iterable of selected coordinates and ``value_fn`` maps a
    coordinate to its display value. ``label`` is a plain label while
    ``label_parts`` is an optional list of ``(text, colour)`` pairs for a
    coloured label. ``explanation`` is an optional list of lines drawn below.
    ``theme`` chooses the palette and geometry, ``precision`` controls float
    formatting, ``legend`` toggles the axis legend, and ``hover`` toggles the
    per-cell title used for tooltips.
    """
    theme = resolve_theme(theme)
    explanation = explanation or []
    selected_list = list(selected or [])

    body, body_w, body_h, theme = _render_body(
        shape, selected_list, value_fn, theme, precision, hover
    )
    has_selection = len(selected_list) > 0
    parts = [body]

    if label_parts:
        label_len = sum(len(text) for text, _ in label_parts)
    else:
        label_len = len(label)

    legend_items = _legend_items(shape, theme, has_selection) if legend else []

    content_width = _text_width(label_len, LABEL_FONT_SIZE, SANS_CHAR_RATIO) + 2 * TEXT_MARGIN
    for line in explanation:
        content_width = max(
            content_width,
            _text_width(len(line), EXPLANATION_FONT_SIZE, MONO_CHAR_RATIO) + 2 * TEXT_MARGIN,
        )
    if legend_items:
        content_width = max(content_width, _legend_width(legend_items))

    width = max(body_w, content_width)
    y = body_h + LABEL_HEIGHT - 8

    if label_parts:
        parts.append(_label_element(TEXT_MARGIN, y, label_parts))
        y += LINE_HEIGHT
    elif label:
        parts.append(
            f'<text x="{TEXT_MARGIN}" y="{y:.0f}" font-size="{LABEL_FONT_SIZE}" '
            f'font-weight="700" fill="{escape(theme.heading)}">{escape(label)}</text>'
        )
        y += LINE_HEIGHT

    if legend_items:
        y += LEGEND_HEIGHT - LINE_HEIGHT + 8
        parts.append(_render_legend(legend_items, TEXT_MARGIN, y, theme))
        y += 6

    for line in explanation:
        y += LINE_HEIGHT - 4
        parts.append(
            f'<text x="{TEXT_MARGIN}" y="{y:.0f}" font-size="{EXPLANATION_FONT_SIZE}" '
            f'font-family="{theme.mono_family}" fill="{escape(theme.text_muted)}">'
            f"{escape(line)}</text>"
        )

    height = y + 18
    return svg_document("".join(parts), width, height, theme=theme)


PANEL_GAP = 64  # horizontal room between panels, holding the connector glyph
CONNECTOR_FONT_SIZE = 26
CAPTION_FONT_SIZE = 14
CAPTION_HEIGHT = 26


def _centered_parts(cx, y, parts, font_size, weight="700"):
    """Render coloured ``parts`` centred on ``cx`` at baseline ``y``."""
    spans = "".join(
        f'<tspan fill="{escape(color)}">{escape(text)}</tspan>' for text, color in parts
    )
    return (
        f'<text x="{cx:.0f}" y="{y:.0f}" text-anchor="middle" '
        f'font-size="{font_size}" font-weight="{weight}">{spans}</text>'
    )


def render_panels(panels, connectors=None, explanation=None, theme=None, precision=2, hover=True):
    """Render several tensors side by side in one SVG.

    Each panel is a dict with a ``shape``, an optional ``value_fn`` and
    ``selected`` iterable, an optional ``theme`` override used to tint an
    operand, and optional ``caption_parts`` drawn under the panel as coloured
    ``(text, colour)`` pairs. ``connectors`` is a list of glyph strings drawn
    between consecutive panels, for example ``"->"`` or ``"+"``. ``explanation``
    is a shared list of lines drawn below every panel. This composes the
    existing single tensor body, so the source and the result of an operation,
    or several operands, sit in one figure.
    """
    theme = resolve_theme(theme)
    connectors = connectors or []
    explanation = explanation or []

    bodies = []
    for panel in panels:
        ptheme = panel.get("theme") or theme
        body, w, h, ptheme = _render_body(
            panel["shape"],
            list(panel.get("selected") or []),
            panel.get("value_fn"),
            ptheme,
            precision,
            hover,
        )
        bodies.append((body, w, h, panel))

    row_height = max(h for _, _, h, _ in bodies)
    has_caption = any(p.get("caption_parts") for _, _, _, p in bodies)

    parts = []
    centers = []
    cursor = 0.0
    for i, (body, w, h, _) in enumerate(bodies):
        dy = (row_height - h) / 2
        parts.append(f'<g transform="translate({cursor:.0f}, {dy:.0f})">{body}</g>')
        centers.append(cursor + w / 2)
        cursor += w
        if i < len(bodies) - 1:
            glyph = connectors[i] if i < len(connectors) else "->"
            cx = cursor + PANEL_GAP / 2
            parts.append(
                f'<text x="{cx:.0f}" y="{row_height / 2:.0f}" text-anchor="middle" '
                f'dominant-baseline="central" font-size="{CONNECTOR_FONT_SIZE}" '
                f'fill="{escape(theme.text_muted)}">{escape(glyph)}</text>'
            )
            cursor += PANEL_GAP

    row_width = cursor
    y = row_height

    if has_caption:
        y += CAPTION_HEIGHT - 8
        for center, (_, _, _, panel) in zip(centers, bodies):
            caption = panel.get("caption_parts")
            if caption:
                parts.append(_centered_parts(center, y, caption, CAPTION_FONT_SIZE))
        y += 6

    for line in explanation:
        y += LINE_HEIGHT - 4
        parts.append(
            f'<text x="{TEXT_MARGIN}" y="{y:.0f}" font-size="{EXPLANATION_FONT_SIZE}" '
            f'font-family="{theme.mono_family}" fill="{escape(theme.text_muted)}">'
            f"{escape(line)}</text>"
        )

    content_width = row_width
    for line in explanation:
        content_width = max(
            content_width,
            _text_width(len(line), EXPLANATION_FONT_SIZE, MONO_CHAR_RATIO) + 2 * TEXT_MARGIN,
        )

    return svg_document("".join(parts), content_width, y + 18, theme=theme)
