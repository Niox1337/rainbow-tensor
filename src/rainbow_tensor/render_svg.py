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
import unicodedata

from .explanations import get_resolved_language
from .explanations import t as translate
from .index_mapping import CompactSelection
from .layout import build_layout
from .shape import format_shape_label
from .theme import _AUTO_FALLBACKS, LIGHT, _auto_stylesheet, resolve_theme

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


def _paint_attributes(**colors):
    """Keep light presentation attributes when adaptive CSS is unavailable.

    Each adaptive paint also receives an inline CSS declaration. A browser
    resolves its semantic token when the preferred scheme changes, while SVG
    readers without CSS custom properties retain the literal light colour.
    Custom colour strings pass through without palette substitution.
    """
    attributes = []
    styles = []
    for name, color in colors.items():
        fallback = _AUTO_FALLBACKS.get(color, color)
        attributes.append(f'{name}="{escape(fallback)}"')
        if color in _AUTO_FALLBACKS:
            styles.append(f"{name}:{color}")
    if styles:
        attributes.append(f'style="{escape(";".join(styles))}"')
    return " ".join(attributes)


def svg_document(content, width, height, theme=None, aria_label=None):
    """Wrap rendered elements in a complete SVG document.

    A background rectangle in the theme colour fills the whole canvas so the
    dark preset reads as dark even on a transparent host page.
    Adaptive palettes include scoped CSS in the exported SVG itself.
    """
    t = theme or LIGHT
    accessible_name = aria_label or translate("svg.visualisation")
    language = get_resolved_language()
    language_attribute = f'lang="{escape(language)}" ' if language != "en" else ""
    background = (
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" '
        f'rx="14" {_paint_attributes(fill=t.background, stroke=t.card_border)}/>'
    )
    adaptive_attribute = 'data-rt-theme="auto" ' if t.adaptive else ""
    stylesheet = _auto_stylesheet() if t.adaptive else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'{adaptive_attribute}'
        f'{language_attribute}'
        f'role="img" aria-label="{escape(accessible_name)}" '
        f'width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" '
        f'style="max-width:100%;height:auto" '
        f'font-family="{t.sans_family}">'
        f"{stylesheet}{background}{content}"
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
    """Estimate width using full glyph cells for CJK and zero for combining marks.

    Numeric character counts remain supported for existing renderer callers.
    """
    if isinstance(char_count, str):
        char_count = sum(
            0 if unicodedata.combining(char) else
            2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
            for char in char_count
        )
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
            f'{_paint_attributes(fill=theme.text_muted)}>'
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
        f'{_paint_attributes(fill=text_color)}>{escape(display)}</text>'
    )

    if hover and cell.coord is not None:
        coord = ", ".join(str(c) for c in cell.coord)
        title = translate("svg.cell", coord=coord, value=display, flat=cell.flat)
        detail = escape(title)
        return (
            f'<g aria-label="{detail}" style="cursor:help">'
            f"<title>{detail}</title>{rect}{text}</g>"
        )
    return f"{rect}{text}"


def _label_element(x, y, parts):
    """Build a label text element from coloured parts."""
    spans = "".join(
        f'<tspan {_paint_attributes(fill=color)}>{escape(text)}</tspan>' for text, color in parts
    )
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" font-size="{LABEL_FONT_SIZE}" '
        f'font-weight="700">{spans}</text>'
    )


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
        return _render_empty_body(layout, theme)
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
    for cell in layout.cells:
        tint = cell_tint(cell.coord) if cell_tint and cell.coord is not None else None
        parts.append(_render_cell(cell, has_selection, theme, precision, hover, tint))
    return "".join(parts), layout.width, layout.height, theme


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
        f'{_paint_attributes(fill=theme.text_muted)}>{escape(marker)}</text></g>'
    )
    return body, width, height, theme


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
    coloured label. ``explanation`` is accepted for renderer interface
    compatibility, but plain explanation text is emitted by ``TensorVisual``.
    ``theme`` chooses the palette and geometry, ``precision`` controls float
    formatting, ``legend`` toggles the axis legend, and ``hover`` toggles the
    per-cell title used for tooltips.

    The logical scalar shape ``()`` displays one cell. Shapes with a zero
    dimension display an empty marker and never invoke ``value_fn``.
    """
    theme = resolve_theme(theme)
    selected_list = [] if 0 in shape else _selection_for_render(selected)
    if (not shape or 0 in shape) and not label_parts and not label:
        label = f"Shape {format_shape_label(shape)}"

    body, body_w, body_h, theme = _render_body(
        shape, selected_list, value_fn, theme, precision, hover
    )
    has_selection = bool(selected_list)
    parts = [body]

    if label_parts:
        label_text = "".join(str(text) for text, _ in label_parts)
    else:
        label_text = str(label)
    label_len = label_text

    legend_items = _legend_items(shape, theme, has_selection) if legend else []

    content_width = _text_width(label_len, LABEL_FONT_SIZE, SANS_CHAR_RATIO) + 2 * TEXT_MARGIN
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
            f'font-weight="700" {_paint_attributes(fill=theme.heading)}>{escape(label)}</text>'
        )
        y += LINE_HEIGHT

    if legend_items:
        y += LEGEND_HEIGHT - LINE_HEIGHT + 8
        parts.append(_render_legend(legend_items, TEXT_MARGIN, y, theme))
        y += 6

    height = y + 18
    aria_label = label_text or translate("svg.tensor_shape", shape=tuple(shape))
    return svg_document(
        "".join(parts), width, height, theme=theme, aria_label=aria_label
    )


PANEL_GAP = 64  # horizontal room between panels, holding the connector glyph
CONNECTOR_FONT_SIZE = 26
CAPTION_FONT_SIZE = 14
CAPTION_HEIGHT = 26


def _centered_parts(cx, y, parts, font_size, weight="700"):
    """Render coloured ``parts`` centred on ``cx`` at baseline ``y``."""
    spans = "".join(
        f'<tspan {_paint_attributes(fill=color)}>{escape(text)}</tspan>' for text, color in parts
    )
    return (
        f'<text x="{cx:.0f}" y="{y:.0f}" text-anchor="middle" '
        f'font-size="{font_size}" font-weight="{weight}">{spans}</text>'
    )


def render_panels(panels, connectors=None, explanation=None, theme=None, precision=2, hover=True):
    """Render several tensors side by side in one SVG.

    Each panel is a dict with a ``shape``, an optional ``value_fn`` and
    ``selected`` iterable, an optional ``theme`` override used to tint an
    operand, an optional ``cell_tint`` function colouring each cell by its
    origin, and optional ``caption_parts`` drawn under the panel as coloured
    ``(text, colour)`` pairs. ``connectors`` is a list of glyph strings drawn
    between consecutive panels, for example ``"->"`` or ``"+"``. ``explanation``
    is accepted for renderer interface compatibility, but plain explanation
    text is emitted by ``TensorVisual``. This composes the existing single
    tensor body, so the source and the result of an operation, or several
    operands, sit in one figure.

    Scalar panels keep shape ``()`` and pass that coordinate to ``value_fn``.
    Empty panels show their shape and an empty marker without value callbacks.
    A panel with its own background paints behind both its body and caption,
    so fixed and adaptive palettes remain readable in the same document.
    """
    theme = resolve_theme(theme)
    connectors = connectors or []
    adaptive = theme.adaptive

    bodies = []
    panel_themes = []
    panel_labels = []
    for panel in panels:
        ptheme = panel.get("theme") or theme
        adaptive = adaptive or ptheme.adaptive
        body, w, h, ptheme = _render_body(
            panel["shape"],
            [] if 0 in panel["shape"] else _selection_for_render(panel.get("selected")),
            panel.get("value_fn"),
            ptheme,
            precision,
            hover,
            panel.get("cell_tint"),
        )
        caption = panel.get("caption_parts")
        if not panel["shape"] or 0 in panel["shape"]:
            if not caption:
                caption = [(
                    translate("svg.shape", shape=format_shape_label(panel['shape'])), ptheme.heading
                )]
                panel = dict(panel, caption_parts=caption)
        if caption:
            caption_length = "".join(str(text) for text, _ in caption)
            caption_width = (
                _text_width(caption_length, CAPTION_FONT_SIZE, SANS_CHAR_RATIO)
                + 2 * TEXT_MARGIN
            )
            if caption_width > w:
                body = f'<g transform="translate({(caption_width - w) / 2:.0f}, 0)">{body}</g>'
                w = caption_width
        bodies.append((body, w, h, panel))
        panel_themes.append(ptheme)
        if caption:
            panel_labels.append("".join(str(text) for text, _ in caption))
        else:
            panel_labels.append(translate("svg.tensor_shape", shape=tuple(panel['shape'])))

    row_height = max(h for _, _, h, _ in bodies)
    has_caption = any(p.get("caption_parts") for _, _, _, p in bodies)
    height = row_height + 18
    if has_caption:
        height += CAPTION_HEIGHT - 2

    parts = []
    centers = []
    cursor = 0.0
    for i, (body, w, h, _) in enumerate(bodies):
        ptheme = panel_themes[i]
        if ptheme.background != theme.background:
            parts.append(
                f'<rect x="{cursor:.0f}" y="0" width="{w:.0f}" height="{height:.0f}" '
                f'rx="14" {_paint_attributes(fill=ptheme.background)}/>'
            )
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
                f'{_paint_attributes(fill=theme.text_muted)}>{escape(glyph)}</text>'
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

    content_width = row_width

    accessible_parts = []
    for i, panel_label in enumerate(panel_labels):
        accessible_parts.append(panel_label)
        if i < len(panel_labels) - 1:
            connector = connectors[i] if i < len(connectors) else "->"
            accessible_parts.append(str(connector))
    aria_label = " ".join(accessible_parts) or translate("svg.visualisation")

    return svg_document(
        "".join(parts),
        content_width,
        height,
        theme=theme.variant(adaptive=True) if adaptive and not theme.adaptive else theme,
        aria_label=aria_label,
    )
