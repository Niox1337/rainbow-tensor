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

from ._svg.elements import (
    EXPLANATION_FONT_SIZE,
    LABEL_FONT_SIZE,
    LABEL_HEIGHT,
    LEGEND_FONT_SIZE,
    LEGEND_GAP,
    LEGEND_HEIGHT,
    LINE_HEIGHT,
    MONO_CHAR_RATIO,
    SANS_CHAR_RATIO,
    SWATCH,
    TEXT_MARGIN,
    VALUE_FONT_SIZE,
    VALUE_PAD,
    _centered_parts,
    _is_numeric,
    _label_element,
    _paint_attributes,
    _text_width,
    escape,
    format_value,
    svg_document,
)
from ._svg.tensor import (
    _fit_cell_width,
    _frame_color,
    _legend_items,
    _legend_width,
    _render_body,
    _render_cell,
    _render_empty_body,
    _render_legend,
    _selection_for_render,
)
from .explanations import t as translate
from .shape import format_shape_label
from .theme import LIGHT, resolve_theme

# Preserve the historical rendering entry points and helper imports.
__all__ = [
    "AXIS_FRAME_COLORS",
    "CAPTION_FONT_SIZE",
    "CAPTION_HEIGHT",
    "CONNECTOR_FONT_SIZE",
    "EXPLANATION_FONT_SIZE",
    "LABEL_COLOR",
    "LABEL_FONT_SIZE",
    "LABEL_HEIGHT",
    "LEGEND_FONT_SIZE",
    "LEGEND_GAP",
    "LEGEND_HEIGHT",
    "LINE_HEIGHT",
    "MONO_CHAR_RATIO",
    "NEUTRAL_COLOR",
    "PANEL_GAP",
    "SANS_CHAR_RATIO",
    "SELECT_VALUE_COLOR",
    "SWATCH",
    "TEXT_COLOR",
    "TEXT_MARGIN",
    "VALUE_FONT_SIZE",
    "VALUE_PAD",
    "_centered_parts",
    "_fit_cell_width",
    "_frame_color",
    "_is_numeric",
    "_label_element",
    "_legend_items",
    "_legend_width",
    "_paint_attributes",
    "_render_body",
    "_render_cell",
    "_render_empty_body",
    "_render_legend",
    "_selection_for_render",
    "_text_width",
    "escape",
    "format_value",
    "render_panels",
    "render_svg",
    "svg_document",
]

# Back-compatible colour constants, sourced from the light preset so existing
# imports and the notebook layer keep working.
AXIS_FRAME_COLORS = {0: LIGHT.axis_colors[0], 1: LIGHT.axis_colors[1]}
SELECT_VALUE_COLOR = LIGHT.surface_selected
NEUTRAL_COLOR = LIGHT.neutral
TEXT_COLOR = LIGHT.text
LABEL_COLOR = LIGHT.heading


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
    return svg_document("".join(parts), width, height, theme=theme, aria_label=aria_label)


PANEL_GAP = 64  # horizontal room between panels, holding the connector glyph
CONNECTOR_FONT_SIZE = 26
CAPTION_FONT_SIZE = 14
CAPTION_HEIGHT = 26


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
                caption = [
                    (
                        translate("svg.shape", shape=format_shape_label(panel["shape"])),
                        ptheme.heading,
                    )
                ]
                panel = dict(panel, caption_parts=caption)
        if caption:
            caption_length = "".join(str(text) for text, _ in caption)
            caption_width = (
                _text_width(caption_length, CAPTION_FONT_SIZE, SANS_CHAR_RATIO) + 2 * TEXT_MARGIN
            )
            if caption_width > w:
                body = f'<g transform="translate({(caption_width - w) / 2:.0f}, 0)">{body}</g>'
                w = caption_width
        bodies.append((body, w, h, panel))
        panel_themes.append(ptheme)
        if caption:
            panel_labels.append("".join(str(text) for text, _ in caption))
        else:
            panel_labels.append(translate("svg.tensor_shape", shape=tuple(panel["shape"])))

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
                f"{_paint_attributes(fill=theme.text_muted)}>{escape(glyph)}</text>"
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
