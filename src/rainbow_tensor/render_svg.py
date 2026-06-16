"""SVG rendering.

This module turns a :class:`~rainbow_tensor.layout.Layout` into an SVG
string. Each axis has its own colour. Axis 0 frames are red and axis 1
frames are orange. The leaf axis elements are plain text. Selected elements
are drawn green and, in an index view, only the selected frames keep their
axis colour while the rest are drawn in a neutral dark tone.
"""

from .layout import build_layout

AXIS_FRAME_COLORS = {0: "#dc2626", 1: "#f97316"}
SELECT_VALUE_COLOR = "#16a34a"
NEUTRAL_COLOR = "#111827"
TEXT_COLOR = "#111827"
LABEL_COLOR = "#334155"

LABEL_HEIGHT = 30
LINE_HEIGHT = 20
FRAME_WIDTH = 3
FONT_FAMILY = "ui-monospace, Menlo, Consolas, monospace"


def escape(text):
    """Escape a value so it is safe inside SVG text and attributes."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_document(content, width, height):
    """Wrap rendered elements in a complete SVG document."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" '
        f'font-family="{FONT_FAMILY}">'
        f"{content}"
        f"</svg>"
    )


def _frame_color(frame, has_selection):
    """Pick the stroke colour for a frame.

    In a shape view every frame uses its axis colour. In an index view only
    selected frames keep their axis colour, the rest are neutral.
    """
    axis_color = AXIS_FRAME_COLORS.get(frame.axis, NEUTRAL_COLOR)
    if not has_selection:
        return axis_color
    return axis_color if frame.selected else NEUTRAL_COLOR


def _cell_color(cell, has_selection):
    """Pick the text colour for a cell value."""
    if has_selection and cell.selected:
        return SELECT_VALUE_COLOR
    return TEXT_COLOR


def _label_element(x, y, parts):
    """Build a label text element from coloured parts."""
    spans = "".join(
        f'<tspan fill="{escape(color)}">{escape(text)}</tspan>' for text, color in parts
    )
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" font-size="16" font-weight="600">{spans}</text>'
    )


def render_svg(
    shape, selected=None, value_fn=None, label="", label_parts=None, explanation=None
):
    """Render a tensor as an SVG string.

    ``selected`` is an iterable of selected coordinates. ``value_fn`` maps a
    coordinate to its display value. ``label`` is a plain label string while
    ``label_parts`` is an optional list of ``(text, colour)`` pairs for a
    coloured label. ``explanation`` is an optional list of lines drawn below.
    """
    explanation = explanation or []
    selected_list = list(selected or [])
    has_selection = len(selected_list) > 0
    layout = build_layout(shape, selected=selected_list, value_fn=value_fn)

    parts = []

    for frame in layout.frames:
        color = _frame_color(frame, has_selection)
        parts.append(
            f'<rect x="{frame.x:.0f}" y="{frame.y:.0f}" '
            f'width="{frame.width:.0f}" height="{frame.height:.0f}" '
            f'rx="6" fill="none" stroke="{color}" stroke-width="{FRAME_WIDTH}"/>'
        )

    for cell in layout.cells:
        color = _cell_color(cell, has_selection)
        weight = "700" if (has_selection and cell.selected) else "400"
        cx = cell.x + cell.width / 2
        cy = cell.y + cell.height / 2
        parts.append(
            f'<text x="{cx:.0f}" y="{cy:.0f}" text-anchor="middle" '
            f'dominant-baseline="central" font-size="15" font-weight="{weight}" '
            f'fill="{color}">{escape(cell.value)}</text>'
        )

    width = layout.width
    y = layout.height + LABEL_HEIGHT

    if label_parts:
        parts.append(_label_element(20, y, label_parts))
        y += LINE_HEIGHT
    elif label:
        parts.append(
            f'<text x="20" y="{y:.0f}" font-size="16" font-weight="600" '
            f'fill="{LABEL_COLOR}">{escape(label)}</text>'
        )
        y += LINE_HEIGHT

    for line in explanation:
        y += LINE_HEIGHT - 4
        parts.append(
            f'<text x="20" y="{y:.0f}" font-size="13" fill="{LABEL_COLOR}">'
            f"{escape(line)}</text>"
        )

    height = y + 16
    return svg_document("".join(parts), width, height)
