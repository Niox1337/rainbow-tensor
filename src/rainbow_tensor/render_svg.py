"""SVG rendering.

This module turns a :class:`~rainbow_tensor.layout.Layout` into an SVG
string. It draws the tensor blocks, cells, values, and a label. Selected
cells are highlighted and, when a selection exists, the unselected cells are
de-emphasised but still readable. An optional explanation is drawn below.
"""

from .layout import build_layout

CELL_FILL = "#ffffff"
CELL_STROKE = "#94a3b8"
BLOCK_STROKE = "#cbd5e1"
HIGHLIGHT_FILL = "#fde68a"
HIGHLIGHT_STROKE = "#d97706"
TEXT_COLOR = "#0f172a"
LABEL_COLOR = "#334155"
DIM_OPACITY = 0.3

LABEL_HEIGHT = 28
LINE_HEIGHT = 20
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


def render_svg(shape, selected=None, value_fn=None, label="", explanation=None):
    """Render a tensor as an SVG string.

    ``selected`` is an iterable of selected coordinates. ``value_fn`` maps a
    coordinate to its display value. ``label`` is drawn under the tensor and
    ``explanation`` is an optional list of lines drawn below the label.
    """
    explanation = explanation or []
    selected_list = list(selected or [])
    has_selection = len(selected_list) > 0
    layout = build_layout(shape, selected=selected_list, value_fn=value_fn)

    parts = []

    for block in layout.blocks:
        if has_selection:
            block_selected = any(
                cell.selected for cell in layout.cells if cell.coord[0] == block.index
            )
            opacity = 1.0 if block_selected else DIM_OPACITY
        else:
            opacity = 1.0
        parts.append(
            f'<rect x="{block.x:.0f}" y="{block.y:.0f}" '
            f'width="{block.width:.0f}" height="{block.height:.0f}" '
            f'rx="8" fill="none" stroke="{BLOCK_STROKE}" stroke-width="1.5" '
            f'opacity="{opacity:.2f}"/>'
        )

    for cell in layout.cells:
        if cell.selected:
            fill = HIGHLIGHT_FILL
            stroke = HIGHLIGHT_STROKE
            opacity = 1.0
            weight = "700"
        else:
            fill = CELL_FILL
            stroke = CELL_STROKE
            opacity = DIM_OPACITY if has_selection else 1.0
            weight = "400"
        cx = cell.x + cell.width / 2
        cy = cell.y + cell.height / 2
        parts.append(
            f'<g opacity="{opacity:.2f}">'
            f'<rect x="{cell.x:.0f}" y="{cell.y:.0f}" '
            f'width="{cell.width:.0f}" height="{cell.height:.0f}" '
            f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{cx:.0f}" y="{cy:.0f}" text-anchor="middle" '
            f'dominant-baseline="central" font-size="15" font-weight="{weight}" '
            f'fill="{TEXT_COLOR}">{escape(cell.value)}</text>'
            f"</g>"
        )

    width = layout.width
    y = layout.height + LABEL_HEIGHT

    if label:
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
