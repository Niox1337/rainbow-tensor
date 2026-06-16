"""SVG rendering.

This module turns a :class:`~rainbow_tensor.layout.Layout` into an SVG
string. It draws the tensor blocks, cells, values, and a label.
"""

from .layout import build_layout

CELL_FILL = "#ffffff"
CELL_STROKE = "#94a3b8"
BLOCK_STROKE = "#cbd5e1"
TEXT_COLOR = "#0f172a"
LABEL_COLOR = "#334155"

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


def render_svg(shape, value_fn=None, label=""):
    """Render a tensor shape as an SVG string."""
    layout = build_layout(shape, value_fn=value_fn)
    parts = []

    for block in layout.blocks:
        parts.append(
            f'<rect x="{block.x:.0f}" y="{block.y:.0f}" '
            f'width="{block.width:.0f}" height="{block.height:.0f}" '
            f'rx="8" fill="none" stroke="{BLOCK_STROKE}" stroke-width="1.5"/>'
        )

    for cell in layout.cells:
        cx = cell.x + cell.width / 2
        cy = cell.y + cell.height / 2
        parts.append(
            f'<rect x="{cell.x:.0f}" y="{cell.y:.0f}" '
            f'width="{cell.width:.0f}" height="{cell.height:.0f}" '
            f'rx="6" fill="{CELL_FILL}" stroke="{CELL_STROKE}" stroke-width="1.5"/>'
            f'<text x="{cx:.0f}" y="{cy:.0f}" text-anchor="middle" '
            f'dominant-baseline="central" font-size="15" '
            f'fill="{TEXT_COLOR}">{escape(cell.value)}</text>'
        )

    width = layout.width
    y = layout.height + LABEL_HEIGHT
    if label:
        parts.append(
            f'<text x="20" y="{y:.0f}" font-size="16" font-weight="600" '
            f'fill="{LABEL_COLOR}">{escape(label)}</text>'
        )
        y += LINE_HEIGHT

    height = y + 16
    return svg_document("".join(parts), width, height)
