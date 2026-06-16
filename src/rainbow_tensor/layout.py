"""Layout calculation.

This module converts a tensor shape into 2D drawing coordinates. It knows
nothing about SVG, so the same layout could be rendered by any backend.

The tensor is drawn as nested per-axis frames. Axis 0 is the outer frame,
each following non-leaf axis is an inner frame, and the leaf axis elements
are placed as plain text cells inside the innermost frame.
"""

from dataclasses import dataclass, field

from .shape import flat_index

CELL_W = 48
CELL_H = 34
CELL_GAP = 8
ROW_PAD = 9
ROW_GAP = 12
BLOCK_PAD = 12
BLOCK_GAP = 30
PADDING = 20


@dataclass
class Cell:
    """A single tensor element placed on the canvas."""

    x: float
    y: float
    width: float
    height: float
    value: object
    coord: tuple
    selected: bool = False


@dataclass
class Frame:
    """A grouping rectangle for one axis.

    ``axis`` is the axis this frame represents, or ``None`` for a neutral
    container that is not tied to a specific axis (used for 1D tensors).
    ``selected`` is true when the region contains at least one selected cell.
    """

    x: float
    y: float
    width: float
    height: float
    axis: object
    selected: bool = False


@dataclass
class Layout:
    """All drawing data for one tensor."""

    cells: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


def build_layout(shape, selected=None, value_fn=None):
    """Compute the layout for a tensor.

    ``selected`` is an iterable of coordinates to mark as selected.
    ``value_fn`` maps a coordinate to its display value. When it is ``None``
    sequential row-major values starting at 0 are used.
    """
    sel = {tuple(coord) for coord in (selected or [])}
    ndim = len(shape)
    cols = shape[-1]

    row_w = cols * CELL_W + (cols - 1) * CELL_GAP + 2 * ROW_PAD
    row_h = CELL_H + 2 * ROW_PAD

    layout = Layout()

    def place_cells(rx, ry, prefix):
        for c in range(cols):
            x = rx + ROW_PAD + c * (CELL_W + CELL_GAP)
            coord = prefix + (c,)
            value = value_fn(coord) if value_fn is not None else flat_index(coord, shape)
            layout.cells.append(
                Cell(x, ry + ROW_PAD, CELL_W, CELL_H, value, coord, coord in sel)
            )

    if ndim == 1:
        rx, ry = PADDING, PADDING
        layout.frames.append(Frame(rx, ry, row_w, row_h, axis=None, selected=bool(sel)))
        place_cells(rx, ry, ())
        layout.width = PADDING * 2 + row_w
        layout.height = PADDING * 2 + row_h
        return layout

    if ndim == 2:
        rows = shape[0]
        for r in range(rows):
            rx = PADDING
            ry = PADDING + r * (row_h + ROW_GAP)
            row_sel = any(co[0] == r for co in sel)
            layout.frames.append(Frame(rx, ry, row_w, row_h, axis=0, selected=row_sel))
            place_cells(rx, ry, (r,))
        layout.width = PADDING * 2 + row_w
        layout.height = PADDING * 2 + rows * row_h + (rows - 1) * ROW_GAP
        return layout

    blocks, rows, _ = shape
    block_w = row_w + 2 * BLOCK_PAD
    block_h = rows * row_h + (rows - 1) * ROW_GAP + 2 * BLOCK_PAD
    for b in range(blocks):
        bx = PADDING + b * (block_w + BLOCK_GAP)
        by = PADDING
        block_sel = any(co[0] == b for co in sel)
        layout.frames.append(Frame(bx, by, block_w, block_h, axis=0, selected=block_sel))
        for r in range(rows):
            rx = bx + BLOCK_PAD
            ry = by + BLOCK_PAD + r * (row_h + ROW_GAP)
            row_sel = any(co[0] == b and co[1] == r for co in sel)
            layout.frames.append(Frame(rx, ry, row_w, row_h, axis=1, selected=row_sel))
            place_cells(rx, ry, (b, r))
    layout.width = PADDING * 2 + blocks * block_w + (blocks - 1) * BLOCK_GAP
    layout.height = PADDING * 2 + block_h
    return layout
