"""Layout calculation.

This module converts a tensor shape into 2D drawing coordinates. It knows
nothing about SVG, so the same layout could be rendered by any backend.

The tensor is drawn as nested per-axis frames. Axis 0 is the outer frame,
each following non-leaf axis is an inner frame, and the leaf axis elements
are placed as cells inside the innermost frame. Geometry such as cell size
and padding comes from the active :class:`~rainbow_tensor.theme.Theme`, so the
same shape can render compact or roomy without touching this module.

A long axis is truncated. When an axis is longer than the theme limit only a
head and a tail of positions are drawn, with a single ellipsis cell standing
in for the hidden middle, so a wide tensor stays readable instead of
overflowing the canvas.
"""

from dataclasses import dataclass, field

from .shape import flat_index
from .theme import LIGHT

# A sentinel marking a hidden run of positions along an axis.
ELLIPSIS = object()

# The glyph drawn for the gap, chosen to match the axis orientation.
COL_ELLIPSIS = "…"  # horizontal ellipsis for a hidden column run
ROW_ELLIPSIS = "⋮"  # vertical ellipsis for a hidden row run
BLOCK_ELLIPSIS = "⋯"  # midline ellipsis for a hidden block run


@dataclass
class Cell:
    """A single tensor element placed on the canvas.

    An ellipsis cell stands in for a hidden run of elements. It carries no
    coordinate and is never selected.
    """

    x: float
    y: float
    width: float
    height: float
    value: object
    coord: object
    selected: bool = False
    ellipsis: bool = False
    flat: object = None


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


def visible_positions(size, limit):
    """Return the positions drawn along an axis of ``size``.

    When ``size`` fits within ``limit`` every position is returned. Otherwise
    a head and a tail are kept with an :data:`ELLIPSIS` sentinel marking the
    hidden middle, so the drawn length never exceeds ``limit``.
    """
    if limit is None or size <= limit:
        return list(range(size))
    keep = limit - 1
    head = (keep + 1) // 2
    tail = keep // 2
    return list(range(head)) + [ELLIPSIS] + list(range(size - tail, size))


def build_layout(shape, selected=None, value_fn=None, theme=None):
    """Compute the layout for a tensor.

    ``selected`` is an iterable of coordinates to mark as selected.
    ``value_fn`` maps a coordinate to its display value. When it is ``None``
    sequential row-major values starting at 0 are used. ``theme`` supplies the
    geometry and the truncation limit, defaulting to the light preset.
    """
    t = theme or LIGHT
    sel = {tuple(coord) for coord in (selected or [])}

    cell_w, cell_h, gap = t.cell_w, t.cell_h, t.cell_gap
    row_pad, row_gap = t.row_pad, t.row_gap
    block_pad, block_gap, pad = t.block_pad, t.block_gap, t.padding
    limit = t.max_cells

    cols = shape[-1]
    col_pos = visible_positions(cols, limit)
    n_cols = len(col_pos)

    row_w = n_cols * cell_w + (n_cols - 1) * gap + 2 * row_pad
    row_h = cell_h + 2 * row_pad

    layout = Layout()

    def place_cells(rx, ry, prefix):
        for i, c in enumerate(col_pos):
            x = rx + row_pad + i * (cell_w + gap)
            y = ry + row_pad
            if c is ELLIPSIS:
                layout.cells.append(
                    Cell(x, y, cell_w, cell_h, COL_ELLIPSIS, None, ellipsis=True)
                )
                continue
            coord = prefix + (c,)
            value = value_fn(coord) if value_fn is not None else flat_index(coord, shape)
            layout.cells.append(
                Cell(
                    x,
                    y,
                    cell_w,
                    cell_h,
                    value,
                    coord,
                    selected=coord in sel,
                    flat=flat_index(coord, shape),
                )
            )

    def place_gap_cell(rx, ry, glyph):
        # A single centred cell standing in for a hidden row or block run.
        x = rx + (row_w - cell_w) / 2
        layout.cells.append(
            Cell(x, ry + row_pad, cell_w, cell_h, glyph, None, ellipsis=True)
        )

    ndim = len(shape)

    if ndim == 1:
        rx, ry = pad, pad
        layout.frames.append(Frame(rx, ry, row_w, row_h, axis=None, selected=bool(sel)))
        place_cells(rx, ry, ())
        layout.width = pad * 2 + row_w
        layout.height = pad * 2 + row_h
        return layout

    if ndim == 2:
        row_positions = visible_positions(shape[0], limit)
        for i, r in enumerate(row_positions):
            rx = pad
            ry = pad + i * (row_h + row_gap)
            if r is ELLIPSIS:
                layout.frames.append(Frame(rx, ry, row_w, row_h, axis=None))
                place_gap_cell(rx, ry, ROW_ELLIPSIS)
                continue
            row_sel = any(co[0] == r for co in sel)
            layout.frames.append(Frame(rx, ry, row_w, row_h, axis=0, selected=row_sel))
            place_cells(rx, ry, (r,))
        n_rows = len(row_positions)
        layout.width = pad * 2 + row_w
        layout.height = pad * 2 + n_rows * row_h + (n_rows - 1) * row_gap
        return layout

    block_positions = visible_positions(shape[0], limit)
    row_positions = visible_positions(shape[1], limit)
    n_blocks, n_rows = len(block_positions), len(row_positions)
    block_w = row_w + 2 * block_pad
    block_h = n_rows * row_h + (n_rows - 1) * row_gap + 2 * block_pad

    for bi, b in enumerate(block_positions):
        bx = pad + bi * (block_w + block_gap)
        by = pad
        if b is ELLIPSIS:
            layout.frames.append(Frame(bx, by, block_w, block_h, axis=None))
            cx = bx + (block_w - cell_w) / 2
            cy = by + (block_h - cell_h) / 2
            layout.cells.append(
                Cell(cx, cy, cell_w, cell_h, BLOCK_ELLIPSIS, None, ellipsis=True)
            )
            continue
        block_sel = any(co[0] == b for co in sel)
        layout.frames.append(Frame(bx, by, block_w, block_h, axis=0, selected=block_sel))
        for ri, r in enumerate(row_positions):
            rx = bx + block_pad
            ry = by + block_pad + ri * (row_h + row_gap)
            if r is ELLIPSIS:
                layout.frames.append(Frame(rx, ry, row_w, row_h, axis=None))
                place_gap_cell(rx, ry, ROW_ELLIPSIS)
                continue
            row_sel = any(co[0] == b and co[1] == r for co in sel)
            layout.frames.append(Frame(rx, ry, row_w, row_h, axis=1, selected=row_sel))
            place_cells(rx, ry, (b, r))

    layout.width = pad * 2 + n_blocks * block_w + (n_blocks - 1) * block_gap
    layout.height = pad * 2 + block_h
    return layout
