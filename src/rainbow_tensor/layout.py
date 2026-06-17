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
    """Compute the layout for a tensor of any rank.

    ``selected`` is an iterable of coordinates to mark as selected.
    ``value_fn`` maps a coordinate to its display value. When it is ``None``
    sequential row-major values starting at 0 are used. ``theme`` supplies the
    geometry and the truncation limit, defaulting to the light preset.

    The leaf axis is a horizontal row of cells. Every axis above it draws a
    frame, alternating horizontal and vertical with depth so a deep tensor
    stays compact. Each axis truncates independently through
    :func:`visible_positions`, so a large N-D tensor still fits on screen.
    """
    t = theme or LIGHT
    sel = {tuple(coord) for coord in (selected or [])}

    cell_w, cell_h, gap = t.cell_w, t.cell_h, t.cell_gap
    row_pad, row_gap = t.row_pad, t.row_gap
    block_pad, block_gap, pad = t.block_pad, t.block_gap, t.padding
    limit = t.max_cells

    col_pos = visible_positions(shape[-1], limit)
    n_cols = len(col_pos)
    row_w = n_cols * cell_w + (n_cols - 1) * gap + 2 * row_pad
    row_h = cell_h + 2 * row_pad

    ndim = len(shape)
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

    # 1D is a single neutral frame around one row of cells.
    if ndim == 1:
        layout.frames.append(Frame(pad, pad, row_w, row_h, axis=None, selected=bool(sel)))
        place_cells(pad, pad, ())
        layout.width = pad * 2 + row_w
        layout.height = pad * 2 + row_h
        return layout

    # Depth from the leaf decides orientation: even axes lay out horizontally,
    # odd axes vertically. For 3D this is blocks across, rows down, cells across,
    # matching the original hand-written layout.
    def horizontal(axis):
        return (ndim - 1 - axis) % 2 == 0

    def group_size(axis):
        """Size of the region drawn for one fixed value of ``axis - 1``."""
        n = len(visible_positions(shape[axis], limit))
        if axis == ndim - 2:
            tile_w, tile_h = row_w, row_h
        else:
            iw, ih = group_size(axis + 1)
            tile_w, tile_h = iw + 2 * block_pad, ih + 2 * block_pad
        if horizontal(axis):
            return n * tile_w + (n - 1) * block_gap, tile_h
        return tile_w, n * tile_h + (n - 1) * row_gap

    def render(axis, prefix, ox, oy):
        """Draw ``axis`` and every inner axis for ``prefix``; return (w, h)."""
        positions = visible_positions(shape[axis], limit)
        is_row = axis == ndim - 2  # children are leaf rows of cells
        inner_pad = 0 if is_row else block_pad
        content_w, content_h = (row_w, row_h) if is_row else group_size(axis + 1)
        tile_w, tile_h = content_w + 2 * inner_pad, content_h + 2 * inner_pad
        horiz = horizontal(axis)
        step = (tile_w + block_gap) if horiz else (tile_h + row_gap)

        cursor = ox if horiz else oy
        for p in positions:
            tx = cursor if horiz else ox
            ty = oy if horiz else cursor
            if p is ELLIPSIS:
                layout.frames.append(Frame(tx, ty, tile_w, tile_h, axis=None))
                glyph = BLOCK_ELLIPSIS if horiz else ROW_ELLIPSIS
                gx = tx + (tile_w - cell_w) / 2
                gy = ty + (tile_h - cell_h) / 2
                layout.cells.append(Cell(gx, gy, cell_w, cell_h, glyph, None, ellipsis=True))
            else:
                key = prefix + (p,)
                frame_sel = any(co[: axis + 1] == key for co in sel)
                layout.frames.append(
                    Frame(tx, ty, tile_w, tile_h, axis=axis, selected=frame_sel)
                )
                if is_row:
                    place_cells(tx, ty, key)
                else:
                    render(axis + 1, key, tx + inner_pad, ty + inner_pad)
            cursor += step

        n = len(positions)
        if horiz:
            return n * tile_w + (n - 1) * block_gap, tile_h
        return tile_w, n * tile_h + (n - 1) * row_gap

    w, h = render(0, (), pad, pad)
    layout.width = pad * 2 + w
    layout.height = pad * 2 + h
    return layout
