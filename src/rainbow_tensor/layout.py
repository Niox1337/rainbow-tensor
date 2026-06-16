"""Layout calculation.

This module converts a tensor shape into 2D drawing coordinates. It knows
nothing about SVG, so the same layout could be rendered by any backend.
"""

from dataclasses import dataclass, field

from .shape import flat_index

CELL = 40
CELL_GAP = 6
ROW_GAP = 6
BLOCK_PAD = 12
BLOCK_GAP = 28
PADDING = 20


@dataclass
class Cell:
    """A single tensor element placed on the canvas."""

    x: float
    y: float
    width: float
    height: float
    value: object
    coord: tuple[int, ...]
    selected: bool = False


@dataclass
class Block:
    """A grouping rectangle around the rows of one axis-0 block."""

    x: float
    y: float
    width: float
    height: float
    index: int


@dataclass
class Layout:
    """All drawing data for one tensor."""

    cells: list[Cell] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


def _grid(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Return ``(blocks, rows, cols)`` for a 1D, 2D, or 3D shape."""
    ndim = len(shape)
    if ndim == 1:
        return 1, 1, shape[0]
    if ndim == 2:
        return 1, shape[0], shape[1]
    return shape[0], shape[1], shape[2]


def _coord(ndim: int, b: int, r: int, c: int) -> tuple[int, ...]:
    """Build the real tensor coordinate for a grid position."""
    if ndim == 1:
        return (c,)
    if ndim == 2:
        return (r, c)
    return (b, r, c)


def build_layout(shape, selected=None, value_fn=None) -> Layout:
    """Compute the layout for a tensor.

    ``selected`` is an iterable of coordinates to mark as selected.
    ``value_fn`` maps a coordinate to its display value. When it is ``None``
    sequential row-major values starting at 0 are used.
    """
    selected_set = {tuple(coord) for coord in (selected or [])}
    ndim = len(shape)
    blocks, rows, cols = _grid(shape)

    block_w = cols * CELL + (cols - 1) * CELL_GAP + 2 * BLOCK_PAD
    block_h = rows * CELL + (rows - 1) * ROW_GAP + 2 * BLOCK_PAD

    layout = Layout()
    for b in range(blocks):
        bx = PADDING + b * (block_w + BLOCK_GAP)
        by = PADDING
        layout.blocks.append(Block(bx, by, block_w, block_h, b))
        for r in range(rows):
            for c in range(cols):
                x = bx + BLOCK_PAD + c * (CELL + CELL_GAP)
                y = by + BLOCK_PAD + r * (CELL + ROW_GAP)
                coord = _coord(ndim, b, r, c)
                if value_fn is not None:
                    value = value_fn(coord)
                else:
                    value = flat_index(coord, shape)
                layout.cells.append(
                    Cell(
                        x=x,
                        y=y,
                        width=CELL,
                        height=CELL,
                        value=value,
                        coord=coord,
                        selected=coord in selected_set,
                    )
                )

    layout.width = PADDING * 2 + blocks * block_w + (blocks - 1) * BLOCK_GAP
    layout.height = PADDING * 2 + block_h
    return layout
