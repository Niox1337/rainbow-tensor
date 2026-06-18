"""Tests for layout calculation."""

from rainbow_tensor.layout import ELLIPSIS, axis_visible_limits, build_layout, visible_positions
from rainbow_tensor.theme import LIGHT


def test_one_dimensional_layout_cell_count():
    layout = build_layout((3,))
    assert len(layout.cells) == 3
    assert len(layout.frames) == 1
    assert layout.frames[0].axis is None


def test_two_dimensional_layout_cell_count():
    layout = build_layout((2, 3))
    assert len(layout.cells) == 6
    assert len(layout.frames) == 2
    assert all(frame.axis == 0 for frame in layout.frames)


def test_three_dimensional_layout_structure():
    layout = build_layout((2, 2, 2))
    block_frames = [f for f in layout.frames if f.axis == 0]
    row_frames = [f for f in layout.frames if f.axis == 1]
    assert len(block_frames) == 2
    assert len(row_frames) == 4
    assert len(layout.cells) == 8
    by_value = {cell.value: cell.coord for cell in layout.cells}
    assert by_value[0] == (0, 0, 0)
    assert by_value[1] == (0, 0, 1)
    assert by_value[3] == (0, 1, 1)
    assert by_value[4] == (1, 0, 0)
    assert by_value[7] == (1, 1, 1)


def test_cells_do_not_overlap():
    layout = build_layout((2, 3))
    boxes = [(c.x, c.y, c.width, c.height) for c in layout.cells]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax, ay, aw, ah = boxes[i]
            bx, by, bw, bh = boxes[j]
            separated = ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay
            assert separated


def test_value_fn_overrides_generated_values():
    layout = build_layout((3,), value_fn=lambda coord: coord[0] * 10)
    values = sorted(cell.value for cell in layout.cells)
    assert values == [0, 10, 20]


def test_selected_block_and_row_frames_are_marked():
    layout = build_layout((2, 2, 2), selected=[(0, 0, 1), (0, 1, 1)])
    block_frames = [f for f in layout.frames if f.axis == 0]
    assert block_frames[0].selected is True
    assert block_frames[1].selected is False
    selected_cells = [cell.coord for cell in layout.cells if cell.selected]
    assert selected_cells == [(0, 0, 1), (0, 1, 1)]


def test_big_tensor_limits_visible_cells():
    limits = axis_visible_limits((100, 100, 100, 100), LIGHT.max_cells, 240)
    layout = build_layout((100, 100, 100, 100))
    real_cells = [cell for cell in layout.cells if not cell.ellipsis]

    assert limits == (3, 4, 4, 4)
    assert len(real_cells) <= LIGHT.max_visible_cells


def test_visible_positions_pin_selected_middle_value():
    positions = visible_positions(100, 6, pinned={50})

    assert positions == [0, 1, ELLIPSIS, 50, ELLIPSIS, 99]


def test_selected_middle_value_renders_in_large_axis():
    layout = build_layout((100,), selected=[(50,)])
    selected_cells = [cell for cell in layout.cells if cell.selected]

    assert [cell.coord for cell in selected_cells] == [(50,)]
    assert selected_cells[0].value == 50
