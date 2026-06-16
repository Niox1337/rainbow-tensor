"""Tests for layout calculation."""

from rainbow_tensor.layout import build_layout


def test_one_dimensional_layout_cell_count():
    layout = build_layout((3,))
    assert len(layout.cells) == 3


def test_two_dimensional_layout_cell_count():
    layout = build_layout((2, 3))
    assert len(layout.cells) == 6


def test_three_dimensional_layout_structure():
    layout = build_layout((2, 2, 2))
    assert len(layout.blocks) == 2
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


def test_selected_coordinates_are_marked():
    layout = build_layout((2, 2, 2), selected=[(0, 0, 1)])
    selected = [cell.coord for cell in layout.cells if cell.selected]
    assert selected == [(0, 0, 1)]
