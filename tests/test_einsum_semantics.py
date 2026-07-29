"""Compare broadcast shapes, rendered cell values, and selections with NumPy."""

import numpy as np
import pytest

from rainbow_tensor import einsum
from rainbow_tensor.layout import build_layout
from rainbow_tensor.ops import einsum_result_shape


class ResultCellRenderer:
    """Capture the output cells that a renderer would display."""

    name = "result-cells"
    mime_type = "text/plain"

    def render_tensor(self, **kwargs):
        raise AssertionError("einsum should render panels")

    def render_panels(self, *, panels, theme, **kwargs):
        output = panels[-1]
        layout = build_layout(
            output["shape"],
            selected=output.get("selected"),
            value_fn=output["value_fn"],
            theme=output.get("theme", theme),
        )
        self.values = {cell.coord: cell.value for cell in layout.cells if not cell.ellipsis}
        return "captured output cells"


@pytest.mark.parametrize(
    "subscripts, shapes",
    [
        ("ij,ij->ij", [(2, 1), (1, 3)]),
        ("ij,ij->ij", [(1, 3), (2, 1)]),
        ("ij,jk->ik", [(2, 1), (3, 2)]),
        ("ij,jk->ik", [(2, 3), (1, 2)]),
        ("i,i->", [(1,), (4,)]),
        ("ii,i->i", [(1, 1), (3,)]),
        ("i,ii->", [(3,), (1, 1)]),
        ("...ij,...jk->...ik", [(2, 1, 2, 3), (4, 3, 2)]),
        ("...ij,...jk->...ik", [(4, 2, 3), (2, 1, 3, 2)]),
        ("...ij,...jk", [(2, 1, 2, 3), (4, 3, 2)]),
        ("...ij,...jk->...ik", [(2, 3), (2, 1, 3, 2)]),
        ("a...b,b...c->a...c", [(2, 2, 1, 3), (3, 4, 2)]),
    ],
)
def test_einsum_broadcast_shape_and_rendered_values_match_numpy(subscripts, shapes):
    """Singleton axes broadcast without changing the values of visible output cells."""
    arrays = [
        np.arange(np.prod(shape)).reshape(shape) + operand + 1
        for operand, shape in enumerate(shapes)
    ]
    expected = np.einsum(subscripts, *arrays)
    renderer = ResultCellRenderer()

    visual = einsum(subscripts, *arrays, renderer=renderer)

    assert einsum_result_shape(subscripts, shapes) == expected.shape
    assert visual.result_shape == expected.shape
    expected_cells = {
        coord if expected.shape else (0,): expected[coord]
        for coord in np.ndindex(expected.shape)
    }
    assert renderer.values == expected_cells
    for shape, selected in zip(shapes, visual.selected):
        assert all(
            len(coord) == len(shape) and all(0 <= index < size for index, size in zip(coord, shape))
            for coord in selected
        )


@pytest.mark.parametrize(
    "subscripts, shapes",
    [
        ("ii->i", [(1, 3)]),
        ("ii->i", [(3, 1)]),
        ("i,ii->i", [(3,), (1, 3)]),
        ("...ii,...i->...i", [(2, 1, 3), (1, 3)]),
    ],
)
def test_einsum_repeated_labels_require_equal_sizes_within_each_operand(subscripts, shapes):
    """A diagonal cannot broadcast its axes even when another operand can broadcast."""
    arrays = [np.ones(shape) for shape in shapes]

    with pytest.raises(ValueError):
        np.einsum(subscripts, *arrays)
    with pytest.raises(ValueError, match="inconsistent sizes"):
        einsum_result_shape(subscripts, shapes)
    with pytest.raises(ValueError, match="inconsistent sizes"):
        einsum(subscripts, *arrays)


@pytest.mark.parametrize(
    "shapes, expected_selected",
    [
        ([(2, 1), (3, 2)], [[(0, 0)], [(0, 0), (1, 0), (2, 0)]]),
        ([(2, 3), (1, 2)], [[(0, 0), (0, 1), (0, 2)], [(0, 0)]]),
    ],
)
def test_einsum_contracted_singleton_selects_only_existing_source_cells(shapes, expected_selected):
    """Broadcast contraction terms reuse coordinate zero in the singleton operand."""
    arrays = [np.ones(shape) for shape in shapes]

    visual = einsum("ij,jk->ik", *arrays)

    assert visual.selected == expected_selected
