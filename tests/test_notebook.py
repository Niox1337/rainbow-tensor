"""Tests for the notebook display layer."""

import pytest

from rainbow_tensor import show_index, show_shape
from rainbow_tensor.notebook import TensorVisual


class FakeArray:
    def __init__(self, shape, values):
        self.shape = shape
        self._values = values

    def __getitem__(self, coord):
        return self._values[coord]


def test_show_shape_returns_visual_with_svg():
    visual = show_shape((2, 2, 2))
    assert isinstance(visual, TensorVisual)
    assert visual.svg.startswith("<svg")
    assert visual.shape == (2, 2, 2)


def test_show_shape_repr_svg_matches_svg():
    visual = show_shape((3,))
    assert visual._repr_svg_() == visual.svg
    assert "Shape (3)" in visual.svg


def test_show_shape_uses_array_values():
    array = FakeArray((3,), {(0,): 10, (1,): 20, (2,): 30})
    visual = show_shape(array)
    assert ">20<" in visual.svg


def test_show_index_computes_selection_and_result_shape():
    visual = show_index((2, 2, 2), (0, slice(None), 1))
    assert visual.selected == [(0, 0, 1), (0, 1, 1)]
    assert visual.result_shape == (2,)
    assert "Result shape: (2,)" in visual.svg


def test_show_index_highlights_selected_values():
    visual = show_index((2, 2, 2), (0, slice(None), 1))
    assert "#16a34a" in visual.svg
    assert visual.svg.startswith("<svg")


def test_show_index_uses_array_values():
    values = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                values[(i, j, k)] = (i * 4 + j * 2 + k) * 100
    array = FakeArray((2, 2, 2), values)
    visual = show_index(array, (0, slice(None), 1))
    assert ">100<" in visual.svg
    assert ">300<" in visual.svg


def test_show_index_with_numpy_array():
    np = pytest.importorskip("numpy")
    x = np.arange(8).reshape(2, 2, 2)
    visual = show_index(x, (0, slice(None), 1))
    assert ">1<" in visual.svg
    assert ">3<" in visual.svg
    assert visual.result_shape == (2,)
