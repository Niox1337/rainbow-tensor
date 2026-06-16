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


def test_show_index_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        show_index((2, 2, 2), (0, slice(None), 1))
