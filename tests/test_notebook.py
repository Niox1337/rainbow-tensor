"""Tests for the notebook display layer."""

import pytest

from rainbow_tensor import index, shape
from rainbow_tensor.notebook import TensorVisual


class FakeArray:
    def __init__(self, shape, values):
        self.shape = shape
        self._values = values

    def __getitem__(self, coord):
        return self._values[coord]


def test_shape_returns_visual_with_svg():
    visual = shape((2, 2, 2))
    assert isinstance(visual, TensorVisual)
    assert visual.svg.startswith("<svg")
    assert visual.shape == (2, 2, 2)


def test_shape_repr_svg_matches_svg():
    visual = shape((3,))
    assert visual._repr_svg_() == visual.svg
    assert "Shape (" in visual.svg


def test_show_functions_do_not_call_display(monkeypatch):
    # The result renders once through _repr_svg_, so the functions must not
    # call display themselves, otherwise a notebook cell shows it twice.
    import sys

    calls = []
    fake_module = type(sys)("IPython.display")
    fake_module.display = lambda *args, **kwargs: calls.append(args)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_module)

    shape((2, 2, 2))
    index((2, 2, 2), (0, slice(None), 1))
    assert calls == []


def test_shape_label_colours_frame_axes():
    visual = shape((2, 2, 2))
    # axis 0 red and axis 1 orange appear both as frames and as label numbers
    assert "#dc2626" in visual.svg
    assert "#ea580c" in visual.svg
    assert "<tspan" in visual.svg


def test_shape_uses_array_values():
    array = FakeArray((3,), {(0,): 10, (1,): 20, (2,): 30})
    visual = shape(array)
    assert ">20<" in visual.svg


def test_index_computes_selection_and_result_shape():
    visual = index((2, 2, 2), (0, slice(None), 1))
    assert visual.selected == [(0, 0, 1), (0, 1, 1)]
    assert visual.result_shape == (2,)
    assert "Result shape: (2,)" in visual.svg


def test_index_highlights_selected_values():
    visual = index((2, 2, 2), (0, slice(None), 1))
    assert "#16a34a" in visual.svg
    assert visual.svg.startswith("<svg")


def test_index_label_colours_tokens_by_axis():
    visual = index((2, 2, 2), (0, slice(None), 1))
    # red axis 0 token, orange axis 1 token, green leaf token, all as tspans
    assert '<tspan fill="#dc2626">0</tspan>' in visual.svg
    assert '<tspan fill="#ea580c">:</tspan>' in visual.svg
    assert '<tspan fill="#16a34a">1</tspan>' in visual.svg


def test_index_uses_array_values():
    values = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                values[(i, j, k)] = (i * 4 + j * 2 + k) * 100
    array = FakeArray((2, 2, 2), values)
    visual = index(array, (0, slice(None), 1))
    assert ">100<" in visual.svg
    assert ">300<" in visual.svg


def test_index_with_numpy_array():
    np = pytest.importorskip("numpy")
    x = np.arange(8).reshape(2, 2, 2)
    visual = index(x, (0, slice(None), 1))
    assert ">1<" in visual.svg
    assert ">3<" in visual.svg
    assert visual.result_shape == (2,)
