"""Scalar and empty math follows NumPy shapes without fictitious source reads."""

import tracemalloc
import warnings
from math import isnan

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.layout import build_layout
from rainbow_tensor.ops.einsum import einsum_selected_coords, parse_einsum_subscripts
from rainbow_tensor.tracing import _normalize_focus


class RecordingRenderer:
    """Evaluate logical result cells while leaving source panels unread."""

    name = "degenerate-math-results"
    mime_type = "text/plain"

    def render_tensor(self, **kwargs):
        raise AssertionError("math operations must render multiple panels")

    def render_panels(self, *, panels, theme, **kwargs):
        self.panels = panels
        result = panels[-1]
        layout = build_layout(
            result["shape"], selected=result.get("selected"),
            value_fn=result["value_fn"], theme=result.get("theme", theme),
        )
        self.values = {cell.coord: cell.value for cell in layout.cells if not cell.ellipsis}
        return "recorded math"


class CountingArray:
    """Expose arbitrary shapes and record only actual scalar accesses."""

    def __init__(self, shape, value=7):
        self.shape = shape
        self.value = value
        self.reads = []

    def __getitem__(self, coordinate):
        assert len(coordinate) == len(self.shape)
        assert all(0 <= value < size for value, size in zip(coordinate, self.shape))
        self.reads.append(coordinate)
        return self.value


def assert_result_values(renderer, expected):
    """Compare values including NaNs at the exact logical output coordinates."""
    assert set(renderer.values) == set(np.ndindex(expected.shape))
    for coordinate, value in renderer.values.items():
        if np.isnan(expected[coordinate]):
            assert isnan(value)
        else:
            assert value == expected[coordinate]


@pytest.mark.parametrize("operation, axis", [
    (rt.sum, None), (rt.sum, ()), (rt.sum, 0), (rt.sum, -1),
    (rt.sum, np.int64(0)), (rt.mean, None), (rt.mean, ()),
])
@pytest.mark.parametrize("keepdims", [False, True])
def test_scalar_reduction_axes_match_numpy(operation, axis, keepdims):
    array = CountingArray(())
    renderer = RecordingRenderer()
    visual = operation(array, axis=axis, keepdims=keepdims, focus=(), renderer=renderer)
    expected = getattr(np, operation.__name__)(np.array(7), axis=axis, keepdims=keepdims)

    assert visual.result_shape == ()
    assert renderer.panels[-1]["shape"] == ()
    assert renderer.panels[-1]["selected"] == [()]
    assert_result_values(renderer, expected)
    assert array.reads == [()]
    assert visual.trace.output_coord == ()
    assert visual.trace.terms[0][0].coordinate == ()
    assert visual.trace.term_count == visual.trace.divisor == 1


@pytest.mark.parametrize("operation, axis", [
    (rt.sum, (0,)), (rt.sum, (-1,)), (rt.sum, 1), (rt.sum, -2),
    (rt.mean, 0), (rt.mean, -1), (rt.mean, (0,)),
    (rt.sum, True), (rt.mean, np.bool_(False)),
])
def test_invalid_scalar_axes_fail_before_any_reads(operation, axis):
    array = CountingArray(())
    with pytest.raises((TypeError, ValueError), match="axis"):
        operation(array, axis=axis)
    assert array.reads == []


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("shape, axis", [((0,), None), ((2, 0, 3), 1), ((2, 0, 3), (0, 1))])
@pytest.mark.parametrize("keepdims", [False, True])
def test_empty_reduction_groups_produce_zero_or_nan_without_reads(operation, shape, axis, keepdims):
    array = CountingArray(shape)
    renderer = RecordingRenderer()
    visual = operation(array, axis=axis, keepdims=keepdims, renderer=renderer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = getattr(np, operation.__name__)(np.empty(shape), axis=axis, keepdims=keepdims)

    assert visual.result_shape == expected.shape
    assert_result_values(renderer, expected)
    assert array.reads == []
    assert visual.trace.complete and visual.trace.terms == ()
    assert visual.trace.term_count == 0
    assert visual.trace.divisor == (0 if operation is rt.mean else 1)
    assert visual.metadata["value_evaluation"]["total_terms"] == 0
    assert "NaN" in visual.text if operation is rt.mean else "0" in visual.text


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("axis", [1, ()])
def test_empty_output_has_no_focus_trace_or_computation(operation, axis):
    array = CountingArray((0, 3))
    renderer = RecordingRenderer()
    visual = operation(array, axis=axis, renderer=renderer, max_terms=1)
    expected = getattr(np, operation.__name__)(np.empty((0, 3)), axis=axis)

    assert visual.result_shape == expected.shape
    assert visual.trace is None
    assert not visual.selected
    assert renderer.values == {}
    assert not renderer.panels[-1]["selected"]
    assert array.reads == []
    evaluation = visual.metadata["value_evaluation"]
    assert evaluation["status"] == "evaluated"
    assert evaluation["output_count"] == evaluation["total_terms"] == 0
    assert "no output" in visual.text.lower()


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_empty_output_focus_is_absent_or_explicitly_rejected(shape):
    assert _normalize_focus(None, shape) is None
    with pytest.raises(IndexError, match="empty output"):
        _normalize_focus((0,) * len(shape), shape)


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_explicit_focus_on_empty_reduction_fails_before_reads(operation):
    array = CountingArray((0, 3))
    with pytest.raises(IndexError, match="empty output"):
        operation(array, axis=1, focus=(0,))
    assert array.reads == []


@pytest.mark.parametrize("shapes", [
    ((0,), (0,)), ((2, 0), (0, 3)), ((0, 2), (2, 3)),
    ((0, 2, 3), (1, 3, 4)), ((2, 0), (0,)),
])
def test_empty_matmul_matches_numpy_without_contraction_reads(shapes):
    arrays = [CountingArray(shape) for shape in shapes]
    renderer = RecordingRenderer()
    visual = rt.matmul(*arrays, renderer=renderer)
    expected = np.matmul(*(np.empty(shape) for shape in shapes))

    assert visual.result_shape == expected.shape
    assert_result_values(renderer, expected)
    assert all(array.reads == [] for array in arrays)
    assert all(not panel["selected"] for panel in renderer.panels[:-1])
    if expected.size:
        assert visual.trace.term_count == 0
        assert visual.trace.terms == () and visual.trace.complete
    else:
        assert visual.trace is None


@pytest.mark.parametrize("shapes", [((), (3,)), ((3,), ()), ((), ())])
def test_scalar_matmul_is_rejected_before_reads(shapes):
    arrays = [CountingArray(shape) for shape in shapes]
    with pytest.raises(ValueError, match="at least 1-D"):
        rt.matmul(*arrays)
    assert all(array.reads == [] for array in arrays)


@pytest.mark.parametrize("expression, shapes", [
    ("->", [()]), ("", [()]), (",->", [(), ()]), (",i->i", [(), (3,)]),
    (",i->i", [(), (0,)]), ("...,...->...", [(), (0,)]),
    ("i,i->i", [(0,), (1,)]), ("i,i->", [(0,), (1,)]),
    ("i,i->", [(1,), (0,)]), ("ii->", [(0, 0)]), ("ii->i", [(0, 0)]),
    ("ij,jk->ik", [(2, 0), (0, 3)]), ("ij,jk->ik", [(0, 2), (2, 3)]),
])
def test_scalar_and_empty_einsum_matches_numpy(expression, shapes):
    arrays = [np.full(shape, 2) for shape in shapes]
    renderer = RecordingRenderer()
    visual = rt.einsum(expression, *arrays, renderer=renderer)
    expected = np.einsum(expression, *arrays)

    assert visual.result_shape == expected.shape
    assert_result_values(renderer, expected)
    if expected.size == 0:
        assert visual.trace is None
        assert all(not selection for selection in visual.selected)
    elif any(0 in shape for shape in shapes):
        assert visual.trace.term_count == 0
        assert all(not selection for selection in visual.selected)


def test_empty_einsum_does_not_pool_an_unrelated_million_element_axis():
    expression, shapes = "ij,k->i", [(2, 0), (1_000_000,)]
    arrays = [CountingArray(shape) for shape in shapes]
    renderer = RecordingRenderer()
    tracemalloc.start()
    try:
        visual = rt.einsum(expression, *arrays, renderer=renderer)
        inputs, output = parse_einsum_subscripts(expression, 2, shapes)
        assert einsum_selected_coords(inputs, output, shapes) == [[], []]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert renderer.values == {(0,): 0, (1,): 0}
    assert visual.trace.term_count == 0
    assert all(array.reads == [] for array in arrays)
    assert peak < 1_000_000


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_zero_term_focus_equation_never_displays_division_by_zero(operation):
    visual = operation((0,), focus=(), renderer=RecordingRenderer())
    assert " / 0" not in visual.text
    assert "NaN" in visual.text if operation is rt.mean else "output[()] = 0" in visual.text
