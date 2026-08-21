"""Multi-axis reduction shapes and ordered source terms agree with NumPy."""

import tracemalloc
from itertools import islice
from math import prod

import numpy as np
import pytest

from rainbow_tensor.ops import (
    normalize_reduction_axes,
    reduce_result_shape,
    reduce_source_coords,
    reduce_source_index,
    reduce_term_count,
    validate_keepdims,
)
from rainbow_tensor.selection import BasicSelection
from rainbow_tensor.tracing import _build_trace


class IndexValue:
    """Represent an axis using only Python's integer index protocol."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


REDUCTION_CASES = [
    ((4,), None, (0,)),
    ((4,), (), ()),
    ((2, 3, 4), None, (0, 1, 2)),
    ((2, 3, 4), 0, (0,)),
    ((2, 3, 4), 1, (1,)),
    ((2, 3, 4), -1, (2,)),
    ((2, 3, 4), (), ()),
    ((2, 3, 4), (0, 2), (0, 2)),
    ((2, 3, 4), (2, 0), (0, 2)),
    ((2, 3, 4), (-1, -3), (0, 2)),
    ((2, 3, 4), (1,), (1,)),
    ((2, 3, 4), (2, 0, 1), (0, 1, 2)),
    ((1, 3, 1), (2, 0), (0, 2)),
]


@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("shape, axis, normalized", REDUCTION_CASES)
def test_reduction_shapes_values_and_sources_match_numpy(shape, axis, normalized, keepdims):
    array = np.arange(prod(shape)).reshape(shape)
    expected_sum = np.sum(array, axis=axis, keepdims=keepdims)
    expected_mean = np.mean(array, axis=axis, keepdims=keepdims)
    result = reduce_result_shape(shape, axis, keepdims=keepdims)

    # Reduce one-hot inputs to identify every contributing source independently
    # of this package's coordinate mapping. Flat source order defines the trace.
    one_hot = np.eye(array.size, dtype=int).reshape((array.size,) + shape)
    contributions = np.sum(
        one_hot, axis=tuple(a + 1 for a in normalized), keepdims=keepdims
    )
    assert normalize_reduction_axes(axis, len(shape)) == normalized
    assert result == expected_sum.shape == expected_mean.shape
    assert reduce_term_count(shape, axis) == prod(shape[a] for a in normalized)

    for coordinate in np.ndindex(result):
        terms = list(reduce_source_coords(coordinate, shape, axis, keepdims=keepdims))
        expected_sources = [
            np.unravel_index(position, shape)
            for position in np.flatnonzero(contributions[(slice(None),) + coordinate])
        ]
        assert terms == expected_sources
        assert sum(array[source] for source in terms) == expected_sum[coordinate]
        assert sum(array[source] for source in terms) / len(terms) == expected_mean[coordinate]
        source_index = reduce_source_index(coordinate, shape, axis, keepdims=keepdims)
        assert list(BasicSelection(shape, source_index)) == terms


def test_default_axis_reduces_every_dimension():
    assert reduce_result_shape((2, 3)) == ()
    assert reduce_term_count((2, 3)) == 6
    assert reduce_source_index((), (2, 3)) == (slice(None), slice(None))
    assert list(reduce_source_coords((), (2, 3))) == list(np.ndindex(2, 3))


@pytest.mark.parametrize("keepdims", [False, True])
def test_empty_axis_tuple_preserves_shape_and_one_source_term(keepdims):
    shape = (2, 3, 4)
    coordinate = (1, 2, 3)
    assert reduce_result_shape(shape, (), keepdims=keepdims) == shape
    assert reduce_term_count(shape, ()) == 1
    assert reduce_source_index(coordinate, shape, (), keepdims=keepdims) == coordinate
    assert list(reduce_source_coords(coordinate, shape, (), keepdims=keepdims)) == [coordinate]


@pytest.mark.parametrize("axis, expected", [
    (np.int64(-1), (2,)),
    (np.uint64(1), (1,)),
    (IndexValue(-2), (1,)),
    ((np.uint64(0), IndexValue(-1)), (0, 2)),
])
def test_axes_follow_the_integer_index_protocol(axis, expected):
    resolved = normalize_reduction_axes(axis, 3)
    assert resolved == expected
    assert all(type(value) is int for value in resolved)
    assert reduce_term_count((2, 3, 4), axis) == prod((2, 3, 4)[a] for a in expected)


@pytest.mark.parametrize("axis", [
    True, np.bool_(False), 1.0, np.float64(1.0), "1", [0, 1],
    np.array([0, 1]), iter([0, 1]), (0, True), (0, np.bool_(True)), (0, 1.0),
])
def test_non_integer_and_non_tuple_axes_are_rejected(axis):
    with pytest.raises(TypeError, match="axis"):
        normalize_reduction_axes(axis, 3)


@pytest.mark.parametrize("axis", [(0, 0), (0, -3), (-1, 2), (IndexValue(1), 1)])
def test_duplicate_axes_are_rejected_after_negative_normalization(axis):
    with pytest.raises(ValueError, match="repeat"):
        normalize_reduction_axes(axis, 3)


@pytest.mark.parametrize("axis", [3, -4, (0, 3), (-4, 1)])
def test_out_of_range_axes_are_rejected(axis):
    with pytest.raises(ValueError, match="out of range"):
        normalize_reduction_axes(axis, 3)


@pytest.mark.parametrize("keepdims, expected", [
    (False, False), (True, True), (np.bool_(False), False), (np.bool_(True), True),
])
def test_keepdims_accepts_boolean_scalars(keepdims, expected):
    assert validate_keepdims(keepdims) is expected
    assert reduce_result_shape((2, 3), 0, keepdims=keepdims) == ((1, 3) if expected else (3,))


@pytest.mark.parametrize("keepdims", [None, 0, 1, np.int64(1), 1.0, "yes", []])
def test_keepdims_rejects_non_boolean_values(keepdims):
    with pytest.raises(TypeError, match="keepdims"):
        reduce_result_shape((2, 3), 0, keepdims=keepdims)
    with pytest.raises(TypeError, match="keepdims"):
        reduce_source_index((1,), (2, 3), 0, keepdims=keepdims)
    with pytest.raises(TypeError, match="keepdims"):
        next(reduce_source_coords((1,), (2, 3), 0, keepdims=keepdims))


def test_reordered_axes_keep_source_row_major_trace_order():
    expected = [(i, 1, k) for i in range(2) for k in range(4)]
    for axes in ((0, 2), (2, 0), (-1, -3)):
        assert list(reduce_source_coords((1,), (2, 3, 4), axes)) == expected
        assert list(reduce_source_coords((0, 1, 0), (2, 3, 4), axes, keepdims=True)) == expected


def test_trillion_term_trace_and_source_selection_stay_lazy():
    shape = (1_000_000, 3, 1_000_000)
    axes = (2, 0)
    tracemalloc.start()
    try:
        count = reduce_term_count(shape, axes)
        terms = ((coordinate,) for coordinate in reduce_source_coords((1,), shape, axes))
        trace = _build_trace("mean", (1,), count, terms, divisor=count)
        selected = BasicSelection(shape, reduce_source_index((1,), shape, axes))
        assert selected.count == count
        assert (999_999, 1, 999_999) in selected
        assert selected.matches_prefix((500_000, 1))
        assert not selected.matches_prefix((500_000, 2))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert count == trace.divisor == 10**12
    assert [term[0].coordinate for term in trace.terms] == [(0, 1, k) for k in range(8)]
    assert not trace.complete
    assert peak < 1_000_000


def test_odometer_carries_between_reduced_axes_without_pooling_ranges():
    terms = reduce_source_coords((1,), (1_000_000, 3, 2), (2, 0))
    assert list(islice(terms, 5)) == [
        (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 0),
    ]
