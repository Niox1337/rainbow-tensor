"""Repeat provenance follows NumPy without materialising its repeated output."""

import itertools
import sys
import tracemalloc
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rainbow_tensor.ops import repeat_source_coord, repeat_source_lookup


@pytest.mark.parametrize("axis_size", range(5))
def test_variable_counts_match_numpy_including_zero_runs(axis_size):
    """Repeated cumulative ends skip every zero-length source interval."""
    for counts in itertools.product(range(4), repeat=axis_size):
        expected = np.repeat(np.arange(axis_size), counts)
        lookup = repeat_source_lookup(axis_size, counts)
        assert lookup.count == len(expected)
        assert len(lookup) == len(expected)
        assert [lookup[i] for i in range(lookup.count)] == expected.tolist()
        assert [lookup[-i - 1] for i in range(lookup.count)] == expected[::-1].tolist()


@pytest.mark.parametrize("axis_size", [0, 1, 5])
@pytest.mark.parametrize("repeats", [0, 1, 3, [0], [3], np.int64(2)])
def test_uniform_counts_match_numpy(axis_size, repeats):
    expected = np.repeat(np.arange(axis_size), repeats)
    lookup = repeat_source_lookup(axis_size, repeats)
    assert lookup.count == len(expected)
    assert [lookup[i] for i in range(lookup.count)] == expected.tolist()


def test_huge_uniform_repeat_uses_constant_memory_and_exact_positions():
    """The last logical copy stays queryable beyond the platform length limit."""
    axis_size = 10**30
    copies = 10**20
    tracemalloc.start()
    try:
        lookup = repeat_source_lookup(axis_size, copies)
        assert lookup.count == axis_size * copies
        assert lookup[0] == 0
        assert lookup[copies - 1] == 0
        assert lookup[copies] == 1
        assert lookup[-1] == axis_size - 1
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100_000
    assert lookup.count > sys.maxsize
    with pytest.raises(OverflowError):
        len(lookup)


def test_variable_huge_counts_store_only_input_boundaries():
    copies = 10**30
    tracemalloc.start()
    try:
        lookup = repeat_source_lookup(5, [0, copies, 0, copies, 0])
        assert lookup.count == 2 * copies
        assert lookup[0] == 1
        assert lookup[copies - 1] == 1
        assert lookup[copies] == 3
        assert lookup[-1] == 3
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100_000


def test_lookup_snapshots_iterable_counts_and_is_immutable():
    counts = [1, 0, 2]
    lookup = repeat_source_lookup(3, iter(counts))
    counts[:] = [0, 0, 0]
    assert [lookup[i] for i in range(lookup.count)] == [0, 2, 2]
    with pytest.raises(FrozenInstanceError):
        lookup.count = 0


def test_source_coordinate_accepts_lookup_and_scalar_source():
    lookup = repeat_source_lookup(3, [0, 2, 1])
    assert repeat_source_coord((4, 0), lookup, 1) == (4, 1)
    assert repeat_source_coord((4, 2), lookup, 1) == (4, 2)
    assert repeat_source_coord((2,), repeat_source_lookup(1, 3), 0, shape=()) == ()


@pytest.mark.parametrize("axis_size, counts", [(0, []), (0, 5), (3, 0), (3, [0, 0, 0])])
def test_empty_lookup_has_no_valid_coordinate(axis_size, counts):
    lookup = repeat_source_lookup(axis_size, counts)
    assert lookup.count == 0
    for position in [0, -1, 1]:
        with pytest.raises(IndexError, match="out of range"):
            lookup[position]


@pytest.mark.parametrize("position", [3, -4])
def test_lookup_rejects_out_of_bounds_position(position):
    with pytest.raises(IndexError, match="out of range"):
        repeat_source_lookup(3, 1)[position]


@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5])
def test_lookup_rejects_noninteger_axis_count_and_position(value):
    with pytest.raises(TypeError):
        repeat_source_lookup(value, 1)
    with pytest.raises(TypeError):
        repeat_source_lookup(2, value)
    with pytest.raises(TypeError):
        repeat_source_lookup(2, [1, value])
    with pytest.raises(TypeError):
        repeat_source_lookup(2, 1)[value]


@pytest.mark.parametrize("axis_size, counts", [(-1, 2), (3, -1), (3, [1, -1, 2]), (3, [])])
def test_lookup_rejects_negative_sizes_counts_and_wrong_length(axis_size, counts):
    with pytest.raises(ValueError):
        repeat_source_lookup(axis_size, counts)


def test_lookup_accepts_integer_index_protocol():
    class IndexValue:
        def __index__(self):
            return 2

    lookup = repeat_source_lookup(IndexValue(), [IndexValue()])
    assert lookup.count == 4
    assert lookup[IndexValue()] == 1
