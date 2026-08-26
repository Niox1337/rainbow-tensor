"""Scalar and empty sources retain indexing provenance without fabricated reads."""

import tracemalloc

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.index_mapping import IndexMapping
from rainbow_tensor.indexing import _range_length, selected_coordinates
from rainbow_tensor.selection import BasicSelection


class ReadGuard:
    """Reject nonexistent reads and record the native coordinate of a scalar."""

    def __init__(self, shape):
        self.shape = shape
        self.reads = []

    def __getitem__(self, coordinate):
        assert self.shape == () and coordinate == ()
        self.reads.append(coordinate)
        return 7


@pytest.mark.parametrize("shape, index", [
    ((), ()),
    ((), (Ellipsis,)),
    ((), (None,)),
    ((), (None, Ellipsis, None)),
    ((0,), ()),
    ((0,), (slice(None, None, -1),)),
    ((2, 0, 3), (slice(None), slice(None), 1)),
    ((2, 0, 3), (1, Ellipsis)),
    ((2, 0, 3), (None, Ellipsis, None)),
    ((0, 3), ([], slice(None))),
    ((0, 3), (np.empty(0, dtype=bool), Ellipsis)),
    ((2, 0, 3), (np.array([1, 0, 1]), slice(None), slice(None))),
    ((2, 0), np.empty((2, 0), dtype=bool)),
])
def test_index_mapping_and_visual_shapes_match_numpy(shape, index):
    array = np.arange(np.prod(shape, dtype=int)).reshape(shape)
    expected = array[index]
    mapping = IndexMapping(shape, index)
    assert mapping.result_shape == expected.shape
    assert mapping.result_count == expected.size
    assert [array[coordinate] for coordinate in mapping] == list(expected.ravel())
    for coordinate in np.ndindex(expected.shape):
        assert array[mapping.source_coord(coordinate)] == expected[coordinate]
    for show_result in (False, True):
        visual = rt.index(array, index, show_result=show_result)
        assert visual.shape == shape
        assert visual.result_shape == expected.shape
        assert visual.index_mapping.result_count == expected.size


@pytest.mark.parametrize("index", [(), (Ellipsis,), (None,), (None, None)])
def test_scalar_index_reads_only_the_empty_coordinate(index):
    array = ReadGuard(())
    visual = rt.index(array, index, show_result=True)
    assert array.reads and set(array.reads) == {()}
    assert list(visual.selected) == [()]
    if index == ():
        assert "Index: ()" in visual.text


@pytest.mark.parametrize("shape, index", [
    ((), (0,)), ((), (slice(None),)), ((0,), (0,)),
    ((2, 0), (0, 0)), ((0,), ([0],)),
])
def test_invalid_degenerate_indices_fail_before_array_reads(shape, index):
    array = ReadGuard(shape)
    with pytest.raises(IndexError):
        rt.index(array, index, show_result=True)
    assert array.reads == []


@pytest.mark.parametrize("shape", [(10**12, 0), (0, 10**12), (2, 0, 10**12)])
def test_huge_empty_selection_and_rendering_remain_bounded(shape):
    array = ReadGuard(shape)
    tracemalloc.start()
    try:
        mapping = IndexMapping(shape, ())
        assert mapping.result_count == 0
        assert list(mapping) == list(mapping.selection) == []
        assert selected_coordinates(shape, ()) == []
        assert not mapping.selection.matches_prefix(())
        visual = rt.index(array, (), show_result=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert "No elements" in visual.svg
    assert array.reads == []
    assert peak < 1_000_000
    with pytest.raises(IndexError, match="empty"):
        mapping.source_coord(tuple(0 for _ in shape))


def test_empty_selection_short_circuits_dimensions_larger_than_platform_length():
    selected = BasicSelection((10**30, 0), ())
    assert selected.count == 0
    assert list(selected) == []
    assert selected_coordinates((10**30, 0), ()) == []


@pytest.mark.parametrize("shape, index, expected", [
    ((10**30, 0), (), (10**30, 0)),
    ((0, 10**30), (slice(None), slice(None, None, -2)), (0, (10**30 + 1) // 2)),
    ((10**30, 0, 3), (slice(1, None, 3), Ellipsis), ((10**30 + 1) // 3, 0, 3)),
    ((10**30, 0), (slice(None), []), (10**30, 0)),
    ((0, 10**30), ([], slice(None, None, -1)), (0, 10**30)),
    ((10**30, 0, 3), (slice(None), [], [1]), (10**30, 0)),
])
def test_empty_index_shape_metadata_exceeds_platform_length_without_reads(shape, index, expected):
    array = ReadGuard(shape)
    mapping = IndexMapping(shape, index)
    assert mapping.result_shape == expected
    assert mapping.result_count == 0
    assert list(mapping) == list(mapping.selection) == []
    visual = rt.index(array, index, show_result=True)
    assert visual.result_shape == expected
    assert "No elements" in visual.svg
    assert array.reads == []


@pytest.mark.parametrize("start, stop, step", [
    (0, 5, 1), (1, 10, 3), (9, -1, -2), (0, 0, 1),
    (5, 0, 1), (0, 5, -1), (-5, -12, -3),
])
def test_arithmetic_range_length_matches_small_python_ranges(start, stop, step):
    positions = range(start, stop, step)
    assert _range_length(positions) == len(positions)


def test_large_nonempty_selection_uses_arbitrary_size_counts_and_bounded_pins():
    size = 10**30
    selected = BasicSelection((size,), (slice(None, None, -1),))
    assert selected.count == size
    assert selected[0] == (size - 1,)
    assert selected[-1] == (0,)
    assert selected[size // 2] == (size - 1 - size // 2,)
    assert len(selected.axis_pins(0, 6)) <= 6
    assert (size - 1,) in selected


@pytest.mark.parametrize("shape, index", [
    ((0, 0), ([], [0])),
    ((0, 0), ([99], [])),
    ((2, 2), ([], [99])),
    ((2, 0), ([[-10]], np.empty((2, 0), dtype=int))),
    ((0, 2), (None, [], [99], None)),
    ((2, 0, 3), (slice(None), [], [99])),
    ((2, 0), (np.array(0), [])),
])
def test_empty_advanced_broadcast_skips_unused_gather_bounds_like_numpy(shape, index):
    expected = np.empty(shape)[index]
    mapping = IndexMapping(shape, index)
    assert mapping.result_shape == expected.shape
    assert mapping.result_count == expected.size == 0
    assert list(mapping) == list(mapping.selection) == []


@pytest.mark.parametrize("shape, index", [
    ((0, 2), (0, [])),
    ((2, 0), ([], 7)),
    ((2, 0), (np.int64(99), [])),
    ((2, 0), (np.array(99), [])),
    ((2, 0), ([1.5], [])),
    ((2, 0), (np.array([1.0]), [])),
    ((2, 0), ([0, 1], [])),
])
def test_empty_advanced_broadcast_still_validates_scalars_types_and_shapes(shape, index):
    with pytest.raises((IndexError, TypeError, ValueError)):
        np.empty(shape)[index]
    with pytest.raises((IndexError, TypeError, ValueError)):
        IndexMapping(shape, index)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_memory_reports_degenerate_metadata_without_inventing_aliases(shape):
    array = np.empty(shape, dtype=np.float32)
    visual = rt.memory(array)
    assert visual.shape == shape
    assert visual.metadata["strides"] == array.strides
    assert visual.metadata["c_contiguous"] == bool(array.flags.c_contiguous)
    if array.size == 0:
        assert "no element addresses" in visual.text
        assert "reuses the same storage" not in visual.text
