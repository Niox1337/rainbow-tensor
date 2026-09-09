"""Equal-shape gathers share candidate coordinates without projection joins."""

import itertools

import numpy as np
import pytest

from rainbow_tensor.index_mapping import IndexMapping


class CountedCandidates:
    """Count the actual candidate rows consumed by a selection query."""

    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.visits = 0
        self.scans = 0

    def __len__(self):
        return len(self.coordinates)

    def __iter__(self):
        self.scans += 1
        for coordinate in self.coordinates:
            self.visits += 1
            yield coordinate


def assert_matches_numpy(shape, index):
    """Compare membership, all prefixes and result order with NumPy."""
    source = np.arange(np.prod(shape)).reshape(shape)
    expected_values = source[index]
    selected = np.zeros(shape, dtype=bool)
    selected[index] = True
    mapping = IndexMapping(shape, index)
    assert mapping.result_shape == expected_values.shape
    for coordinate in np.ndindex(shape):
        assert (coordinate in mapping.selection) == bool(selected[coordinate])
        for size in range(len(shape) + 1):
            prefix = coordinate[:size]
            assert mapping.selection.matches_prefix(prefix) == bool(selected[prefix].any())
    for coordinate in np.ndindex(expected_values.shape):
        assert source[mapping.source_coord(coordinate)] == expected_values[coordinate]


@pytest.mark.parametrize("array_shape", [(6,), (2, 3), (1, 2, 3), (2, 1, 3), ()])
def test_equal_shape_multiway_gathers_match_numpy(array_shape):
    generator = np.random.default_rng(106)
    source_shape = (3, 4, 2)
    for _ in range(12):
        arrays = tuple(generator.integers(-size, size, array_shape) for size in source_shape)
        assert_matches_numpy(source_shape, arrays)


@pytest.mark.parametrize("separator", [slice(None, None, -1), 1, None, Ellipsis])
def test_equal_shape_gathers_preserve_separated_axes_and_insertions(separator):
    left = np.array([[0, -1], [0, 1]])
    right = np.array([[-1, 0], [1, 1]])
    shape = (3, 2) if separator is None else (3, 4, 2)
    assert_matches_numpy(shape, (left, separator, right))


@pytest.mark.parametrize("array_shape", [(0,), (0, 2), (1, 0, 2)])
def test_empty_equal_shape_gathers_have_no_selected_source(array_shape):
    empty = np.empty(array_shape, dtype=int)
    assert_matches_numpy((2, 3), (empty, empty))


def test_multiway_intersection_requires_one_position_common_to_every_array():
    arrays = (np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    for order in itertools.permutations(arrays):
        assert_matches_numpy((2, 2, 2), order)
        selection = IndexMapping((2, 2, 2), order).selection
        assert selection.matches_prefix((0, 0))
        assert (0, 0, 0) not in selection


def test_short_disjoint_candidates_skip_a_large_remaining_relation():
    size = 20_000
    dense = np.zeros(size, dtype=int)
    first = np.ones(size, dtype=int)
    second = np.ones(size, dtype=int)
    first[0] = 0
    second[-1] = 0
    mapping = IndexMapping((2, 2, 2), (dense, first, second))
    observed = []
    for array in mapping.arrays.values():
        candidates = CountedCandidates(array.positions[0])
        array.positions[0] = candidates
        observed.append(candidates)

    assert (0, 0, 0) not in mapping.selection
    assert observed[0].visits == 0
    assert sum(candidates.visits for candidates in observed) == 2
    assert all(candidates.scans <= 1 for candidates in observed)


def test_equal_shape_duplicates_are_scanned_at_most_once_per_query():
    repeats = 2_000
    left = np.repeat([0, 1], repeats).reshape(20, 200)
    right = np.repeat([1, 0], repeats).reshape(20, 200)
    mapping = IndexMapping((2, 2), (left, right))
    observed = []
    for array in mapping.arrays.values():
        candidates = CountedCandidates(array.positions[0])
        array.positions[0] = candidates
        observed.append(candidates)

    assert (0, 0) not in mapping.selection
    assert sum(candidates.visits for candidates in observed) == 2 * repeats
    assert all(candidates.scans == 1 for candidates in observed)


def test_differently_shaped_arrays_keep_broadcast_coordinate_semantics():
    left = np.array([[0], [1]])
    right = np.array([[1, 0, 1]])
    assert_matches_numpy((2, 2), (left, right))
    assert_matches_numpy((2, 3), ([0, 1], np.array([[0, 1], [2, 0], [1, 2]])))


def test_equal_shape_prefix_can_leave_a_larger_broadcast_array_unconstrained():
    arrays = (
        np.array([0, 1, 0]),
        np.array([1, 0, 0]),
        np.array([[0, 1, 0], [1, 0, 1]]),
    )
    assert_matches_numpy((2, 2, 2), arrays)
