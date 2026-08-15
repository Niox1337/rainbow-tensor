"""Compact membership joins inspect candidate inputs without repeated full scans."""

from collections import Counter

import numpy as np
import pytest

from rainbow_tensor.index_mapping import IndexMapping


class ObservedPosition(int):
    """Count hash probes and comparisons after candidate coordinates are copied."""

    def __new__(cls, value, work):
        position = int.__new__(cls, value)
        position.work = work
        return position

    def __hash__(self):
        self.work["hashes"] += 1
        return int.__hash__(self)

    def __eq__(self, other):
        self.work["comparisons"] += 1
        return int.__eq__(self, other)


class ObservedCoordinate:
    """Count coordinate reads both through projection and through iteration."""

    def __init__(self, coordinate, work):
        self.coordinate = coordinate
        self.work = work

    def __getitem__(self, axis):
        self.work["axis_reads"] += 1
        return ObservedPosition(self.coordinate[axis], self.work)

    def __iter__(self):
        for value in self.coordinate:
            self.work["axis_reads"] += 1
            yield ObservedPosition(value, self.work)


class ObservedPositions:
    """Expose the existing candidate relation while counting each visited row."""

    def __init__(self, positions, work):
        self.positions = positions
        self.work = work

    def __len__(self):
        return len(self.positions)

    def __iter__(self):
        self.work["scans"] += 1
        for coordinate in self.positions:
            self.work["candidate_visits"] += 1
            yield ObservedCoordinate(coordinate, self.work)


def observe_candidates(mapping):
    """Instrument the stored inverse lookup, after the real index has been parsed."""
    work = Counter()
    for array in mapping.arrays.values():
        array.positions = {
            value: ObservedPositions(positions, work)
            for value, positions in array.positions.items()
        }
    return work


@pytest.mark.parametrize("count", [40, 80, 160])
@pytest.mark.parametrize("use_prefix", [False, True])
def test_disjoint_duplicate_gathers_scan_each_candidate_list_once(count, use_prefix):
    left = np.repeat([0, 1], count)
    right = np.repeat([1, 0], count)
    shape = (2, 3, 2, 4) if use_prefix else (2, 2)
    index = (left, slice(None), right, slice(None)) if use_prefix else (left, right)
    mapping = IndexMapping(shape, index)
    work = observe_candidates(mapping)
    coordinate = (0, 2, 0) if use_prefix else (0, 0)

    reference = np.zeros(shape, dtype=bool)
    reference[index] = True
    assert not np.any(reference[coordinate])
    if use_prefix:
        assert not mapping.selection.matches_prefix(coordinate)
    else:
        assert coordinate not in mapping.selection

    assert work["scans"] <= 2
    assert work["candidate_visits"] <= 2 * count
    assert work["axis_reads"] <= 2 * count


@pytest.mark.parametrize("count", [20, 40, 80])
def test_disconnected_joins_do_not_cross_product_their_candidate_projections(count):
    same = np.repeat([0, 1], count)
    opposite = np.repeat([1, 0], count)
    index = (same[:, None], same[None, :], same[:, None], opposite[None, :])
    mapping = IndexMapping((2,) * 4, index)
    work = observe_candidates(mapping)

    reference = np.zeros((2,) * 4, dtype=bool)
    reference[index] = True
    assert not reference[0, 0, 0, 0]
    assert (0, 0, 0, 0) not in mapping.selection
    assert work["candidate_visits"] <= 4 * count
    assert work["axis_reads"] <= 4 * count
    assert work["hashes"] <= 10 * count
    assert work["comparisons"] <= 4 * count


def test_independent_broadcast_factors_need_no_candidate_enumeration():
    left = np.zeros((200, 1), dtype=int)
    right = np.ones((1, 300), dtype=int)
    mapping = IndexMapping((2, 2), (left, right))
    work = observe_candidates(mapping)

    assert (0, 1) in mapping.selection
    assert mapping.selection.matches_prefix((0,))
    assert mapping.result_count == 60_000
    assert work["candidate_visits"] == 0
    assert work["axis_reads"] == 0


def test_private_axes_are_projected_out_before_joining_repeated_candidates():
    left = np.broadcast_to(np.array([0, 1])[None, :, None], (80, 2, 1))
    right = np.broadcast_to(np.array([1, 0])[None, :, None], (1, 2, 70))
    mapping = IndexMapping((2, 2), (left, right))
    work = observe_candidates(mapping)

    assert (0, 0) not in mapping.selection
    assert work["candidate_visits"] <= 150
    assert work["axis_reads"] <= 150


def cyclic_indices():
    """Return pairwise-compatible relations with an impossible three-way pick."""
    x, y, z = np.indices((2, 2, 2))
    return (
        np.where(x[:, :, :1] == y[:, :, :1], 0, 1),
        np.where(y[:1, :, :] == z[:1, :, :], 0, 1),
        np.where(x[:, :1, :] != z[:, :1, :], 0, 1),
    )


@pytest.mark.parametrize(
    "shape, index",
    [
        ((2, 2), (np.array([[0], [1]]), np.array([[1, 0]]))),
        ((2, 2), ([0, 1], np.array([[0, 1], [1, 0], [0, 1]]))),
        ((2, 2), (np.array([[0]]), np.array([[0, 1], [1, 0]]))),
        ((2, 2), (np.empty((0, 1), dtype=int), np.array([[0, 1]]))),
        ((2, 2), (np.empty((0, 2), dtype=bool),)),
        ((2, 2), (np.array([[True, False], [False, True]]), None)),
        ((2, 3, 2), ([0, 1, 0], slice(None, None, -1), [1, 0, 1])),
        ((2, 3, 2), ([0, 1, 0], slice(0, 0), [1, 0, 1])),
        ((2, 2, 2), (None, [1, 0], 1, [0, 1])),
        ((2, 2, 2), cyclic_indices()),
    ],
)
def test_membership_and_prefixes_preserve_numpy_broadcast_correlations(shape, index):
    reference = np.zeros(shape, dtype=bool)
    reference[index] = True
    expected = set(map(tuple, np.argwhere(reference)))
    selection = IndexMapping(shape, index).selection

    for coordinate in np.ndindex(shape):
        assert (coordinate in selection) == (coordinate in expected)
        for length in range(len(shape) + 1):
            prefix = coordinate[:length]
            assert selection.matches_prefix(prefix) == any(
                selected[:length] == prefix for selected in expected
            )


def test_three_way_join_does_not_confuse_pairwise_matches_with_a_full_match():
    mapping = IndexMapping((2, 2, 2), cyclic_indices())
    assert mapping.selection.matches_prefix((0, 0))
    assert (0, 0, 0) not in mapping.selection
    assert (0, 0, 1) in mapping.selection
