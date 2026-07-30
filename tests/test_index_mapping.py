"""Lazy indexing provenance preserves NumPy output order without expanding slices."""

from itertools import islice
from math import prod

import numpy as np
import pytest

import rainbow_tensor.index_mapping as mapping_module
from rainbow_tensor.index_mapping import CompactSelection, IndexMapping
from rainbow_tensor.indexing import advanced_index
from rainbow_tensor.layout import build_layout
from rainbow_tensor.render_svg import render_panels, render_svg


@pytest.mark.parametrize(
    "shape, index",
    [
        ((3, 4), ()),
        ((3, 4), (None, -1, Ellipsis, None)),
        ((3, 4), (slice(None, None, -1), slice(None, None, -2))),
        ((3, 4), (np.int64(1),)),
        ((3, 4), (1, 2)),
        ((3, 4), (slice(0, 0), None)),
        ((3, 4), ([2, 0, 2], slice(None, None, -2))),
        ((3, 4), ([2, 0, 2], [3, 1, 3])),
        ((3, 4), ([[2], [0]], [3, 1, 3])),
        ((3, 4), (slice(None), [[3, 1], [0, 1]])),
        ((3, 4), ([True, 2], slice(None))),
        ((3, 4), (np.array([True, False, True]),)),
        ((3, 4), (np.array([True, False, True]), [3, 1])),
        ((3, 4), np.array([[True, False, True, False], [False] * 4, [True] * 4])),
        ((3, 4), (np.empty(0, dtype=int), None)),
        ((3, 4), (np.empty((0, 2), dtype=int), slice(None))),
        ((3, 4), (np.empty((0, 4), dtype=bool), None)),
        ((3, 4), np.empty((3, 0), dtype=bool)),
        ((3, 4, 5), ([2, 0], slice(None, None, -1), [4, 1])),
        ((3, 4, 5), (0, slice(None), [4, 1])),
        ((3, 4, 5), ([2, 0], slice(None), 1)),
        ((3, 4, 5), (slice(None), np.array(1), [4, 1])),
        ((3, 4, 5), (slice(None), [2, 0], Ellipsis, [4, 1])),
        ((3, 4, 5), (slice(None), 1, Ellipsis, [4, 1])),
        ((3, 4, 5), (None, [2, 0], slice(None), [4, 1])),
        ((3, 4, 5), (slice(None), None, [2, 0], [4, 1])),
        ((3, 4, 5), (slice(None), [2, 0], None, [4, 1])),
        ((3, 4, 5), (slice(None), [2, 0], [4, 1], None)),
        ((3, 4, 5), (None, 1, None, [2, 0])),
        ((3, 4, 5), ([2, 0], None, 1)),
        ((3, 4, 5), (None, np.empty((3, 0), dtype=bool), Ellipsis)),
    ],
)
def test_result_shape_values_and_coordinate_order_match_numpy(shape, index):
    array = np.arange(prod(shape)).reshape(shape)
    expected = array[index]
    mapping = IndexMapping(shape, index)

    assert mapping.result_shape == expected.shape
    assert mapping.result_count == expected.size
    assert bool(mapping.selection) == bool(expected.size)
    assert isinstance(mapping.selection, CompactSelection)
    assert [array[coordinate] for coordinate in mapping] == list(expected.ravel())
    for coordinate in np.ndindex(expected.shape):
        assert array[mapping.source_coord(coordinate)] == expected[coordinate]
    assert {array[coordinate] for coordinate in mapping.selection} == set(expected.ravel())


def test_repeated_gathers_retain_result_order_and_unique_source_highlights():
    mapping = IndexMapping((3, 4), ([2, 0, 2], slice(None, None, -2)))

    assert mapping.result_count == 6
    assert list(mapping) == [(2, 3), (2, 1), (0, 3), (0, 1), (2, 3), (2, 1)]
    assert mapping.selection == [(2, 3), (2, 1), (0, 3), (0, 1)]
    selected, shape, explanation = advanced_index((3, 4), ([2, 0, 2], slice(None, None, -2)))
    assert isinstance(selected, list)
    assert selected == list(mapping.selection)
    assert shape == mapping.result_shape
    assert explanation == mapping.explanation


@pytest.mark.parametrize(
    "index",
    [
        ([2, 0, 2], slice(None, None, -2), [4, 1, 4]),
        ([[2], [0]], [3, 1], slice(None)),
        ([[2, 0], [2, 1]], slice(None), [[4, 1], [3, 2]]),
        (np.array([True, False, True]), slice(None), [4, 1]),
        (slice(None), [2, 0], 1),
        (None, [2, 0], slice(0, 0)),
    ],
)
def test_compact_membership_and_prefixes_match_selected_source_coordinates(index):
    shape = (3, 4, 5)
    reference = np.zeros(shape, dtype=bool)
    reference[index] = True
    expected = set(map(tuple, np.argwhere(reference)))
    selection = IndexMapping(shape, index).selection

    for coordinate in np.ndindex(shape):
        assert (coordinate in selection) == (coordinate in expected)
        for size in range(len(shape) + 1):
            prefix = coordinate[:size]
            assert selection.matches_prefix(prefix) == any(
                picked[:size] == prefix for picked in expected
            )
    assert (3, 0, 0) not in selection
    assert (0, 0) not in selection
    assert (0, 0, 0.5) not in selection
    assert not selection.matches_prefix((0, 0, 0, 0))


@pytest.mark.parametrize("index", [(1, 2), (None, [2, 0], slice(None, None, -1))])
def test_source_coord_accepts_negative_result_positions(index):
    mapping = IndexMapping((3, 4), index)
    last = tuple(size - 1 for size in mapping.result_shape)
    negative = (-1,) * len(mapping.result_shape)
    assert mapping.source_coord(negative) == mapping.source_coord(last)


def test_source_coord_validates_result_coordinates_and_empty_results():
    mapping = IndexMapping((3, 4), ([2, 0],))
    with pytest.raises(TypeError, match="tuple"):
        mapping.source_coord([0, 0])
    with pytest.raises((ValueError, IndexError)):
        mapping.source_coord((0,))
    with pytest.raises((ValueError, IndexError)):
        mapping.source_coord((2, 0))
    with pytest.raises(TypeError):
        mapping.source_coord((0.5, 0))
    empty = IndexMapping((3, 4), ([],))
    with pytest.raises(IndexError, match="empty"):
        empty.source_coord((0, 0))


def test_mapping_copies_index_arrays_before_later_mutation():
    indices = np.array([2, 0, 2])
    mapping = IndexMapping((3,), (indices,))
    indices[:] = 1
    assert list(mapping) == [(2,), (0,), (2,)]
    assert (1,) not in mapping.selection


def test_large_slice_product_is_lazy_with_repeated_gathers(monkeypatch):
    shape = (1_000_000, 1_000_000, 1_000_000)
    original = mapping_module._coordinates

    def small_coordinates(shape):
        assert prod(shape) <= 3, "only the three index entries may be visited"
        return original(shape)

    monkeypatch.setattr(mapping_module, "_coordinates", small_coordinates)
    mapping = IndexMapping(shape, ([999_999, 1, 999_999], slice(None), slice(None, None, -1)))

    assert mapping.result_count == 3_000_000_000_000
    assert mapping.source_coord((2, 654_321, 7)) == (999_999, 654_321, 999_992)
    assert (999_999, 654_321, 7) in mapping.selection
    assert (2, 654_321, 7) not in mapping.selection
    assert mapping.selection.matches_prefix((999_999, 654_321))
    assert len(mapping.selection.axis_pins(2, 6)) <= 6


def test_broadcast_indices_do_not_expand_output_for_membership_or_rendering(monkeypatch):
    count = 1_000
    rows = np.arange(count)[:, None]
    columns = np.arange(count)[None, :]
    original = mapping_module._coordinates

    def input_coordinates(shape):
        assert prod(shape) <= count, "the broadcast output must not be enumerated"
        return original(shape)

    monkeypatch.setattr(mapping_module, "_coordinates", input_coordinates)
    mapping = IndexMapping((1_000_000, 1_000_000), (rows, columns))
    selection = mapping.selection
    assert mapping.result_count == 1_000_000
    assert mapping.source_coord((789, 456)) == (789, 456)
    assert (789, 456) in selection
    assert (1_000, 456) not in selection
    assert selection.matches_prefix((999,))
    assert len(selection.axis_pins(0, 6)) <= 6

    def reject_enumeration(self):
        raise AssertionError("compact source highlights must not be enumerated")

    monkeypatch.setattr(type(selection), "__iter__", reject_enumeration)
    assert render_svg(mapping.source_shape, selected=selection).startswith("<svg")
    panels = [{"shape": mapping.source_shape, "selected": selection}]
    assert render_panels(panels).startswith("<svg")


def test_empty_slice_avoids_enumerating_a_large_broadcast_selection(monkeypatch):
    count = 1_000
    original = mapping_module._coordinates

    def input_coordinates(shape):
        assert prod(shape) <= count, "an empty selection has no gather groups to visit"
        return original(shape)

    monkeypatch.setattr(mapping_module, "_coordinates", input_coordinates)
    mapping = IndexMapping(
        (count, count, 5),
        (np.arange(count)[:, None], np.arange(count)[None, :], slice(0, 0)),
    )
    assert mapping.result_count == 0
    assert list(mapping.selection) == []


def test_advanced_selection_renders_like_explicit_small_highlights():
    shape = (3, 4, 5)
    selection = IndexMapping(shape, ([2, 0], slice(None), [4, 1])).selection
    explicit = list(selection)
    assert build_layout(shape, selected=selection) == build_layout(shape, selected=explicit)
    assert render_svg(shape, selected=selection) == render_svg(shape, selected=explicit)


def test_mapping_iteration_is_an_explicit_lazy_opt_in():
    mapping = IndexMapping((1_000_000, 1_000_000), ([2, 0, 2], slice(None)))
    assert list(islice(mapping, 3)) == [(2, 0), (2, 1), (2, 2)]
    assert list(islice(mapping.selection, 3)) == [(2, 0), (2, 1), (2, 2)]
