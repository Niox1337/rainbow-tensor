"""Basic selection metadata stays compact while previews retain their highlights."""

from itertools import islice

import pytest

import rainbow_tensor as rt
from rainbow_tensor.indexing import selected_coordinates
from rainbow_tensor.layout import build_layout
from rainbow_tensor.render_svg import render_panels, render_svg
from rainbow_tensor.selection import BasicSelection


@pytest.mark.parametrize(
    "index",
    [
        (slice(None), slice(1, 5, 2), 2),
        (slice(None, None, -1), slice(4, 0, -2), slice(None, None, -2)),
        (None, -1, Ellipsis, None),
        (slice(1, 1), slice(None), slice(None)),
    ],
)
def test_compact_selection_preserves_explicit_coordinate_order(index):
    shape = (4, 5, 6)
    expected = selected_coordinates(shape, index)
    selection = BasicSelection(shape, index)

    assert selection.count == len(expected)
    assert len(selection) == len(expected)
    assert bool(selection) == bool(expected)
    assert list(selection) == expected
    assert selection == expected
    assert expected == selection
    assert selection[:3] == expected[:3]
    assert selection[::-1] == expected[::-1]
    if expected:
        assert selection[0] == expected[0]
        assert selection[-1] == expected[-1]
        assert all(coordinate in selection for coordinate in expected)
    with pytest.raises(IndexError):
        selection[len(expected)]


def test_trillion_coordinate_selection_supports_lazy_access_and_membership():
    selection = BasicSelection((1_000_000, 1_000_000), (Ellipsis,))

    assert selection.count == 1_000_000_000_000
    assert len(selection) == 1_000_000_000_000
    assert list(islice(selection, 3)) == [(0, 0), (0, 1), (0, 2)]
    assert selection[1_000_000] == (1, 0)
    assert selection[-1] == (999_999, 999_999)
    assert (123_456, 654_321) in selection
    assert (1_000_000, 0) not in selection
    assert (0,) not in selection
    assert (0, 0.5) not in selection
    assert selection.matches_prefix((123_456,))
    assert not selection.matches_prefix((1_000_000,))


@pytest.mark.parametrize("step", [1, 3, -1, -3])
@pytest.mark.parametrize("limit", [3, 6, 12])
def test_large_axis_pins_include_head_and_tail_within_the_budget(step, limit):
    selection = BasicSelection((1_000_000,), (slice(None, None, step),))
    axis = selection.axes[0]

    pins = selection.axis_pins(0, limit)

    assert len(pins) <= limit
    assert pins[0] == min(axis[0], axis[-1])
    assert pins[-1] == max(axis[0], axis[-1])
    assert all((pin,) in selection for pin in pins)


def test_compact_layout_matches_explicit_small_selection():
    shape = (5, 4, 3)
    index = (slice(1, 5, 2), slice(None), 2)
    selection = BasicSelection(shape, index)

    assert build_layout(shape, selected=selection) == build_layout(
        shape, selected=selected_coordinates(shape, index)
    )
    assert not selection.matches_prefix((0,))
    assert selection.matches_prefix((1, 3))
    assert not selection.matches_prefix((1, 3, 1))


def test_empty_selection_retains_unselected_rendering():
    shape = (4, 5)
    selection = BasicSelection(shape, (slice(0, 0), slice(None)))

    assert selection.count == 0
    assert not selection.matches_prefix(())
    assert selection.axis_pins(1, 12) == ()
    assert render_svg(shape, selected=selection) == render_svg(shape, selected=[])


def _reject_enumeration(selection):
    raise AssertionError("rendering must not enumerate a compact selection")


def test_large_basic_index_previews_without_enumerating_coordinates(monkeypatch):
    class CountingArray:
        shape = (1_000_000, 1_000_000)
        reads = 0

        def __getitem__(self, coordinate):
            self.reads += 1
            return 1

    monkeypatch.setattr(BasicSelection, "__iter__", _reject_enumeration)
    array = CountingArray()

    visual = rt.index(array, (None, slice(None), slice(None, None, -1)))

    assert visual.result_shape == (1, 1_000_000, 1_000_000)
    assert visual.selected.count == 1_000_000_000_000
    assert visual.selected[-1] == (999_999, 0)
    assert 0 < array.reads <= rt.LIGHT.max_visible_cells
    assert visual.svg.startswith("<svg")


def test_renderers_use_bool_when_selection_count_exceeds_python_len(monkeypatch):
    shape = (1_000_000,) * 4
    selection = BasicSelection(shape, (Ellipsis,))
    monkeypatch.setattr(BasicSelection, "__iter__", _reject_enumeration)

    assert selection.count == 10**24
    assert bool(selection)
    with pytest.raises(OverflowError):
        len(selection)
    assert render_svg(shape, selected=selection).startswith("<svg")
    assert render_panels([{"shape": shape, "selected": selection}]).startswith("<svg")


def test_explicit_selection_generators_still_render():
    coordinates = [(0, 1), (2, 3)]

    assert render_svg((3, 4), selected=iter(coordinates)) == render_svg(
        (3, 4), selected=coordinates
    )
