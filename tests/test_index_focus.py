"""Index focus preserves output identity while highlighting one source coordinate."""

import xml.etree.ElementTree as ET
from math import prod

import numpy as np
import pytest

import rainbow_tensor as rt
import rainbow_tensor.index_mapping as mapping_module
import rainbow_tensor.views.shapes as shape_views
from rainbow_tensor.tracing import _trace_explanation
from rainbow_tensor.views.shapes import _index_visual


class RecordingRenderer:
    """Capture one render without requesting backend values during trace construction."""

    name = "index-focus-recording"
    mime_type = "text/plain"

    def __init__(self):
        self.calls = []

    def render_tensor(self, **kwargs):
        self.calls.append(("tensor", kwargs))
        return "tensor"

    def render_panels(self, **kwargs):
        self.calls.append(("panels", kwargs))
        self.panels = kwargs["panels"]
        return "panels"


class CountingArray:
    """Record scalar reads from arbitrary logical shapes without allocating their data."""

    def __init__(self, shape):
        self.shape = shape
        self.reads = []

    def __getitem__(self, coordinate):
        assert len(coordinate) == len(self.shape)
        assert all(0 <= index < size for index, size in zip(coordinate, self.shape))
        self.reads.append(coordinate)
        return 1


@pytest.mark.parametrize(
    ("shape", "index", "focus"),
    [
        ((3, 4), ([2, 0, 2], slice(None, None, -2)), (2, 1)),
        ((3, 4), (slice(None, None, -1), slice(None, None, -2)), (-1, -1)),
        ((3, 4), ([[2], [0]], [3, 1, 3]), (1, 2)),
        ((3, 4), (None, [2, 0], slice(None)), (0, 1, 3)),
        ((3, 4, 5), ([2, 0], slice(None, None, -1), [4, 1]), (1, 2)),
        ((3, 4), np.array([[True, False, True, False], [False] * 4, [True] * 4]), (4,)),
        ((3, 4), (2, 1), ()),
        ((), (), ()),
        ((), (None, None), (0, 0)),
    ],
)
def test_focused_pair_values_and_full_selection_match_numpy(shape, index, focus):
    array = np.arange(prod(shape)).reshape(shape)
    expected = array[index]
    normalized = tuple(position % size for position, size in zip(focus, expected.shape))
    source = tuple(np.unravel_index(int(expected[normalized]), shape))
    renderer = RecordingRenderer()
    visual = rt.index(array, index, focus=focus, renderer=renderer)

    assert visual.result_shape == expected.shape
    assert visual.trace.operation == "index"
    assert visual.trace.output_coord == normalized
    assert visual.trace.term_count == 1
    assert visual.trace.complete
    assert visual.trace.divisor == 1
    reference = visual.trace.terms[0][0]
    assert reference.operand == 0
    assert reference.coordinate == source
    assert renderer.panels[0]["selected"] == [source]
    assert renderer.panels[1]["selected"] == [normalized]
    assert renderer.panels[1]["value_fn"](normalized) == expected[normalized]
    assert visual.index_mapping.source_coord(normalized) == source
    assert set(visual.selected) == {
        tuple(np.unravel_index(int(value), shape)) for value in expected.ravel()
    }
    assert visual.metadata["trace_in_explanation"] is True
    explanation = _trace_explanation(visual.trace)
    assert len(explanation) == 1
    assert visual.explanation.count(explanation[0]) == 1
    assert len(renderer.calls) == 1


def test_duplicate_output_positions_keep_separate_focus_on_the_same_source():
    renderer = RecordingRenderer()
    first = rt.index((3, 4), ([2, 0, 2], slice(None, None, -2)), focus=(0, 0), renderer=renderer)
    first_source = renderer.panels[0]["selected"]
    assert renderer.panels[1]["selected"] == [(0, 0)]
    repeated = rt.index((3, 4), ([2, 0, 2], slice(None, None, -2)), focus=(2, 0), renderer=renderer)
    assert renderer.panels[0]["selected"] == first_source == [(2, 3)]
    assert renderer.panels[1]["selected"] == [(2, 0)]
    assert first.trace.terms == repeated.trace.terms
    assert first.trace.output_coord != repeated.trace.output_coord
    assert repeated.index_mapping.result_count == 6
    assert set(repeated.selected) == {(2, 3), (2, 1), (0, 3), (0, 1)}
    assert renderer.panels[1]["value_fn"]((2, 0)) == 11


@pytest.mark.parametrize(
    ("focus", "error"),
    [
        ([0, 0], TypeError),
        ((0.5, 0), TypeError),
        ((True, 0), TypeError),
        ((np.bool_(True), 0), TypeError),
        ((0,), ValueError),
        ((3, 0), IndexError),
        ((-4, 0), IndexError),
        ((0, 4), IndexError),
    ],
)
def test_invalid_focus_fails_before_any_source_read_or_render(focus, error):
    array = CountingArray((3, 4))
    renderer = RecordingRenderer()
    with pytest.raises(error):
        rt.index(array, (), focus=focus, renderer=renderer)
    assert array.reads == []
    assert renderer.calls == []


@pytest.mark.parametrize("focus", [(), (0,)])
def test_empty_result_rejects_explicit_focus_before_source_reads(focus):
    array = CountingArray((3, 4))
    with pytest.raises(IndexError, match="empty"):
        rt.index(array, (slice(0, 0),), focus=focus)
    assert array.reads == []


@pytest.mark.parametrize(
    ("shape", "index", "focus"),
    [
        ((3, 4), ([2, 0, 2], slice(None, None, -1)), (0, 0)),
        ((3, 4), (1, 2), ()),
        ((), (), ()),
    ],
)
def test_first_focus_prepares_one_mapping_and_renders_once(monkeypatch, shape, index, focus):
    original = shape_views.IndexMapping
    prepared = []

    def counted_mapping(*args, **kwargs):
        mapping = original(*args, **kwargs)
        prepared.append(mapping)
        return mapping

    monkeypatch.setattr(shape_views, "IndexMapping", counted_mapping)
    array = CountingArray(shape)
    renderer = RecordingRenderer()
    visual = _index_visual(array, index, focus_first=True, renderer=renderer)
    assert len(prepared) == 1
    assert visual.index_mapping is prepared[0]
    assert len(renderer.calls) == 1
    assert renderer.calls[0][0] == "panels"
    assert visual.trace.output_coord == focus
    assert array.reads == []


@pytest.mark.parametrize(
    ("shape", "index"),
    [
        ((0, 3), ()),
        ((3, 4), (slice(0, 0),)),
        ((3, 4), np.zeros((3, 4), dtype=bool)),
    ],
)
def test_first_focus_on_empty_result_has_no_trace_or_invented_source(shape, index):
    array = CountingArray(shape)
    renderer = RecordingRenderer()
    visual = _index_visual(array, index, focus_first=True, renderer=renderer)
    assert visual.trace is None
    assert visual.index_mapping.result_count == 0
    assert visual.metadata["trace_in_explanation"] is False
    assert not renderer.panels[0].get("selected")
    assert not renderer.panels[1]["selected"]
    assert array.reads == []
    assert len(renderer.calls) == 1


@pytest.mark.parametrize("show_result", [False, True])
def test_omitted_and_none_focus_keep_the_static_display(show_result):
    args = ((3, 4), ([2, 0, 2], slice(None, None, -2)))
    omitted = rt.index(*args, show_result=show_result)
    explicit = rt.index(*args, show_result=show_result, focus=None)
    assert omitted.svg == explicit.svg
    assert omitted.explanation == explicit.explanation
    assert omitted.trace is explicit.trace is None
    assert omitted.metadata == explicit.metadata == {}
    renderer = RecordingRenderer()
    rt.index(*args, show_result=show_result, renderer=renderer)
    assert renderer.calls[0][0] == ("panels" if show_result else "tensor")
    if show_result:
        assert not renderer.panels[0].get("selected")
        assert set(renderer.panels[1]["selected"]) == set(np.ndindex((3, 2)))


def test_show_result_still_requires_boolean_with_an_explicit_focus():
    with pytest.raises(TypeError, match="show_result"):
        rt.index((3,), (), show_result="yes", focus=(1,))


def test_huge_focused_result_pins_both_coordinates_without_enumeration(monkeypatch):
    def reject_enumeration(self):
        raise AssertionError("Focusing one output must not enumerate the result or full selection")

    monkeypatch.setattr(mapping_module.IndexMapping, "__iter__", reject_enumeration)
    monkeypatch.setattr(mapping_module._AdvancedSelection, "__iter__", reject_enumeration)
    size = 10**12
    array = CountingArray((size, size))
    theme = rt.LIGHT.variant(max_cells=6, max_visible_cells=12)
    focus = (2, size // 2)
    source = (size - 1, size - 1 - focus[1])
    visual = rt.index(
        array,
        ([size - 1, 0, size - 1], slice(None, None, -1)),
        focus=focus,
        theme=theme,
    )
    assert visual.index_mapping.result_count == 3 * size
    assert visual.trace.output_coord == focus
    assert visual.trace.terms[0][0].coordinate == source
    assert source in array.reads
    assert len(array.reads) <= 4 * theme.max_visible_cells
    titles = [node.text for node in ET.fromstring(visual.svg).iter() if node.tag.endswith("title")]
    for coordinate in (source, focus):
        assert any(", ".join(map(str, coordinate)) in title for title in titles), coordinate
