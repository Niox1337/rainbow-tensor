"""Index exploration follows ordered outputs while refreshing mutable inputs."""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.layout import build_layout
from rainbow_tensor.views import shapes as shape_views


class RecordingSvgRenderer:
    """Record the public renderer contract while producing real SVG output."""

    name = "index-explorer-recording"
    mime_type = "image/svg+xml"

    def __init__(self):
        self.calls = []

    def render_tensor(self, **kwargs):
        self.calls.append(("tensor", kwargs))
        return rt.SVG.render_tensor(**kwargs)

    def render_panels(self, **kwargs):
        self.calls.append(("panels", kwargs))
        return rt.SVG.render_panels(**kwargs)


class CountingArray:
    """Expose a large logical shape without allocating its elements."""

    def __init__(self, shape):
        self.shape = shape
        self.reads = []

    def __getitem__(self, coordinate):
        assert len(coordinate) == len(self.shape)
        assert all(0 <= index < size for index, size in zip(coordinate, self.shape))
        self.reads.append(coordinate)
        return 1


@pytest.fixture
def explorers():
    """Exercise required notebook controls and release them after each test."""
    created = []

    def make(array, index, **kwargs):
        explorer = rt.explore(rt.index, array, index, **kwargs)
        created.append(explorer)
        return explorer

    yield make
    for explorer in created:
        explorer.close()


def _assert_trace(visual, output, source):
    """Check the single-source provenance exposed to notebook users."""
    trace = visual.trace
    assert trace.operation == "index"
    assert trace.output_coord == output
    assert trace.term_count == 1
    assert trace.terms == ((rt.OperandRef(0, source),),)
    assert trace.divisor == 1
    assert trace.complete
    assert visual.metadata["trace_in_explanation"] is True
    explanation = f"Output {output} reads source {source}."
    assert visual.text.count(explanation) == 1
    return explanation


@pytest.mark.parametrize("focus, expected", [(None, (0,)), ((-1,), (2,))])
def test_initial_index_explorer_builds_one_mapping_and_one_comparison(
    explorers, monkeypatch, focus, expected,
):
    original = shape_views.IndexMapping
    mapping_calls = []

    def record_mapping(*args, **kwargs):
        mapping = original(*args, **kwargs)
        mapping_calls.append(mapping)
        return mapping

    monkeypatch.setattr(shape_views, "IndexMapping", record_mapping)
    renderer = RecordingSvgRenderer()
    options = {} if focus is None else {"focus": focus}
    explorer = explorers(np.array([10, 20, 30]), ([2, 0, 1],), renderer=renderer, **options)

    assert len(mapping_calls) == 1
    assert explorer.visual.index_mapping is mapping_calls[0]
    assert len(renderer.calls) == 1
    assert renderer.calls[0][0] == "panels"
    assert len(renderer.calls[0][1]["panels"]) == 2
    assert explorer.focus == expected
    source = (2,) if expected == (0,) else (1,)
    explanation = _assert_trace(explorer.visual, expected, source)
    assert explorer.explanation.value.count(explanation) == 1
    assert tuple(control.value for control in explorer.coordinates) == expected
    assert tuple(control.max for control in explorer.coordinates) == (2,)


def test_duplicate_outputs_focus_separately_even_when_the_source_is_the_same(explorers):
    renderer = RecordingSvgRenderer()
    explorer = explorers(np.array([10, 20, 30]), ([2, 0, 2],), renderer=renderer)
    first = explorer.visual
    source, result = renderer.calls[-1][1]["panels"]
    assert set(source["selected"]) == {(2,)}
    assert set(result["selected"]) == {(0,)}
    assert result["value_fn"]((0,)) == result["value_fn"]((2,)) == 30
    _assert_trace(first, (0,), (2,))

    explorer.coordinates[0].value = 2
    explorer.update_button.click()
    source, result = renderer.calls[-1][1]["panels"]
    assert explorer.focus == (2,)
    assert set(source["selected"]) == {(2,)}
    assert set(result["selected"]) == {(2,)}
    _assert_trace(explorer.visual, (2,), (2,))
    assert explorer.visual.svg != first.svg


def test_reverse_slices_and_negative_focus_follow_numpy_output_order(explorers):
    array = np.arange(24).reshape(4, 6)
    index = (slice(None, None, -1), slice(None, None, -2))
    renderer = RecordingSvgRenderer()
    explorer = explorers(array, index, focus=(np.int64(-1), -1), renderer=renderer)
    expected = array[index]

    assert explorer.focus == (3, 2)
    _assert_trace(explorer.visual, (3, 2), (0, 1))
    assert renderer.calls[-1][1]["panels"][1]["value_fn"]((3, 2)) == expected[-1, -1]
    visual = explorer.set_focus((0, 1))
    _assert_trace(visual, (0, 1), (3, 3))
    assert renderer.calls[-1][1]["panels"][1]["value_fn"]((0, 1)) == expected[0, 1]


@pytest.mark.parametrize("scalar_source", [False, True])
def test_scalar_output_has_only_refresh_and_rereads_its_value(explorers, scalar_source):
    array = np.array(7) if scalar_source else np.arange(6).reshape(2, 3)
    index = () if scalar_source else (1, 2)
    source_coordinate = () if scalar_source else (1, 2)
    renderer = RecordingSvgRenderer()
    explorer = explorers(array, index, renderer=renderer)
    before = explorer.visual

    assert explorer.result_shape == ()
    assert explorer.focus == ()
    assert explorer.coordinates == ()
    assert not explorer.update_button.disabled
    _assert_trace(before, (), source_coordinate)
    array[source_coordinate] = 91
    explorer.update_button.click()
    assert explorer.visual is not before
    assert explorer.visual.svg != before.svg
    assert renderer.calls[-1][1]["panels"][1]["value_fn"](()) == 91
    _assert_trace(explorer.visual, (), source_coordinate)


@pytest.mark.parametrize("array, index", [
    (np.arange(6), (slice(0, 0),)),
    (np.empty(0), (slice(None),)),
    (np.arange(6), np.zeros(6, dtype=bool)),
])
def test_empty_index_results_have_a_comparison_but_no_focus_controls(explorers, array, index):
    renderer = RecordingSvgRenderer()
    explorer = explorers(array, index, renderer=renderer)
    before = explorer.visual
    assert renderer.calls[0][0] == "panels"
    assert explorer.result_shape == (0,)
    assert explorer.focus is None
    assert explorer.visual.trace is None
    assert explorer.coordinates == ()
    assert explorer.update_button.disabled
    assert "Empty result" in explorer.status.value
    explorer.update_button.click()
    assert explorer.visual is before
    assert len(renderer.calls) == 1
    with pytest.raises(IndexError, match="empty"):
        explorer.set_focus(())
    assert explorer.visual is before


@pytest.mark.parametrize("as_array", [False, True])
def test_updates_refresh_both_values_and_mutable_integer_indices(explorers, as_array):
    array = np.array([10, 20, 30, 40])
    indices = np.array([3, 1, 3]) if as_array else [3, 1, 3]
    renderer = RecordingSvgRenderer()
    explorer = explorers(array, (indices,), focus=(1,), renderer=renderer)
    original_trace = explorer.visual.trace

    array[1] = 77
    explorer.update_button.click()
    assert renderer.calls[-1][1]["panels"][1]["value_fn"]((1,)) == 77
    _assert_trace(explorer.visual, (1,), (1,))

    indices[1] = 0
    array[0] = 123
    visual = explorer.set_focus((1,))
    assert visual.index_mapping.source_coord((1,)) == (0,)
    assert renderer.calls[-1][1]["panels"][1]["value_fn"]((1,)) == 123
    _assert_trace(visual, (1,), (0,))
    assert original_trace.terms[0][0].coordinate == (1,)
    assert len(renderer.calls) == 3


@pytest.mark.parametrize("coordinate, error", [
    ((3,), IndexError), ((True,), TypeError), ((), ValueError),
])
def test_invalid_focus_keeps_visual_controls_and_backend_reads_unchanged(
    explorers, coordinate, error,
):
    array = CountingArray((4,))
    explorer = explorers(array, ([3, 0, 2],), focus=(1,))
    before = explorer.visual
    figure = explorer.figure.value
    reads = len(array.reads)
    with pytest.raises(error):
        explorer.set_focus(coordinate)
    assert explorer.visual is before
    assert explorer.figure.value == figure
    assert explorer.focus == (1,)
    assert tuple(control.value for control in explorer.coordinates) == (1,)
    assert len(array.reads) == reads


def test_invalid_mutated_index_keeps_the_last_successful_figure(explorers):
    indices = [2, 0, 1]
    explorer = explorers(np.arange(3), (indices,), focus=(1,))
    before = explorer.visual
    figure = explorer.figure.value
    indices[0] = 99
    explorer.update_button.click()
    assert explorer.visual is before
    assert explorer.figure.value == figure
    assert explorer.focus == (1,)
    assert "Could not update focus" in explorer.status.value


@pytest.mark.parametrize("change_source_shape", [False, True])
def test_output_shape_changes_require_a_new_explorer(explorers, change_source_shape):
    array = np.arange(6).reshape(2, 3)
    indices = [0, 1]
    index = (slice(None),) if change_source_shape else (indices,)
    explorer = explorers(array, index)
    before = explorer.visual
    focus = explorer.focus
    controls = tuple(control.value for control in explorer.coordinates)
    if change_source_shape:
        array.shape = (3, 2)
    else:
        indices.append(0)
    with pytest.raises(ValueError, match="output shape changed"):
        explorer.set_focus(focus)
    assert explorer.visual is before
    assert explorer.focus == focus
    assert tuple(control.value for control in explorer.coordinates) == controls


def test_large_reverse_index_keeps_focus_visible_and_reads_only_preview_cells(explorers):
    array = CountingArray((1_000_000_000,))
    theme = rt.LIGHT.variant(max_cells=6, max_visible_cells=12)
    renderer = RecordingSvgRenderer()
    explorer = explorers(
        array, (slice(None, None, -1),), focus=(500_000_000,),
        theme=theme, renderer=renderer,
    )
    for focus in ((500_000_000,), (750_000_000,)):
        if focus != explorer.focus:
            array.reads.clear()
            explorer.set_focus(focus)
        source_coordinate = (array.shape[0] - 1 - focus[0],)
        assert source_coordinate in array.reads
        assert 0 < len(array.reads) <= 4 * theme.max_visible_cells
        panels = renderer.calls[-1][1]["panels"]
        for panel, selected in zip(panels, (source_coordinate, focus)):
            layout = build_layout(
                panel["shape"], selected=panel.get("selected"), theme=theme,
            )
            assert selected in {cell.coord for cell in layout.cells if not cell.ellipsis}
            assert selected in panel["selected"]
        _assert_trace(explorer.visual, focus, source_coordinate)


def test_explicit_options_and_custom_svg_renderer_survive_focus_updates(explorers, tmp_path):
    array = np.array([1.234567, 2.345678, 3.456789])
    index = ([2, 0, 1],)
    theme = rt.DARK.variant(name="index-lesson", cell_w=72, max_cells=8)
    renderer = RecordingSvgRenderer()
    explorer = explorers(array, index, theme=theme, precision=4, renderer=renderer)
    explorer.set_focus((-1,))

    assert len(renderer.calls) == 2
    for method, options in renderer.calls:
        assert method == "panels"
        assert options["theme"] is theme
        assert options["precision"] == 4
        assert options["connectors"] == ["->"]
    assert explorer.visual.renderer == renderer.name
    assert explorer.visual.svg == rt.index(
        array, index, focus=(2,), theme=theme, precision=4,
    ).svg
    path = tmp_path / "index-focus.svg"
    explorer.visual.save(path)
    assert path.read_text(encoding="utf-8") == explorer.visual.svg


def test_exploration_does_not_change_static_index_defaults(explorers):
    array = np.array([10, 20, 30])
    index = ([2, 0, 2],)
    normal = rt.index(array, index)
    comparison = rt.index(array, index, show_result=True)
    assert normal.trace is None
    assert comparison.trace is None

    explorer = explorers(array, index)
    explorer.set_focus((2,))
    assert rt.index(array, index).svg == normal.svg
    assert rt.index(array, index, show_result=True).svg == comparison.svg
    assert set(normal.selected) == {(0,), (2,)}
