"""Multi-axis views preserve values, provenance, colours, and bounded work."""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.layout import build_layout


class RecordingRenderer:
    """Capture panel contracts and optionally evaluate only their visible cells."""

    name = "multi-axis-recording"
    mime_type = "text/plain"

    def __init__(self, read_values=True):
        self.read_values = read_values

    def render_tensor(self, **kwargs):
        raise AssertionError("A reduction must render source and result panels")

    def render_panels(self, *, panels, theme, **kwargs):
        self.panels = panels
        self.values = []
        if self.read_values:
            for panel in panels:
                layout = build_layout(
                    panel["shape"], selected=panel.get("selected"),
                    value_fn=panel.get("value_fn"), theme=panel.get("theme", theme),
                )
                self.values.append({
                    cell.coord: cell.value for cell in layout.cells if not cell.ellipsis
                })
        return "recorded reduction"


class CountingArray:
    """Count scalar reads without allocating the potentially enormous shape."""

    def __init__(self, shape):
        self.shape = shape
        self.reads = 0

    def __getitem__(self, coord):
        assert len(coord) == len(self.shape)
        assert all(0 <= value < size for value, size in zip(coord, self.shape))
        self.reads += 1
        return 1


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("axis", [None, (), 0, (1,), (0, 2), (2, 0), (-1, 0), (0, 1, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_visible_results_match_numpy_for_every_axis_form(operation, axis, keepdims):
    array = np.arange(24).reshape(2, 3, 4)
    renderer = RecordingRenderer()
    visual = operation(array, axis=axis, keepdims=keepdims, renderer=renderer)
    expected = getattr(np, operation.__name__)(array, axis=axis, keepdims=keepdims)

    assert visual.result_shape == expected.shape
    for coordinate, value in renderer.values[-1].items():
        assert value == expected[coordinate if expected.shape else ()]
    assert visual.metadata["reduction"]["keepdims"] is keepdims
    assert visual.metadata["value_evaluation"]["status"] == "evaluated"


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_omitting_axis_reduces_the_entire_array(operation):
    array = np.arange(6).reshape(2, 3)
    renderer = RecordingRenderer()
    visual = operation(array, renderer=renderer)
    assert visual.result_shape == ()
    assert renderer.values[-1] == {(0,): getattr(np, operation.__name__)(array)}
    assert visual.trace.output_coord == ()
    assert visual.metadata["reduction"]["axes"] == (0, 1)


@pytest.mark.parametrize("keepdims", [False, True])
def test_multi_axis_focus_has_stable_source_order_and_mean_divisor(keepdims):
    array = np.arange(24).reshape(2, 3, 4)
    focus = (0, -2, 0) if keepdims else (-2,)
    visuals = [
        rt.mean(array, axis=axes, keepdims=keepdims, focus=focus)
        for axes in [(0, 2), (2, 0), (-1, 0)]
    ]
    expected_sources = [(i, 1, k) for i in range(2) for k in range(4)]
    assert visuals[0].trace == visuals[1].trace == visuals[2].trace
    for visual in visuals:
        assert visual.trace.output_coord == ((0, 1, 0) if keepdims else (1,))
        assert [term[0].coordinate for term in visual.trace.terms] == expected_sources
        assert list(visual.selected) == expected_sources
        assert visual.trace.divisor == visual.trace.term_count == 8
        assert visual.trace.complete
        assert visual.metadata["reduction"]["axes"] == (0, 2)


@pytest.mark.parametrize("theme", [rt.LIGHT, rt.DARK])
def test_keepdims_marks_retained_axes_and_preserves_group_colours(theme):
    renderer = RecordingRenderer()
    visual = rt.sum((2, 3, 4), axis=(0, 2), keepdims=True, theme=theme, renderer=renderer)
    source, result = renderer.panels
    assert result["shape"] == (1, 3, 1)
    assert result["theme"].axis_color(0) == theme.surface_selected
    assert result["theme"].axis_color(1) == theme.axis_color(1)
    assert result["theme"].axis_color(2) == theme.surface_selected
    assert result["caption_parts"].count(("1", theme.surface_selected)) == 2
    for group in range(3):
        assert source["cell_tint"]((1, group, 3)) == result["cell_tint"]((0, group, 0))
    assert "retains axes (0, 2) at length 1" in visual.text


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_empty_axis_tuple_keeps_one_source_term_per_output(operation):
    renderer = RecordingRenderer()
    visual = operation((2, 3), axis=(), keepdims=True, focus=(1, 2), renderer=renderer)
    assert visual.result_shape == (2, 3)
    assert visual.trace.term_count == visual.trace.divisor == 1
    assert visual.trace.terms[0][0].coordinate == (1, 2)
    assert visual.selected.count == 1
    assert visual.metadata["reduction"]["axes"] == ()
    assert "No axes are reduced" in visual.text


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("options, error", [
    ({"axis": (0, -3)}, ValueError),
    ({"axis": (0, 3)}, ValueError),
    ({"axis": (0, True)}, TypeError),
    ({"axis": (0, 1.0)}, TypeError),
    ({"axis": [0, 1]}, TypeError),
    ({"axis": None, "keepdims": 1}, TypeError),
    ({"axis": None, "keepdims": "yes"}, TypeError),
    ({"axis": (0, 2), "keepdims": True, "focus": (1, 0, 0)}, IndexError),
    ({"axis": (0, 2), "keepdims": True, "focus": (1,)}, ValueError),
])
def test_invalid_reduction_arguments_fail_before_backend_reads(operation, options, error):
    array = CountingArray((2, 3, 4))
    with pytest.raises(error):
        operation(array, **options)
    assert array.reads == 0


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("limits, reason", [
    ({"max_terms": 7}, "max_terms"),
    ({"max_terms": 8, "max_total_terms": 23}, "max_total_terms"),
])
def test_budget_uses_the_product_of_reduced_dimensions(operation, limits, reason):
    array = CountingArray((2, 3, 4))
    renderer = RecordingRenderer()
    visual = operation(array, axis=(0, 2), keepdims=True, renderer=renderer, **limits)
    assert visual.metadata["value_evaluation"]["reason"] == reason
    assert visual.metadata["value_evaluation"]["term_count"] == 8
    assert visual.metadata["value_evaluation"]["total_terms"] == 24
    assert set(renderer.values[-1].values()) == {"?"}
    assert array.reads == len(renderer.values[0])


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_trillion_term_trace_and_selection_do_not_read_or_expand_the_source(operation):
    array = CountingArray((1_000_000, 3, 1_000_000))
    visual = operation(
        array, axis=(0, 2), keepdims=True, focus=(0, 1, 0),
        renderer=RecordingRenderer(read_values=False),
    )
    assert array.reads == 0
    assert visual.trace.term_count == visual.selected.count == 10**12
    assert len(visual.trace.terms) == 8
    assert visual.trace.terms[-1][0].coordinate == (0, 1, 7)
    assert visual.trace.divisor == (10**12 if operation is rt.mean else 1)
    assert (999_999, 1, 999_999) in visual.selected
    assert not visual.trace.complete
    assert visual.metadata["value_evaluation"]["status"] == "skipped"
