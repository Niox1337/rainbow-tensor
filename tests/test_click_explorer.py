"""Notebook clicks preserve coordinate provenance and static export semantics."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor._svg.interaction import capture_cells, capturing_cells
from rainbow_tensor.renderers import SvgRenderer


@pytest.fixture
def explorers():
    """Exercise required click widgets and release their communications."""
    created = []

    def make(operation, *args, **kwargs):
        result = rt.explore(operation, *args, **kwargs)
        created.append(result)
        return result

    yield make
    for result in created:
        result.close()


def click(explorer, target, **overrides):
    """Deliver the browser's custom message through the widget's dispatch path."""
    content = {
        "type": "focus", "revision": explorer.figure.revision,
        "coordinate": json.dumps(target), **overrides,
    }
    explorer.figure._handle_custom_msg(content, [])


def buttons(explorer):
    """Read only real result-cell controls from the rendered document."""
    return [
        element for element in ET.fromstring(explorer.figure.value).iter()
        if "data-rt-coordinate" in element.attrib
    ]


def test_click_selects_repeated_gather_and_retains_clean_static_export(explorers):
    source = np.arange(6).reshape(2, 3)
    selection = ([1, 0, 1], slice(None, None, -1))
    explorer = explorers(rt.index, source, selection)
    assert len(buttons(explorer)) == 9
    click(explorer, (2, 1))
    assert explorer.focus == (2, 1)
    assert explorer.visual.trace.terms[0][0].coordinate == (1, 1)
    assert tuple(control.value for control in explorer.coordinates) == (2, 1)
    assert explorer.visual.svg == rt.index(source, selection, focus=(2, 1)).svg
    assert "data-rt-coordinate" not in explorer.visual.svg
    selected = [cell for cell in buttons(explorer) if cell.get("aria-pressed") == "true"]
    assert [json.loads(cell.get("data-rt-coordinate")) for cell in selected] == [[2, 1]]


@pytest.mark.parametrize("operation,args,kwargs,coordinate,source", [
    (rt.reshape, ((2, 3), (3, 2)), {}, (2, 1), (1, 2)),
    (rt.transpose, ((2, 3),), {}, (2, 1), (1, 2)),
    (rt.swapaxes, ((2, 3), 0, 1), {}, (2, 1), (1, 2)),
    (rt.moveaxis, ((2, 3), 0, 1), {}, (2, 1), (1, 2)),
    (rt.squeeze, ((1, 2, 3),), {}, (1, 2), (0, 1, 2)),
    (rt.expand_dims, ((2, 3), 1), {}, (1, 0, 2), (1, 2)),
    (rt.concatenate, ([(2, 3), (1, 3)],), {}, (2, 1), (0, 1)),
    (rt.stack, ([(2, 3), (2, 3)],), {}, (1, 1, 2), (1, 2)),
    (rt.repeat, ((2, 3), 2), {"axis": 1}, (1, 4), (1, 2)),
    (rt.take, ((2, 3), [2, 0, 2]), {"axis": 1}, (1, 2), (1, 2)),
    (rt.broadcast, ((1, 3), (2, 1)), {}, (1, 2), (0, 2)),
    (rt.broadcast, ((1, 3), (2, 1)), {"focus_operand": 1}, (1, 2), (1, 0)),
])
def test_mapped_operations_start_at_first_output_and_accept_clicks(
    explorers, operation, args, kwargs, coordinate, source,
):
    explorer = explorers(operation, *args, **kwargs)
    assert explorer.focus == (0,) * len(explorer.result_shape)
    click(explorer, coordinate)
    assert explorer.focus == coordinate
    assert explorer.visual.trace.terms[0][0].coordinate == source
    assert len(buttons(explorer)) == np.prod(explorer.result_shape)


def test_identical_source_and_result_panels_annotate_only_result(explorers):
    explorer = explorers(rt.transpose, np.zeros((2, 2)))
    root = ET.fromstring(explorer.figure.value)
    groups = [child for child in root if "transform" in child.attrib]
    source_buttons = [node for node in groups[0].iter() if "data-rt-coordinate" in node.attrib]
    result_buttons = [node for node in groups[1].iter() if "data-rt-coordinate" in node.attrib]
    assert source_buttons == []
    assert len(result_buttons) == 4


@pytest.mark.parametrize("override", [
    {"coordinate": [True, 0]}, {"coordinate": [1.0, 0]},
    {"coordinate": "1,0"}, {"coordinate": {"0": 1}},
    {"coordinate": [2, 0]}, {"coordinate": [-1, 0]},
    {"coordinate": [1]}, {"coordinate": []},
    {"revision": -1}, {"revision": True}, {"type": "source"},
])
def test_invalid_or_stale_messages_preserve_previous_visual(explorers, override):
    explorer = explorers(rt.transpose, (2, 2))
    before = explorer.visual
    figure = explorer.figure.value
    click(explorer, (1, 0), **override)
    assert explorer.visual is before
    assert explorer.figure.value == figure
    assert explorer.focus == (0, 0)


def test_hidden_positions_and_ellipsis_cannot_send_clicks(explorers):
    explorer = explorers(rt.reshape, (100,), (100,), theme=rt.LIGHT.variant(max_cells=6))
    before = explorer.visual
    assert all(cell.get("role") == "button" for cell in buttons(explorer))
    assert (50,) not in explorer._clickable
    click(explorer, (50,))
    assert explorer.visual is before
    explorer.set_focus((50,))
    assert explorer.focus == (50,)
    assert (50,) in explorer._clickable


def test_scalar_clicks_and_empty_results_are_distinct(explorers):
    scalar = explorers(rt.reshape, (1,), ())
    assert scalar.focus == ()
    assert len(buttons(scalar)) == 1
    before = scalar.visual
    click(scalar, ())
    assert scalar.visual is not before
    empty = explorers(rt.reshape, (0, 2), (2, 0))
    assert empty.focus is None
    assert buttons(empty) == []
    assert empty.update_button.disabled


def test_click_failure_keeps_previous_figure_and_focus(explorers):
    source = np.arange(6).reshape(2, 3)
    explorer = explorers(rt.sum, source, axis=1)
    before = explorer.visual
    markup = explorer.figure.value
    revision = explorer.figure.revision
    source.shape = (3, 2)
    click(explorer, (1,))
    assert explorer.visual is before
    assert explorer.figure.value == markup
    assert explorer.figure.revision == revision
    assert explorer.focus == (0,)
    assert "changed" in explorer.status.value


def test_closed_explorer_ignores_browser_messages(explorers):
    explorer = explorers(rt.transpose, (2, 2))
    before = explorer.visual
    explorer.close()
    click(explorer, (1, 1))
    assert explorer.visual is before


def test_hover_free_renderer_still_provides_clickable_coordinates(explorers):
    class NoHover(SvgRenderer):
        def render_panels(self, **kwargs):
            return super().render_panels(**kwargs, hover=False)

    explorer = explorers(rt.transpose, (2, 3), renderer=NoHover())
    assert len(buttons(explorer)) == 6
    click(explorer, (2, 1))
    assert explorer.visual.trace.terms[0][0].coordinate == (1, 2)


def test_custom_svg_without_cell_metadata_keeps_coordinate_controls(explorers):
    class Plain(SvgRenderer):
        def render_panels(self, **kwargs):
            return '<svg xmlns="http://www.w3.org/2000/svg"><text>Custom</text></svg>'

    explorer = explorers(rt.transpose, (2, 3), renderer=Plain())
    assert buttons(explorer) == []
    explorer.coordinates[0].value = 2
    explorer.coordinates[1].value = 1
    explorer.update_button.click()
    assert explorer.focus == (2, 1)


def test_wide_value_layout_retry_records_cells_once(explorers):
    array = np.array([[1234567890123456789, 987654321098765432]])
    explorer = explorers(rt.transpose, array)
    assert len(buttons(explorer)) == 2
    assert len(explorer._panels) == 2


def test_capture_does_not_read_values_again(explorers):
    class CountingArray:
        shape = (2, 3)

        def __init__(self):
            self.reads = []

        def __getitem__(self, coordinate):
            self.reads.append(coordinate)
            return coordinate[0] * 3 + coordinate[1]

    static = CountingArray()
    interactive = CountingArray()
    rt.transpose(static, focus=(0, 0))
    explorers(rt.transpose, interactive)
    assert interactive.reads == static.reads


def test_browser_coordinates_preserve_integers_larger_than_javascript_precision(explorers):
    size = 2 ** 54 + 5
    explorer = explorers(rt.reshape, (size,), (size,))
    click(explorer, (size - 1,))
    assert explorer.focus == (size - 1,)
    assert explorer.visual.trace.terms[0][0].coordinate == (size - 1,)


def test_recorded_flow_click_retraces_original_inputs(explorers):
    flow = rt.Flow()
    source = flow.input(np.arange(1, 7).reshape(2, 3), name="X")
    sampled = flow.index(source, ([1, 0, 1], slice(None, None, -1)), name="A")
    turned = flow.transpose(sampled, name="B")
    result = flow.sum(turned, axis=1, name="Y")
    explorer = explorers(result)
    click(explorer, (2,))
    assert explorer.focus == (2,)
    assert explorer.visual.provenance is not None
    assert {
        root.reference.coordinate: root.count
        for root in explorer.visual.provenance.roots
    } == {(1, 0): 2, (0, 0): 1}
    assert len(buttons(explorer)) == 3
    assert explorer.visual.svg == result.visualize(focus=(2,)).svg


def test_capture_restores_outer_context_even_after_failure():
    assert not capturing_cells()
    with capture_cells() as outer:
        rt.shape((2,))
        with pytest.raises(RuntimeError), capture_cells() as inner:
            rt.shape((3,))
            raise RuntimeError("stop")
        rt.shape((4,))
    assert len(outer) == 2
    assert len(inner) == 1
    assert not capturing_cells()


def test_static_import_does_not_load_browser_widget_dependency():
    subprocess.run([
        sys.executable, "-c",
        "import sys\nimport rainbow_tensor as rt\nrt.transpose((2, 3))\n"
        "assert 'anywidget' not in sys.modules\nassert 'ipywidgets' not in sys.modules",
    ], check=True, capture_output=True, text=True)
