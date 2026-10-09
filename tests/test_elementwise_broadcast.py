"""Automatic broadcasts show each expanded operand before the binary result."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.views import elementwise
from test_elementwise import CountingArray, RecordingRenderer

SVG = "{http://www.w3.org/2000/svg}"
OPERATIONS = (
    ("add", "+"), ("subtract", "-"), ("multiply", "*"), ("divide", "/"),
    ("greater", ">"), ("greater_equal", ">="), ("less", "<"), ("less_equal", "<="),
    ("equal", "=="), ("not_equal", "!="),
)


class BroadcastRenderer(RecordingRenderer):
    """Retain the visible operation sequence as well as the panel values."""

    def render_panels(self, **kwargs):
        self.connectors = kwargs["connectors"]
        return super().render_panels(**kwargs)


def captions(root):
    return [
        "".join(text.itertext())
        for text in root.findall(f"{SVG}text")
        if text.findall(f"{SVG}tspan")
    ]


@pytest.mark.parametrize("operation,symbol", OPERATIONS)
def test_binary_operations_show_both_expansions_with_their_own_values(operation, symbol):
    left = np.array([[2], [4]])
    right = np.array([[1, 3, 5]])
    renderer = BroadcastRenderer()
    visual = getattr(elementwise, operation)(left, right, focus=(-1, -1), renderer=renderer)

    assert [panel["shape"] for panel in renderer.panels] == [
        (2, 1), (2, 3), (1, 3), (2, 3), (2, 3),
    ]
    assert renderer.connectors == ["->", symbol, "->", "->"]
    assert [panel["selected"] for panel in renderer.panels] == [
        [(1, 0)], [(1, 2)], [(0, 2)], [(1, 2)], [(1, 2)],
    ]
    expected = getattr(np, operation)(left, right)
    for coordinate in np.ndindex(expected.shape):
        row, column = coordinate
        assert renderer.panels[1]["value_fn"](coordinate) == left[row, 0]
        assert renderer.panels[3]["value_fn"](coordinate) == right[0, column]
        assert renderer.panels[4]["value_fn"](coordinate) == expected[coordinate]
    assert visual.metadata["output_panel_indices"] == (4,)
    assert visual.metadata["focused_output_panel"] == 4
    assert [(reference.operand, reference.coordinate)
            for reference in visual.trace.expression.operands] == [
        (0, (1, 0)), (1, (0, 2)),
    ]


@pytest.mark.parametrize("expanded_on_left", [False, True])
def test_only_the_expanded_operand_gets_an_intermediate_svg_panel(expanded_on_left):
    full = np.array([[1, 2, 3], [4, 5, 6]])
    column = np.array([[10], [20]])
    operands = (column, full) if expanded_on_left else (full, column)
    visual = rt.add(*operands, focus=(1, 2), theme=rt.LIGHT)
    root = ET.fromstring(visual.svg)

    if expanded_on_left:
        expected_captions = ["A (2, 1)", "A stretched (2, 3)", "B (2, 3)", "add (2, 3)"]
        expanded_panel = 1
    else:
        expected_captions = ["A (2, 3)", "B (2, 1)", "B stretched (2, 3)", "add (2, 3)"]
        expanded_panel = 2
    assert captions(root) == expected_captions
    panels = root.findall(f"{SVG}g")
    assert [text.text for text in panels[expanded_panel].iter(f"{SVG}text")] == [
        "10", "10", "10", "20", "20", "20",
    ]
    assert [text.text for text in panels[-1].iter(f"{SVG}text")] == [
        "11", "12", "13", "24", "25", "26",
    ]
    assert visual.metadata["output_panel_indices"] == (3,)
    assert visual.metadata["focused_output_panel"] == 3


@pytest.mark.parametrize("operands", [
    (np.arange(6).reshape(2, 3), np.ones((2, 3))),
    (2, 3),
    ((0, 3), (0, 3)),
])
def test_matching_operands_keep_three_panels_without_stretched_captions(operands):
    visual = rt.add(*operands)
    assert len(captions(ET.fromstring(visual.svg))) == 3
    assert "stretched" not in visual.svg
    assert visual.metadata["output_panel_indices"] == (2,)
    assert visual.metadata["focused_output_panel"] == (None if 0 in visual.result_shape else 2)


@pytest.mark.parametrize("literal_on_left", [False, True])
def test_scalar_expansion_repeats_the_literal_and_keeps_its_scalar_source(literal_on_left):
    array = np.array([[2, 4], [6, 8]])
    operands = (3, array) if literal_on_left else (array, 3)
    renderer = BroadcastRenderer()
    visual = rt.subtract(*operands, focus=(1, 1), renderer=renderer)
    source_index, expanded_index = (0, 1) if literal_on_left else (1, 2)

    assert len(renderer.panels) == 4
    assert renderer.panels[source_index]["shape"] == ()
    assert renderer.panels[source_index]["selected"] == [()]
    assert renderer.panels[expanded_index]["shape"] == array.shape
    assert renderer.panels[expanded_index]["selected"] == [(1, 1)]
    for coordinate in np.ndindex(array.shape):
        assert renderer.panels[expanded_index]["value_fn"](coordinate) == 3
        assert renderer.panels[-1]["value_fn"](coordinate) == np.subtract(*operands)[coordinate]
    literal_position = 0 if literal_on_left else 1
    assert visual.trace.expression.operands[literal_position].coordinate == ()
    assert visual.metadata["focused_output_panel"] == 3


def test_shape_only_expansions_repeat_generated_source_values():
    renderer = BroadcastRenderer()
    visual = rt.add((2, 1), (3,), focus=(1, 2), renderer=renderer)

    assert [panel["shape"] for panel in renderer.panels] == [
        (2, 1), (2, 3), (3,), (2, 3), (2, 3),
    ]
    for row, column in np.ndindex((2, 3)):
        assert renderer.panels[1]["value_fn"]((row, column)) == row
        assert renderer.panels[3]["value_fn"]((row, column)) == column
        assert renderer.panels[-1]["value_fn"]((row, column)) == row + column
    assert visual.metadata["numeric_semantics"]["operand_sources"] == [
        "generated_values", "generated_values",
    ]


def test_stretched_svg_axes_use_the_same_accent_in_frames_and_captions():
    visual = rt.add((1, 3, 4), (2, 1, 4), theme=rt.LIGHT)
    root = ET.fromstring(visual.svg)
    panels = root.findall(f"{SVG}g")
    caption_elements = [
        text for text in root.findall(f"{SVG}text") if text.findall(f"{SVG}tspan")
    ]
    assert captions(root)[1::2] == ["A stretched (2, 3, 4)", "B stretched (2, 3, 4)"]
    accent = rt.LIGHT.surface_selected
    for panel_index, stretched_size in ((1, "2"), (3, "3")):
        parts = [(part.text, part.attrib["fill"])
                 for part in caption_elements[panel_index].findall(f"{SVG}tspan")]
        frame_strokes = [
            rect.attrib["stroke"] for rect in panels[panel_index].iter(f"{SVG}rect")
            if rect.attrib.get("fill") == "none"
        ]
        assert (stretched_size, accent) in parts
        assert accent in frame_strokes


def test_expansion_previews_remain_available_when_result_evaluation_is_skipped():
    left = CountingArray((2, 1), value=4)
    right = CountingArray((1, 3), value=2)
    renderer = BroadcastRenderer()
    visual = rt.divide(left, right, max_total_terms=1, renderer=renderer)

    assert left.reads == right.reads == []
    assert renderer.panels[-1]["value_fn"]((1, 2)) == "?"
    assert renderer.panels[1]["value_fn"]((1, 2)) == 4
    assert renderer.panels[3]["value_fn"]((1, 2)) == 2
    assert left.reads == [(1, 0)]
    assert right.reads == [(0, 2)]
    assert visual.metadata["value_evaluation"]["status"] == "skipped"


def test_empty_broadcast_panels_have_no_selection_or_source_reads():
    left = CountingArray((0, 3))
    right = CountingArray((1, 3))
    renderer = BroadcastRenderer()
    visual = rt.add(left, right, renderer=renderer)

    assert [panel["shape"] for panel in renderer.panels] == [
        (0, 3), (1, 3), (0, 3), (0, 3),
    ]
    assert all(panel["selected"] == [] for panel in renderer.panels)
    assert visual.trace is None
    assert visual.metadata["output_panel_indices"] == (3,)
    assert visual.metadata["focused_output_panel"] is None
    assert visual.metadata["value_evaluation"]["output_count"] == 0
    assert left.reads == right.reads == []


@pytest.mark.parametrize("left,right,output_index", [
    (np.array([[2], [4]]), np.array([[1, 3, 5]]), 4),
    (np.array([[2, 3, 4], [5, 6, 7]]), np.array([[1], [3]]), 3),
])
def test_explorer_clicks_only_target_the_result_after_expansion(left, right, output_index):
    explorer = rt.explore(rt.add, left, right)
    try:
        root = ET.fromstring(explorer.figure.value)
        panels = [child for child in root if "transform" in child.attrib]
        clickable = [
            [node for node in panel.iter() if "data-rt-coordinate" in node.attrib]
            for panel in panels
        ]
        assert [len(cells) for cells in clickable] == [0] * output_index + [6]
        explorer.figure._handle_custom_msg({
            "type": "focus", "revision": explorer.figure.revision,
            "coordinate": json.dumps((1, 2)),
        }, [])
        assert explorer.focus == (1, 2)
        assert explorer.visual.metadata["focused_output_panel"] == output_index
        updated = ET.fromstring(explorer.figure.value)
        selected = [node for node in updated.iter()
                    if node.get("aria-pressed") == "true"]
        assert [json.loads(node.attrib["data-rt-coordinate"]) for node in selected] == [[1, 2]]
        assert "data-rt-coordinate" not in explorer.visual.svg
    finally:
        explorer.close()
