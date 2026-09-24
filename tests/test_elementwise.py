"""Elementwise arithmetic preserves ordered operands and bounded scalar reads."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.layout import build_layout
from rainbow_tensor.ops.elementwise import evaluate_binary, normalize_operand
from rainbow_tensor.theme import LIGHT
from rainbow_tensor.views import elementwise as views


@pytest.mark.parametrize("operation,left,right,expected", [
    ("add", 7, 3, 10),
    ("subtract", 7, 3, 4),
    ("subtract", 3, 7, -4),
    ("multiply", 7, 3, 21),
    ("divide", 7, 2, 3.5),
    ("divide", 2, 8, 0.25),
    ("add", 2 + 3j, 4 - 2j, 6 + 1j),
    ("multiply", 10**100, 10**100, 10**200),
])
def test_binary_scalar_rules_preserve_order_and_python_precision(operation, left, right, expected):
    assert evaluate_binary(operation, left, right) == expected


@pytest.mark.parametrize("numerator", [0, 1, -1, 3.5, 2 + 1j])
@pytest.mark.parametrize("denominator", [0, 0.0, -0.0, 0j])
def test_binary_division_rejects_every_zero_denominator(numerator, denominator):
    with pytest.raises(ZeroDivisionError):
        evaluate_binary("divide", numerator, denominator)


def test_binary_rules_reject_an_unknown_operation():
    with pytest.raises(ValueError, match="unsupported binary operation"):
        evaluate_binary("power", 2, 3)


@pytest.mark.parametrize("literal", [0, 2, -3, 2.5, 1 + 2j, True])
def test_numeric_literals_are_immutable_scalar_operands(literal):
    operand = normalize_operand(literal)
    assert operand.shape == ()
    assert operand[()] == literal
    assert type(operand[()]) is type(literal)
    with pytest.raises(IndexError, match="coordinate"):
        operand[(0,)]
    with pytest.raises(FrozenInstanceError):
        operand.value = 99


@pytest.mark.parametrize("operand", [(2, 3), (), np.array([2, 3]), np.array(4), np.int64(5)])
def test_normalization_preserves_arrays_and_shape_placeholders(operand):
    assert normalize_operand(operand) is operand


def test_operand_normalization_never_reads_array_values():
    class UnreadableArray:
        shape = (3,)

        def __getitem__(self, coordinate):
            raise AssertionError("normalizing an operand must not read its values")

    operand = UnreadableArray()
    assert normalize_operand(operand) is operand


class RecordingRenderer:
    """Expose panel callbacks without asking a backend to read any values."""

    name = "elementwise-recording"
    mime_type = "text/plain"

    def render_tensor(self, **kwargs):
        raise AssertionError("binary arithmetic should render three panels")

    def render_panels(self, **kwargs):
        self.panels = kwargs["panels"]
        self.theme = kwargs["theme"]
        return "recorded elementwise panels"


class CountingArray:
    """Check coordinate validity and record bounded reads from a logical tensor."""

    def __init__(self, shape, value=7):
        self.shape = shape
        self.value = value
        self.reads = []

    def __getitem__(self, coordinate):
        assert len(coordinate) == len(self.shape)
        assert all(0 <= position < size for position, size in zip(coordinate, self.shape))
        self.reads.append(coordinate)
        return self.value


OPERATIONS = ("add", "subtract", "multiply", "divide")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("left_shape,right_shape", [
    ((2, 1), (3,)), ((2, 3), (3,)), ((), (2, 3)), ((2, 0), (1, 0)), ((), ()),
])
def test_static_binary_values_match_numpy_broadcasting(operation, left_shape, right_shape):
    left = np.arange(np.prod(left_shape), dtype=float).reshape(left_shape) + 1
    right = np.arange(np.prod(right_shape), dtype=float).reshape(right_shape) + 2
    expected = getattr(np, operation)(left, right)
    renderer = RecordingRenderer()
    visual = getattr(views, operation)(left, right, renderer=renderer)
    assert visual.result_shape == expected.shape
    value = renderer.panels[-1]["value_fn"]
    for coordinate in np.ndindex(expected.shape):
        assert value(coordinate) == pytest.approx(expected[coordinate])
    if expected.size:
        assert visual.trace.expression.operator == operation
        assert visual.trace.terms == ()
        assert visual.trace.complete
        assert visual.trace.term_count == 1
    else:
        assert visual.trace is None
        assert visual.metadata["value_evaluation"]["total_terms"] == 0


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("literal_on_left", [False, True])
def test_static_binary_views_accept_numeric_literals_in_either_role(operation, literal_on_left):
    array = np.array([2, 4])
    operands = (3, array) if literal_on_left else (array, 3)
    renderer = RecordingRenderer()
    visual = getattr(views, operation)(*operands, focus=(1,), renderer=renderer)
    expected = getattr(np, operation)(*operands)
    assert renderer.panels[-1]["value_fn"]((1,)) == pytest.approx(expected[1])
    literal_position = 0 if literal_on_left else 1
    assert visual.trace.expression.operands[literal_position].coordinate == ()
    assert visual.metadata["numeric_semantics"]["operand_sources"] == [
        "array_values", "array_values",
    ]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_static_binary_focus_keeps_broadcast_sources_and_ordered_roles(operation):
    left, right = CountingArray((2, 1)), CountingArray((3,))
    renderer = RecordingRenderer()
    visual = getattr(views, operation)(left, right, focus=(-1, -1), renderer=renderer)
    expression = visual.trace.expression
    assert [(ref.operand, ref.coordinate) for ref in expression.operands] == [
        (0, (1, 0)), (1, (2,)),
    ]
    assert renderer.panels[0]["selected"] == [(1, 0)]
    assert renderer.panels[1]["selected"] == [(2,)]
    assert renderer.panels[2]["selected"] == [(1, 2)]
    assert visual.metadata["output_panel_indices"] == (2,)
    assert visual.metadata["focused_output_panel"] == 2
    assert left.reads == right.reads == []


def test_reusing_an_array_keeps_two_operand_roles_in_a_binary_expression():
    source = CountingArray((2,))
    visual = views.subtract(source, source, renderer=RecordingRenderer())
    assert [(ref.operand, ref.coordinate) for ref in visual.trace.expression.operands] == [
        (0, (0,)), (1, (0,)),
    ]
    assert source.reads == []


@pytest.mark.parametrize("focus,error", [
    ((True, 0), TypeError), ((0,), ValueError), ((2, 0), IndexError), ([0, 0], TypeError),
])
@pytest.mark.parametrize("operation", OPERATIONS)
def test_static_binary_rejects_invalid_focus_before_any_reads(operation, focus, error):
    left, right = CountingArray((2, 3)), CountingArray((3,))
    with pytest.raises(error):
        getattr(views, operation)(left, right, focus=focus)
    assert left.reads == right.reads == []


@pytest.mark.parametrize("keyword", ["max_terms", "max_total_terms"])
@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_static_binary_validates_calculation_limits_before_reads(keyword, limit):
    left, right = CountingArray((3,)), CountingArray((3,))
    with pytest.raises(ValueError):
        views.add(left, right, **{keyword: limit})
    assert left.reads == right.reads == []


def test_static_binary_rejects_nonbroadcastable_inputs_before_reads():
    left, right = CountingArray((2,)), CountingArray((3,))
    with pytest.raises(ValueError, match="broadcast"):
        views.add(left, right)
    assert left.reads == right.reads == []


@pytest.mark.parametrize("operation", OPERATIONS)
def test_static_binary_skips_all_output_values_when_total_budget_is_exceeded(operation):
    left, right = CountingArray((4,)), CountingArray((1,))
    renderer = RecordingRenderer()
    visual = getattr(views, operation)(left, right, max_total_terms=1, renderer=renderer)
    assert visual.metadata["value_evaluation"]["status"] == "skipped"
    assert visual.metadata["value_evaluation"]["reason"] == "max_total_terms"
    assert renderer.panels[-1]["value_fn"]((0,)) == "?"
    assert renderer.panels[-1]["value_fn"]((3,)) == "?"
    assert left.reads == right.reads == []
    assert visual.trace.expression.operands[1].coordinate == (0,)


def test_static_binary_caches_outputs_and_limits_extra_renderer_requests():
    left, right = CountingArray((20,)), CountingArray((1,))
    renderer = RecordingRenderer()
    visual = views.add(left, right, theme=LIGHT.variant(max_cells=2), renderer=renderer)
    limit = visual.metadata["value_evaluation"]["output_count"]
    assert limit < 20
    value = renderer.panels[-1]["value_fn"]
    for coordinate in range(limit):
        assert value((coordinate,)) == 14
        assert value((coordinate,)) == 14
    assert value((19,)) == "?"
    assert len(left.reads) == len(right.reads) == limit


@pytest.mark.parametrize("operation", OPERATIONS)
def test_huge_binary_preview_reads_only_cells_admitted_by_its_layout(operation):
    left, right = CountingArray((10**20,)), CountingArray((), value=2)
    renderer = RecordingRenderer()
    visual = getattr(views, operation)(left, right, focus=(-1,), renderer=renderer)
    panel = renderer.panels[-1]
    layout = build_layout(
        panel["shape"], selected=panel["selected"], value_fn=panel["value_fn"],
        theme=renderer.theme,
    )
    shown = [cell for cell in layout.cells if not cell.ellipsis]
    assert len(shown) == visual.metadata["value_evaluation"]["output_count"]
    assert len(shown) < 100
    assert len(left.reads) == len(right.reads) == len(shown)
    assert (10**20 - 1,) in left.reads


@pytest.mark.parametrize("operation", OPERATIONS)
def test_empty_binary_outputs_have_no_selectable_coordinate(operation):
    left, right = CountingArray((0, 3)), CountingArray((1, 3))
    renderer = RecordingRenderer()
    visual = getattr(views, operation)(left, right, renderer=renderer)
    assert visual.trace is None
    assert visual.metadata["focused_output_panel"] is None
    assert visual.metadata["value_evaluation"]["output_count"] == 0
    assert left.reads == right.reads == []
    with pytest.raises(IndexError, match="empty"):
        getattr(views, operation)(left, right, focus=(0, 0))
    assert left.reads == right.reads == []


def test_static_binary_reports_generated_shape_values_separately_from_array_values():
    renderer = RecordingRenderer()
    visual = views.add((2, 3), np.array([10, 20, 30]), focus=(1, 2), renderer=renderer)
    assert renderer.panels[-1]["value_fn"]((1, 2)) == 35
    assert visual.metadata["numeric_semantics"]["operand_sources"] == [
        "generated_values", "array_values",
    ]
    assert "generated" in visual.text


@pytest.mark.parametrize("numerator", [0, 2])
def test_static_division_keeps_the_python_zero_denominator_error(numerator):
    with pytest.raises(ZeroDivisionError):
        views.divide(np.array([numerator]), np.array([0]))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_binary_views_render_scalar_svg_without_changing_integer_semantics(operation):
    from xml.etree import ElementTree

    visual = getattr(views, operation)(120, 2)
    assert visual.result_shape == ()
    assert visual.trace.output_coord == ()
    assert ElementTree.fromstring(visual.svg).tag.endswith("svg")
    assert visual.metadata["numeric_semantics"]["arithmetic"] == "python_scalar"


@pytest.mark.parametrize("operation", OPERATIONS)
def test_binary_views_are_available_from_the_public_package(operation):
    public = getattr(rt, operation)
    assert public is getattr(views, operation)
    visual = public(np.array([2, 4]), 2)
    assert visual.result_shape == (2,)
    assert visual.trace.expression.operator == operation


@pytest.mark.parametrize("operation", OPERATIONS)
def test_binary_explorer_updates_both_broadcast_sources_and_keeps_its_budget(operation):
    explorer = rt.explore(
        getattr(rt, operation), np.array([[2], [4]]), np.array([1, 2, 3]),
        max_total_terms=1,
    )
    try:
        assert explorer.focus == (0, 0)
        explorer.set_focus((1, 2))
        assert explorer.focus == (1, 2)
        assert [(reference.operand, reference.coordinate)
                for reference in explorer.visual.trace.expression.operands] == [
            (0, (1, 0)), (1, (2,)),
        ]
        assert explorer.visual.metadata["value_evaluation"]["status"] == "skipped"
        assert explorer.visual.metadata["value_evaluation"]["max_total_terms"] == 1
        assert "data-rt-coordinate" in explorer.figure.value
    finally:
        explorer.close()


def test_division_explorer_keeps_its_previous_visual_when_a_live_denominator_becomes_zero():
    denominator = np.array([1, 2])
    explorer = rt.explore(rt.divide, np.array([4, 8]), denominator)
    try:
        original = explorer.visual
        denominator[1] = 0
        with pytest.raises(ZeroDivisionError):
            explorer.set_focus((1,))
        assert explorer.visual is original
        assert explorer.focus == (0,)
    finally:
        explorer.close()
