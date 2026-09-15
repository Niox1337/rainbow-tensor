"""Cross-operation queries preserve contribution paths and bound work before reads."""

from dataclasses import FrozenInstanceError
from math import isnan

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.explanations import t
from rainbow_tensor.provenance.flow import Flow
from rainbow_tensor.provenance.query import ValueBudgetExceeded
from rainbow_tensor.tracing import _trace_explanation


class CountingArray:
    """Expose live scalar values and record every query to a logical array."""

    def __init__(self, shape, value=7):
        self.shape = shape
        self.current_value = value
        self.reads = []

    def __getitem__(self, coordinate):
        assert len(coordinate) == len(self.shape)
        assert all(0 <= position < size for position, size in zip(coordinate, self.shape))
        self.reads.append(coordinate)
        return self.current_value


class RecordingRenderer:
    """Capture the planned display and query its value callbacks explicitly."""

    name = "provenance-recording"
    mime_type = "text/plain"

    def __init__(self):
        self.calls = 0

    def render_tensor(self, **kwargs):
        raise AssertionError("A flow should render its operation panels together")

    def render_panels(self, **kwargs):
        self.calls += 1
        self.panels = kwargs["panels"]
        return "recorded-flow"


def repeated_chain(array):
    """Build the teaching example whose repeated row contributes twice after a transpose."""
    flow = Flow()
    source = flow.input(array, name="X")
    selected = flow.index(source, ([1, 0, 1], slice(None, None, -1)), name="picked")
    transposed = flow.transpose(selected, name="turned")
    result = flow.sum(transposed, axis=1, name="Y")
    return source, selected, transposed, result


@pytest.mark.parametrize("output,value,column", [(0, 15, 2), (1, 12, 1), (2, 9, 0)])
def test_repeated_chain_preserves_three_paths_to_two_distinct_inputs(output, value, column):
    array = np.arange(1, 7).reshape(2, 3)
    source, selected, transposed, result = repeated_chain(array)
    trace = result.trace((output,))

    assert result.value((output,)) == value
    assert trace.root == result.ref((output,))
    assert trace.complete
    assert trace.truncated_reasons == ()
    assert trace.edge_count == 9
    assert len(trace.steps) == 10
    assert trace.steps[0].operation == "sum"
    assert trace.steps[0].terms == ((1,), (2,), (3,))
    assert [trace.steps[i].reference for i in (1, 2, 3)] == [
        transposed.ref((output, i)) for i in range(3)
    ]
    assert [step.reference for step in trace.steps if step.operation == "index"] == [
        selected.ref((i, output)) for i in range(3)
    ]
    counts = {contribution.reference: contribution.count for contribution in trace.roots}
    assert counts == {source.ref((1, column)): 2, source.ref((0, column)): 1}
    assert sum(counts.values()) == 3


def test_trace_is_frozen_and_does_not_read_values():
    array = CountingArray((2, 3))
    _, _, _, result = repeated_chain(array)
    trace = result.trace((1,))
    assert array.reads == []
    with pytest.raises(FrozenInstanceError):
        trace.complete = False
    with pytest.raises(FrozenInstanceError):
        trace.steps[0].depth = 99
    with pytest.raises(FrozenInstanceError):
        trace.roots[0].count = 99


@pytest.mark.parametrize(
    "limits,reason",
    [
        ({"max_depth": 1}, "max_depth"),
        ({"max_nodes": 4}, "max_nodes"),
        ({"max_edges": 3}, "max_edges"),
    ],
)
def test_each_trace_limit_reports_omissions_without_invented_roots(limits, reason):
    array = CountingArray((2, 3))
    _, _, _, result = repeated_chain(array)
    trace = result.trace((0,), **limits)
    assert not trace.complete
    assert reason in trace.truncated_reasons
    assert trace.roots == ()
    assert array.reads == []
    assert len(trace.steps) <= limits.get("max_nodes", 80)
    assert trace.edge_count <= limits.get("max_edges", 120)
    assert max(step.depth for step in trace.steps) <= limits.get("max_depth", 6)
    assert any(not step.complete for step in trace.steps)


def test_partial_product_does_not_claim_all_factors_were_expanded():
    flow = Flow()
    left = flow.input((2, 3), name="A")
    right = flow.input((3, 2), name="B")
    result = flow.matmul(left, right)
    trace = result.trace((0, 0), max_edges=1)
    assert trace.steps[0].term_count == 3
    assert trace.steps[0].terms == ()
    assert not trace.steps[0].complete
    assert trace.roots == ()
    assert trace.edge_count == 0
    assert not trace.complete
    assert "max_edges" in trace.truncated_reasons


def test_huge_reduction_only_visits_bounded_prefix_without_reading_array():
    flow = Flow()
    array = CountingArray((10**30,))
    source = flow.input(array)
    result = flow.sum(source)
    trace = result.trace(max_nodes=5, max_edges=4)
    assert trace.steps[0].term_count == 10**30
    assert len(trace.steps) == 5
    assert trace.edge_count == 4
    assert {item.reference.coordinate for item in trace.roots} == {(0,), (1,), (2,), (3,)}
    assert sum(item.count for item in trace.roots) == 4
    assert trace.truncated_reasons == ("max_edges", "max_nodes")
    assert array.reads == []


def test_shared_branch_counts_paths_but_reads_each_source_once_per_query():
    flow = Flow()
    array = CountingArray((1,), value=4)
    source = flow.input(array, name="X")
    branch = flow.reshape(source, (), name="scalar")
    stacked = flow.stack((branch, branch), name="twice")
    result = flow.sum(stacked)
    trace = result.trace()
    assert trace.complete
    assert len(trace.roots) == 1
    assert trace.roots[0].reference == source.ref((0,))
    assert trace.roots[0].count == 2
    assert result.value() == 8
    assert array.reads == [(0,)]


def test_nested_means_preserve_local_divisors_and_operation_groups():
    flow = Flow()
    array = np.arange(1, 7).reshape(2, 3)
    source = flow.input(array)
    rows = flow.mean(source, axis=1)
    result = flow.mean(rows)
    trace = result.trace()
    assert result.value() == array.mean()
    assert trace.steps[0].operation == "mean"
    assert trace.steps[0].divisor == 2
    assert trace.steps[0].terms == ((1,), (2,))
    assert [trace.steps[index].divisor for index in (1, 2)] == [3, 3]
    assert all(item.count == 1 for item in trace.roots)
    assert len(trace.roots) == 6


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("max_depth", -1, ValueError),
        ("max_nodes", 0, ValueError),
        ("max_edges", 0, ValueError),
        ("max_nodes", True, TypeError),
        ("max_edges", 2.5, TypeError),
        ("max_depth", None, TypeError),
    ],
)
def test_invalid_structural_limits_fail_without_reads(name, value, error):
    flow = Flow()
    array = CountingArray((3,))
    result = flow.sum(flow.input(array))
    with pytest.raises(error, match=name):
        result.trace(**{name: value})
    assert array.reads == []


@pytest.mark.parametrize(
    "focus,error",
    [
        ([0], TypeError),
        ((True,), TypeError),
        ((1.5,), TypeError),
        ((), ValueError),
        ((3,), IndexError),
        ((-4,), IndexError),
    ],
)
def test_invalid_focus_fails_before_tracing_or_evaluating_inputs(focus, error):
    flow = Flow()
    array = CountingArray((3,))
    result = flow.reshape(flow.input(array), (3,))
    with pytest.raises(error):
        result.trace(focus)
    with pytest.raises(error):
        result.value(focus)
    assert array.reads == []


def test_negative_result_coordinate_is_normalized_through_entire_chain():
    source, _, _, result = repeated_chain(np.arange(1, 7).reshape(2, 3))
    trace = result.trace((-1,))
    assert trace.root == result.ref((2,))
    assert result.value((-1,)) == 9
    assert {item.reference for item in trace.roots} == {source.ref((0, 0)), source.ref((1, 0))}


def test_total_value_budget_counts_terms_factors_and_unique_inputs():
    flow = Flow()
    array = CountingArray((3,), value=2)
    result = flow.sum(flow.input(array))
    with pytest.raises(ValueBudgetExceeded) as caught:
        result.value(max_terms=3, max_total_terms=8)
    assert caught.value.reason == "max_total_terms"
    assert caught.value.total_terms == 9
    assert array.reads == []
    assert result.value(max_terms=3, max_total_terms=9) == 6
    assert array.reads == [(0,), (1,), (2,)]


def test_many_operand_product_cannot_evade_total_work_budget():
    flow = Flow()
    arrays = [CountingArray((), value=2), CountingArray((), value=3), CountingArray((), value=5)]
    inputs = [flow.input(array) for array in arrays]
    result = flow.einsum(",,->", *inputs)
    with pytest.raises(ValueBudgetExceeded, match="max_total_terms"):
        result.value(max_terms=1, max_total_terms=6)
    assert all(array.reads == [] for array in arrays)
    assert result.value(max_terms=1, max_total_terms=7) == 30
    assert all(array.reads == [()] for array in arrays)


def test_oversized_inner_reduction_rejects_before_any_input_reads():
    flow = Flow()
    array = CountingArray((10**30,))
    total = flow.sum(flow.input(array))
    result = flow.reshape(total, (1,))
    with pytest.raises(ValueBudgetExceeded) as caught:
        result.value((0,), max_terms=10)
    assert caught.value.reason == "max_terms"
    assert array.reads == []


def test_value_depth_is_bounded_without_recursing_through_entire_chain():
    flow = Flow()
    array = CountingArray((1,))
    result = flow.input(array)
    for _ in range(100):
        result = flow.reshape(result, (1,))
    with pytest.raises(ValueBudgetExceeded) as caught:
        result.value((0,))
    assert caught.value.reason == "max_value_depth"
    assert array.reads == []
    trace = result.trace((0,), max_depth=3)
    assert len(trace.steps) == 4
    assert trace.truncated_reasons == ("max_depth",)


@pytest.mark.parametrize(
    "keyword,value,error",
    [
        ("max_terms", -1, ValueError),
        ("max_total_terms", -1, ValueError),
        ("max_terms", True, ValueError),
        ("max_total_terms", 0.5, ValueError),
    ],
)
def test_invalid_value_limits_fail_before_reads(keyword, value, error):
    flow = Flow()
    array = CountingArray((1,))
    result = flow.sum(flow.input(array))
    with pytest.raises(error, match=keyword):
        result.value(**{keyword: value})
    assert array.reads == []


def test_each_query_refreshes_values_without_changing_origins():
    flow = Flow()
    array = CountingArray((2,), value=3)
    result = flow.sum(flow.input(array))
    before = result.trace()
    assert result.value() == 6
    array.current_value = 11
    assert result.value() == 22
    assert result.trace() == before
    assert array.reads == [(0,), (1,), (0,), (1,)]


def test_shape_change_in_later_operand_rejects_before_reading_earlier_one():
    flow = Flow()
    first = CountingArray((2,))
    second = CountingArray((2,))
    result = flow.sum(flow.concatenate((flow.input(first), flow.input(second))))
    second.shape = (3,)
    with pytest.raises(ValueError, match="changed shape"):
        result.value()
    assert first.reads == second.reads == []


@pytest.mark.parametrize("operation,expected", [("sum", 0), ("mean", None)])
def test_empty_reduction_has_no_fabricated_input_and_keeps_undefined_mean(operation, expected):
    flow = Flow()
    array = CountingArray((0,))
    result = getattr(flow, operation)(flow.input(array))
    trace = result.trace()
    assert trace.complete
    assert trace.steps[0].term_count == 0
    assert trace.steps[0].terms == ()
    assert trace.roots == ()
    assert trace.edge_count == 0
    value = result.value()
    assert isnan(value) if expected is None else value == expected
    assert trace.steps[0].divisor == (0 if operation == "mean" else 1)
    assert array.reads == []


def test_empty_output_has_no_focus_or_value():
    flow = Flow()
    array = CountingArray((0, 3))
    result = flow.sum(flow.input(array), axis=1)
    trace = result.trace()
    assert trace.root is None
    assert trace.steps == trace.roots == ()
    assert trace.complete
    with pytest.raises(IndexError, match="empty"):
        result.trace((0,))
    with pytest.raises(IndexError, match="empty"):
        result.value()
    assert array.reads == []


def test_flow_visual_matches_intermediate_arrays_and_explains_repeated_sources():
    array = np.arange(1, 7).reshape(2, 3)
    source, selected, transposed, result = repeated_chain(array)
    renderer = RecordingRenderer()
    visual = result.visualize((0,), renderer=renderer)
    expected = [array, array[[1, 0, 1], ::-1], array[[1, 0, 1], ::-1].T, np.array([15, 12, 9])]
    assert renderer.calls == 1
    assert len(renderer.panels) == 4
    for panel, values in zip(renderer.panels, expected):
        assert panel["shape"] == values.shape
        for coordinate in np.ndindex(values.shape):
            assert panel["value_fn"](coordinate) == values[coordinate]
    assert renderer.panels[0]["selected"] == [(0, 2), (1, 2)]
    assert renderer.panels[-1]["selected"] == [(0,)]
    assert visual.trace.operation == "sum"
    assert visual.provenance == result.trace((0,))
    assert visual.metadata["trace_in_explanation"] is True
    assert visual.metadata["provenance"]["panel_nodes"] == tuple(
        (node.node_id, node.output_port) for node in (source, selected, transposed, result)
    )
    assert visual.metadata["value_evaluation"]["status"] == "evaluated"
    assert visual.mime_type == "text/plain"
    assert visual.renderer == renderer.name


def test_panel_limit_keeps_final_result_and_reports_omitted_panels():
    _, _, _, result = repeated_chain(np.arange(1, 7).reshape(2, 3))
    renderer = RecordingRenderer()
    visual = result.visualize((0,), max_panels=1, renderer=renderer)
    assert len(renderer.panels) == 1
    assert renderer.panels[0]["shape"] == (3,)
    assert renderer.panels[0]["value_fn"]((0,)) == 15
    assert visual.metadata["provenance"]["omitted_panels"] == 3
    assert visual.metadata["output_panel_indices"] == (0,)
    assert visual.provenance.complete


def test_over_budget_flow_visual_reads_nothing_and_exposes_unknown_values():
    flow = Flow()
    array = CountingArray((10**12,))
    result = flow.sum(flow.input(array))
    renderer = RecordingRenderer()
    visual = result.visualize(max_terms=10, renderer=renderer)
    assert array.reads == []
    assert visual.metadata["value_evaluation"]["status"] == "skipped"
    assert visual.metadata["value_evaluation"]["reason"] == "max_terms"
    assert renderer.panels[0]["value_fn"]((0,)) == "?"
    assert renderer.panels[-1]["value_fn"](()) == "?"
    assert not visual.provenance.complete
    assert visual.trace.term_count == 10**12
    assert len(visual.trace.terms) == 8


def test_custom_renderer_cannot_expand_planned_value_work():
    flow = Flow()
    array = CountingArray((1000,))
    result = flow.reshape(flow.input(array), (1000,))
    renderer = RecordingRenderer()
    result.visualize(
        (300,), renderer=renderer, theme=rt.LIGHT.variant(max_cells=4, max_visible_cells=4)
    )
    reads_before = list(array.reads)
    assert (300,) in reads_before
    assert len(reads_before) <= 4
    assert renderer.panels[-1]["value_fn"]((500,)) == "?"
    assert renderer.panels[0]["value_fn"]((500,)) == "?"
    assert array.reads == reads_before


def test_visual_refresh_recomputes_values_across_every_panel():
    flow = Flow()
    array = CountingArray((2,), value=3)
    source = flow.input(array)
    result = flow.sum(flow.reshape(source, (1, 2)), axis=1)
    first = RecordingRenderer()
    before = result.visualize(renderer=first)
    array.current_value = 8
    second = RecordingRenderer()
    after = result.visualize(renderer=second)
    assert first.panels[0]["value_fn"]((0,)) == 3
    assert first.panels[-1]["value_fn"]((0,)) == 6
    assert second.panels[0]["value_fn"]((0,)) == 8
    assert second.panels[-1]["value_fn"]((0,)) == 16
    assert before.provenance == after.provenance


@pytest.mark.parametrize("limit,error", [(0, ValueError), (True, TypeError), (1.5, TypeError)])
def test_invalid_panel_limit_fails_without_reads_or_render(limit, error):
    flow = Flow()
    array = CountingArray((2,))
    result = flow.sum(flow.input(array))
    renderer = RecordingRenderer()
    with pytest.raises(error, match="max_panels"):
        result.visualize(max_panels=limit, renderer=renderer)
    assert array.reads == []
    assert renderer.calls == 0


def test_empty_flow_visual_has_no_invented_scalar_panel():
    flow = Flow()
    array = CountingArray((0, 3))
    result = flow.sum(flow.input(array), axis=1)
    renderer = RecordingRenderer()
    visual = result.visualize(renderer=renderer)
    assert visual.result_shape == (0,)
    assert visual.trace is None
    assert visual.provenance.root is None
    assert renderer.panels[-1]["shape"] == (0,)
    assert renderer.panels[-1]["selected"] == []
    assert array.reads == []


@pytest.mark.parametrize("operation,focus", [("stack", (1, 0)), ("concatenate", (2,))])
def test_local_trace_keeps_second_operand_when_inputs_share_the_same_node(operation, focus):
    flow = Flow()
    source = flow.input(np.array([10, 20]))
    result = getattr(flow, operation)((source, source))
    visual = result.visualize(focus, renderer=RecordingRenderer())
    assert visual.trace.terms[0][0].operand == 1
    assert visual.trace.terms[0][0].coordinate == (0,)
    assert visual.provenance.roots[0].reference == source.ref((0,))


def test_local_broadcast_trace_preserves_second_output_port_with_reused_input():
    flow = Flow()
    source = flow.input(np.array([10, 20]))
    first, second = flow.broadcast(source, source)
    assert first.node_id == second.node_id
    assert (first.output_port, second.output_port) == (0, 1)
    visual = second.visualize((1,), renderer=RecordingRenderer())
    assert visual.trace.terms[0][0].operand == 1
    assert visual.provenance.root == second.ref((1,))
    assert visual.provenance.roots[0].reference == source.ref((1,))


def test_singleton_mean_performs_division_even_with_divisor_one():
    flow = Flow()
    result = flow.mean(flow.input(np.array([7], dtype=int)))
    value = result.value()
    assert value == 7.0
    assert isinstance(value, float)


def test_generated_value_disclosure_survives_hidden_ancestry():
    flow = Flow()
    source = flow.input((2, 3))
    result = flow.sum(flow.transpose(source))
    visual = result.visualize(max_depth=0, max_panels=1, renderer=RecordingRenderer())
    assert visual.metadata["numeric_semantics"]["operand_sources"] == ["generated_values"]
    assert t("flow.generated") in visual.explanation
    assert not visual.provenance.complete


def test_many_operand_local_trace_obeys_the_requested_ancestry_limits():
    flow = Flow()
    source = flow.input((1,))
    result = flow.einsum(",".join(["i"] * 200) + "->i", *([source] * 200))
    visual = result.visualize(
        (0,), max_nodes=1, max_edges=1, max_total_terms=1, renderer=RecordingRenderer()
    )
    assert len(visual.provenance.steps) == 1
    assert visual.provenance.edge_count == 0
    assert visual.trace.term_count == 1
    assert visual.trace.terms == ()
    assert not visual.trace.complete


@pytest.mark.parametrize("kind", ["tracked", "visual"])
def test_flow_input_rejects_recorded_and_rendered_objects_as_array_boundaries(kind):
    flow = Flow()
    source = flow.input(np.arange(3)) if kind == "tracked" else rt.shape(np.arange(3))
    with pytest.raises(TypeError):
        flow.input(source)


def test_broadcast_numeric_source_disclosure_follows_each_output_port():
    flow = Flow()
    array = flow.input(np.arange(3))
    generated = flow.input((3,))
    array_output, generated_output = flow.broadcast(array, generated)
    array_visual = array_output.visualize((1,), renderer=RecordingRenderer())
    generated_visual = generated_output.visualize((1,), renderer=RecordingRenderer())
    assert array_output.numeric_sources == frozenset(("array_values",))
    assert generated_output.numeric_sources == frozenset(("generated_values",))
    assert array_visual.metadata["numeric_semantics"]["operand_sources"] == ["array_values"]
    assert generated_visual.metadata["numeric_semantics"]["operand_sources"] == ["generated_values"]
    assert t("flow.generated") not in array_visual.explanation
    assert t("flow.generated") in generated_visual.explanation


@pytest.mark.parametrize("operation", ["index", "reshape"])
def test_unexpanded_local_identity_trace_explains_omitted_source(operation):
    flow = Flow()
    source = flow.input(np.arange(3))
    result = flow.index(source, ()) if operation == "index" else flow.reshape(source, (3,))
    visual = result.visualize((1,), max_depth=0, renderer=RecordingRenderer())
    assert visual.trace.terms == ()
    assert not visual.trace.complete
    lines = _trace_explanation(visual.trace)
    assert t("trace.omitted", shown=0, omitted=1) in lines[0]
    assert lines[1] == t("trace.equation", coordinate="1", expression="...")
