"""Conditional choices separate value-free candidates from evaluated sources."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rainbow_tensor.provenance.flow import Flow
from rainbow_tensor.provenance.lesson import build_lesson
from rainbow_tensor.provenance.query import ValueBudgetExceeded, evaluate_values
from rainbow_tensor.tracing import SelectionExpression, _trace_explanation
from rainbow_tensor.views.selection import where
from test_provenance import CountingArray, RecordingRenderer


@pytest.mark.parametrize("condition,left,right", [
    (np.array([[True], [False]]), np.array([1, 2, 3]), -1),
    (np.array(True), 2, 3.5),
    (np.array([0, 1, -2, np.nan]), np.array([1, 2, 3, 4]), 0),
    (np.empty((0, 3), dtype=bool), np.array([1, 2, 3]), -1),
])
def test_where_matches_numpy_values_and_three_way_broadcasting(condition, left, right):
    renderer = RecordingRenderer()
    visual = where(condition, left, right, renderer=renderer)
    expected = np.where(condition, left, right)
    assert visual.result_shape == expected.shape
    for coordinate in np.ndindex(expected.shape):
        assert renderer.panels[-1]["value_fn"](coordinate) == expected[coordinate]
    json.dumps(visual.metadata["evaluated_selections"])
    json.dumps(visual.metadata["value_evaluation"])
    if expected.size:
        assert isinstance(visual.trace.expression, SelectionExpression)
        assert visual.trace.terms == ()
        assert visual.trace.expression.candidate_count == 3
        assert [item.operand for item in visual.trace.expression.operands] == [0, 1, 2]
        assert "where(" in " ".join(_trace_explanation(visual.trace))
    else:
        assert visual.trace is None
        assert visual.provenance.steps == ()
        assert visual.metadata["evaluated_selections"] == ()


def conditional_flow(condition=None):
    flow = Flow()
    condition = flow.input(np.array([True, False]) if condition is None else condition)
    left = flow.input(np.array([10, 20]))
    right = flow.input(-3)
    return flow, condition, left, right, flow.where(condition, left, right)


def test_structural_where_trace_records_candidates_without_reading_values():
    arrays = [CountingArray((2,)), CountingArray((2,)), CountingArray(())]
    flow = Flow()
    inputs = tuple(flow.input(array) for array in arrays)
    result = flow.where(*inputs)
    trace = result.trace((1,))
    assert trace.steps[0].selection_operator == "where"
    assert trace.steps[0].terms == ()
    assert [trace.steps[index].reference for index in trace.steps[0].candidates] == [
        inputs[0].ref((1,)), inputs[1].ref((1,)), inputs[2].ref(()),
    ]
    assert all(array.reads == [] for array in arrays)
    assert trace.complete
    assert len(trace.roots) == 3


@pytest.mark.parametrize("max_edges", [1, 2])
def test_conditional_candidate_group_is_not_partially_expanded(max_edges):
    _, _, _, _, result = conditional_flow()
    trace = result.trace((0,), max_edges=max_edges)
    assert trace.steps[0].candidates == ()
    assert trace.roots == ()
    assert trace.edge_count == 0
    assert not trace.complete


@pytest.mark.parametrize("limits,error", [
    ({"max_terms": 0}, ValueError), ({"max_total_terms": 6}, ValueBudgetExceeded),
])
def test_where_exceeding_a_budget_reads_no_condition_or_branch(limits, error):
    arrays = [CountingArray((2,)), CountingArray((2,)), CountingArray(())]
    flow = Flow()
    result = flow.where(*(flow.input(array) for array in arrays))
    with pytest.raises(error):
        result.value((0,), **limits)
    assert all(array.reads == [] for array in arrays)


def test_selected_source_is_separate_immutable_and_refreshed_with_condition():
    condition_array = np.array([True, False])
    flow, condition, left, right, result = conditional_flow(condition_array)
    structural = result.trace((0,))
    values, evaluation = evaluate_values(flow, (result.ref((0,)),))
    decision = evaluation["selected_sources"][0]
    assert values[result.ref((0,))] == 10
    assert decision.output == result.ref((0,))
    assert decision.source == left.ref((0,))
    assert decision.position == 1
    assert decision.reason == "condition_true"
    assert decision.candidate_count == 3
    with pytest.raises(FrozenInstanceError):
        decision.position = 2
    condition_array[0] = False
    lesson = build_lesson(result, (0,))
    assert lesson.selected_source.source == right.ref(())
    assert lesson.selected_source.reason == "condition_false"
    assert lesson.term_value == -3
    assert lesson.factor_values == (False, 10, -3)
    assert lesson.subtotal is None
    assert lesson.selection_operator == "where"
    assert lesson.available_terms == 1
    assert result.trace((0,)) == structural
    assert lesson.trace.steps[0].terms == ()


def test_where_preserves_repeated_roles_but_reads_shared_input_once():
    array = CountingArray(())
    flow = Flow()
    source = flow.input(array)
    result = flow.where(flow.input(True), source, source)
    trace = result.trace()
    root = next(item for item in trace.roots if item.reference == source.ref(()))
    assert root.count == 2
    assert result.value() == 7
    assert array.reads == [()]


def test_where_conservatively_evaluates_an_unused_failing_branch():
    flow = Flow()
    safe = flow.input(3)
    unsafe = flow.divide(safe, flow.input(0))
    result = flow.where(flow.input(True), safe, unsafe)
    assert result.trace().complete
    with pytest.raises(ZeroDivisionError):
        result.value()
    with pytest.raises(ValueBudgetExceeded):
        result.value(max_total_terms=1)


def test_where_budget_failure_has_no_evaluated_choice_in_lesson_or_visual():
    _, _, _, _, result = conditional_flow()
    lesson = build_lesson(result, (0,), max_total_terms=1)
    assert lesson.selected_source is None
    assert lesson.output_value is None
    assert lesson.factor_values == ()
    assert lesson.available_terms == 1
    visual = result.visualize((0,), renderer=RecordingRenderer(), max_total_terms=1)
    assert visual.metadata["evaluated_selections"] == ()
    assert visual.metadata["value_evaluation"]["status"] == "skipped"


def test_where_rejects_bad_broadcast_shapes_before_reading_any_input():
    arrays = [CountingArray((2,)), CountingArray((3,)), CountingArray(())]
    with pytest.raises(ValueError):
        where(*arrays, renderer=RecordingRenderer())
    assert all(array.reads == [] for array in arrays)


def test_where_rejects_untracked_and_cross_flow_operands():
    flow = Flow()
    source = flow.input(1)
    with pytest.raises(TypeError, match="tracked"):
        flow.where(source, source, 0)
    with pytest.raises(ValueError, match="same flow"):
        flow.where(source, source, Flow().input(0))
