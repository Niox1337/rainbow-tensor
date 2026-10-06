"""Extrema expose candidate groups and a separate first-occurrence decision."""

from math import copysign, isnan

import numpy as np
import pytest

from rainbow_tensor.ops.selection import extrema_spec, extremum_choice
from rainbow_tensor.provenance.flow import Flow
from rainbow_tensor.provenance.lesson import build_lesson
from rainbow_tensor.provenance.query import ValueBudgetExceeded, evaluate_values
from rainbow_tensor.views import selection
from test_provenance import CountingArray, RecordingRenderer

OPERATIONS = ("min", "max", "argmin", "argmax")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_extrema_views_match_numpy_values_shapes_and_source_positions(operation, axis, keepdims):
    array = np.array([[5, 1, 1], [3, 4, 2]])
    expected = getattr(np, operation)(array, axis=axis, keepdims=keepdims)
    renderer = RecordingRenderer()
    visual = getattr(selection, operation)(
        array, axis, keepdims=keepdims, renderer=renderer,
    )
    assert visual.result_shape == expected.shape
    assert visual.shape == array.shape
    for coordinate in np.ndindex(expected.shape):
        assert renderer.panels[-1]["value_fn"](coordinate) == expected[coordinate]
    assert visual.trace.expression.operator == operation
    assert visual.trace.terms == ()
    count = array.size if axis is None else array.shape[axis]
    assert visual.trace.expression.candidate_count == count
    assert visual.metadata["evaluated_selections"]
    assert all(item["operation"] == operation for item in visual.metadata["evaluated_selections"])


@pytest.mark.parametrize("operation", ["min", "max"])
@pytest.mark.parametrize("axis", [(), (0, 2), (2, 0), (-1, 0), (0, 1, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_value_extrema_support_normalized_axis_tuples(operation, axis, keepdims):
    array = np.arange(24).reshape(2, 3, 4)[::-1, :, ::-1]
    flow = Flow()
    source = flow.input(array)
    result = getattr(flow, operation)(source, axis=axis, keepdims=keepdims)
    expected = getattr(np, operation)(array, axis=axis, keepdims=keepdims)
    assert result.shape == expected.shape
    for coordinate in np.ndindex(expected.shape):
        assert result.value(coordinate) == expected[coordinate]
        values, evaluation = evaluate_values(flow, (result.ref(coordinate),))
        decision = evaluation["selected_sources"][0]
        assert array[decision.source.coordinate] == values[decision.output]
        assert decision.source.node_id == source.node_id


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("axis", [None, 0, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_scalar_extrema_have_one_candidate_and_scalar_index_zero(operation, axis, keepdims):
    flow = Flow()
    source = flow.input(7)
    result = getattr(flow, operation)(source, axis=axis, keepdims=keepdims)
    assert result.shape == ()
    assert result.value() == (0 if operation.startswith("arg") else 7)
    trace = result.trace()
    assert trace.steps[0].candidates == (1,)
    assert trace.steps[1].reference == source.ref(())


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("array,axis", [(np.empty((0, 3)), 0), (np.empty((0, 0)), 1)])
def test_empty_reduced_groups_fail_before_values_are_read(operation, array, axis):
    source_array = CountingArray(array.shape)
    flow = Flow()
    source = flow.input(source_array)
    with pytest.raises(ValueError, match="empty group"):
        getattr(flow, operation)(source, axis=axis)
    assert source_array.reads == []


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("keepdims", [False, True])
def test_empty_surviving_outputs_have_no_fake_selection(operation, keepdims):
    array = CountingArray((0, 3))
    renderer = RecordingRenderer()
    visual = getattr(selection, operation)(array, axis=1, keepdims=keepdims, renderer=renderer)
    assert visual.result_shape == ((0, 1) if keepdims else (0,))
    assert visual.trace is None
    assert visual.metadata["evaluated_selections"] == ()
    assert array.reads == []


@pytest.mark.parametrize("operation", OPERATIONS)
def test_first_equal_extremum_and_first_nan_have_exact_selected_coordinates(operation):
    array = np.array([[2, 2, 2], [5, np.nan, np.nan]])
    flow = Flow()
    source = flow.input(array)
    result = getattr(flow, operation)(source, axis=1)
    first = build_lesson(result, (0,), term=2)
    assert first.selected_source.source == source.ref((0, 0))
    assert first.selected_source.position == 0
    assert first.term_value == 2
    assert first.subtotal is None
    assert first.available_terms == 3
    second = build_lesson(result, (1,))
    assert second.selected_source.source == source.ref((1, 1))
    assert second.selected_source.position == 1
    assert second.selected_source.reason == "first_nan"
    if operation.startswith("arg"):
        assert second.output_value == 1
    else:
        assert isnan(second.output_value)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_arg_positions_and_selected_sources_remain_correct_through_transpose(operation):
    array = np.array([[9, 3, 5], [2, 6, 1]])
    flow = Flow()
    source = flow.input(array)
    turned = flow.transpose(source)
    result = getattr(flow, operation)(turned)
    lesson = build_lesson(result)
    coordinate = (1, 2) if operation in ("min", "argmin") else (0, 0)
    assert lesson.selected_source.source == turned.ref(coordinate[::-1])
    expected = getattr(np, operation)(array.T)
    assert lesson.output_value == expected
    assert any(root.reference == source.ref(coordinate) for root in lesson.trace.roots)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_extrema_budget_failure_and_structural_queries_never_read_values(operation):
    array = CountingArray((10**15,))
    flow = Flow()
    result = getattr(flow, operation)(flow.input(array))
    trace = result.trace(max_nodes=5)
    assert len(trace.steps) == 5
    assert trace.steps[0].terms == ()
    assert trace.steps[0].candidates == (1, 2, 3, 4)
    assert not trace.complete
    assert trace.steps[0].term_count == 10**15
    with pytest.raises(ValueBudgetExceeded):
        result.value(max_terms=4)
    lesson = build_lesson(result, max_terms=4)
    assert lesson.selected_source is None
    assert lesson.factor_values == ()
    assert lesson.subtotal is None
    assert array.reads == []


@pytest.mark.parametrize("operation", OPERATIONS)
def test_values_are_rechecked_and_choices_are_not_cached_across_requests(operation):
    array = np.array([1, 3, 2])
    flow = Flow()
    source = flow.input(array)
    result = getattr(flow, operation)(source)
    before = build_lesson(result).selected_source
    array[:] = array[::-1]
    after = build_lesson(result).selected_source
    assert before.position == (0 if operation in ("min", "argmin") else 1)
    assert after.position == (2 if operation in ("min", "argmin") else 1)
    assert result.trace().steps[0].selection_operator == operation


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("value", [2 + 1j, "text"])
def test_extrema_reject_nonreal_values_after_structural_trace(operation, value):
    flow = Flow()
    source = flow.input(np.array([value]))
    result = getattr(flow, operation)(source)
    assert result.trace().complete
    with pytest.raises(TypeError, match="real scalar"):
        result.value()


@pytest.mark.parametrize("operation", OPERATIONS)
def test_first_tie_retains_original_zero_sign_and_python_integer_precision(operation):
    output, position, reason = extremum_choice(operation, (0.0, -0.0))
    assert position == 0
    if not operation.startswith("arg"):
        assert copysign(1, output) == 1
    huge = 10**300
    output, position, reason = extremum_choice(operation, (huge, huge + 1))
    assert output == (0 if operation == "argmin" else 1 if operation == "argmax"
                      else huge if operation == "min" else huge + 1)


@pytest.mark.parametrize("operation", ["argmin", "argmax"])
@pytest.mark.parametrize("axis", [(), (0,), (0, 1)])
def test_index_extrema_reject_tuple_axes(operation, axis):
    with pytest.raises(TypeError, match="one integer axis"):
        extrema_spec((2, 3), axis, False, operation)


@pytest.mark.parametrize("axis", [(0, 0), (2,), True, 1.5, [0]])
def test_value_extrema_reject_invalid_axes(axis):
    with pytest.raises((TypeError, ValueError, IndexError)):
        extrema_spec((2, 3), axis, False, "max")


def test_extrema_reject_unknown_operations_and_invalid_keepdims():
    with pytest.raises(ValueError, match="unsupported extremum"):
        extrema_spec((3,), None, False, "median")
    with pytest.raises(ValueError, match="unsupported extremum"):
        extremum_choice("median", [1, 2, 3])
    with pytest.raises(TypeError, match="boolean"):
        extrema_spec((3,), None, 1, "min")
