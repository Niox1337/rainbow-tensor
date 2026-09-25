"""Recorded arithmetic explains complete expressions while keeping source roles."""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.provenance.lesson import build_lesson


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "divide"])
@pytest.mark.parametrize("shapes", [((2, 1), (1, 3)), ((), (2, 3)), ((2, 0, 3), (1, 3))])
def test_recorded_binary_shapes_values_and_roles_match_numpy(operation, shapes):
    arrays = [np.arange(np.prod(shape), dtype=float).reshape(shape) + 1 for shape in shapes]
    expected = getattr(np, operation)(*arrays)
    flow = rt.Flow()
    inputs = [flow.input(array) for array in arrays]
    result = getattr(flow, operation)(*inputs)
    assert result.shape == expected.shape
    for coordinate in np.ndindex(expected.shape):
        assert result.value(coordinate) == pytest.approx(expected[coordinate])
        trace = result.trace(coordinate)
        assert trace.complete
        assert trace.steps[0].terms == ()
        assert trace.steps[0].binary_operator == operation
        assert tuple(trace.steps[index].reference.node_id for index in trace.steps[0].operands) == (
            inputs[0].node_id, inputs[1].node_id,
        )


def test_row_normalization_retains_numerator_and_denominator_paths():
    scores = np.array([[2., 4., 6.], [3., 6., 9.]])
    flow = rt.Flow()
    source = flow.input(scores, name="X")
    totals = flow.sum(source, axis=1, keepdims=True, name="total")
    normalized = flow.divide(source, totals, name="Y")
    expected = scores / scores.sum(axis=1, keepdims=True)
    assert normalized.shape == (2, 3)
    assert normalized.value((1, 2)) == expected[1, 2] == 0.5
    trace = normalized.trace((1, 2))
    assert trace.complete
    assert trace.steps[0].operands == (1, 2)
    assert trace.steps[1].reference == source.ref((1, 2))
    assert trace.steps[2].reference == totals.ref((1, 0))
    counts = {item.reference.coordinate: item.count for item in trace.roots}
    assert counts == {(1, 2): 2, (1, 0): 1, (1, 1): 1}
    visual = normalized.visualize((1, 2))
    assert "X[1, 2] / total[1, 0]" in visual.text
    assert "X[1, 0] + X[1, 1] + X[1, 2]" in visual.text
    assert visual.trace.expression.operator == "divide"
    assert [ref.operand for ref in visual.trace.expression.operands] == [0, 1]
    lesson = build_lesson(normalized, (1, 2))
    assert lesson.factor_values == (9, 18)
    assert lesson.term_value == 0.5
    assert lesson.subtotal is None


def test_literal_inputs_are_scalar_values_and_can_be_reused():
    flow = rt.Flow()
    source = flow.input(np.array([6, 12]))
    constant = flow.input(3, name="divisor")
    result = flow.divide(source, constant)
    assert constant.shape == ()
    assert constant.value() == 3
    assert [result.value((i,)) for i in range(2)] == [2, 4]
    assert flow.subtract(constant, source).value((0,)) == -3


def test_binary_methods_reject_cross_flow_and_implicit_arrays_without_registering_nodes():
    flow = rt.Flow()
    source = flow.input((2,))
    foreign = rt.Flow().input((2,))
    before = dict(flow._nodes)
    for operation in (flow.add, flow.subtract, flow.multiply, flow.divide):
        with pytest.raises(ValueError, match="same flow"):
            operation(source, foreign)
        with pytest.raises(TypeError, match="tracked tensors"):
            operation(source, np.ones(2))
    assert flow._nodes == before


def test_failed_broadcast_and_duplicate_names_leave_the_flow_unchanged():
    flow = rt.Flow()
    a, b = flow.input((2,), name="A"), flow.input((3,), name="B")
    before = dict(flow._nodes)
    with pytest.raises(ValueError):
        flow.add(a, b)
    with pytest.raises(ValueError, match="already used"):
        flow.multiply(a, a, name="A")
    assert flow._nodes == before


def test_binary_empty_outputs_and_truncated_equations_stay_explicit():
    flow = rt.Flow()
    empty = flow.add(flow.input((0, 3)), flow.input(2))
    assert empty.trace().root is None
    assert empty.visualize().result_shape == (0, 3)
    scalar = flow.divide(flow.input(9), flow.input(3))
    visual = scalar.visualize(max_edges=1)
    assert not visual.provenance.complete
    assert not visual.trace.complete
    assert visual.trace.expression is None
    assert "..." in visual.text


def test_binary_explorer_updates_both_ordered_origins():
    flow = rt.Flow()
    result = flow.subtract(flow.input(np.array([[8], [9]])), flow.input(np.array([1, 2, 3])))
    explorer = rt.explore(result)
    try:
        explorer.set_focus((1, 2))
        assert explorer.visual.trace.expression.operands == (
            rt.OperandRef(0, (1, 0)), rt.OperandRef(1, (2,)),
        )
        assert result.value((1, 2)) == 6
    finally:
        explorer.close()


def test_zero_denominator_fails_only_when_values_are_requested():
    flow = rt.Flow()
    result = flow.divide(flow.input(0), flow.input(0))
    assert result.trace().complete
    with pytest.raises(ZeroDivisionError):
        result.value()
    with pytest.raises(ZeroDivisionError):
        result.visualize()
