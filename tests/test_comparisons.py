"""Comparisons preserve boolean results, ordered origins, and bounded reads."""

import numpy as np
import pytest

from rainbow_tensor.provenance.flow import Flow
from rainbow_tensor.provenance.query import ValueBudgetExceeded
from rainbow_tensor.views import elementwise
from test_elementwise import CountingArray, RecordingRenderer

OPERATIONS = ("greater", "greater_equal", "less", "less_equal", "equal", "not_equal")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("left,right", [
    (np.array([[1], [3]]), np.array([1, 2, 3])),
    (np.array(2), 3),
    (np.empty((0, 3)), np.array([1, 2, 3])),
    (np.array([np.nan, np.inf, -np.inf]), np.array([np.nan, np.inf, 1])),
])
def test_comparison_views_match_numpy_with_ordered_broadcast_sources(operation, left, right):
    renderer = RecordingRenderer()
    visual = getattr(elementwise, operation)(left, right, renderer=renderer)
    expected = getattr(np, operation)(left, right)
    assert visual.result_shape == expected.shape
    for coordinate in np.ndindex(expected.shape):
        value = renderer.panels[-1]["value_fn"](coordinate)
        assert isinstance(value, bool)
        assert value == bool(expected[coordinate])
    if expected.size:
        assert visual.trace.expression.operator == operation
        assert visual.trace.terms == ()
    else:
        assert visual.trace is None


@pytest.mark.parametrize("operation", OPERATIONS)
def test_flow_comparisons_keep_scalar_operand_and_live_input_values(operation):
    array = np.array([[1, 3], [5, 7]])
    flow = Flow()
    source = flow.input(array)
    limit = flow.input(3)
    result = getattr(flow, operation)(source, limit)
    trace = result.trace((1, 0))
    assert trace.steps[0].binary_operator == operation
    assert trace.steps[0].terms == ()
    assert [trace.steps[index].reference for index in trace.steps[0].operands] == [
        source.ref((1, 0)), limit.ref(()),
    ]
    for coordinate in np.ndindex(array.shape):
        assert result.value(coordinate) == bool(getattr(np, operation)(array, 3)[coordinate])
    array[1, 0] = 2
    assert result.value((1, 0)) == bool(getattr(np, operation)(2, 3))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_comparison_traces_and_rejected_budgets_read_no_sources(operation):
    left = CountingArray((3,))
    right = CountingArray(())
    flow = Flow()
    result = getattr(flow, operation)(flow.input(left), flow.input(right))
    assert result.trace((1,)).complete
    assert left.reads == right.reads == []
    with pytest.raises(ValueBudgetExceeded):
        result.value((1,), max_total_terms=1)
    assert left.reads == right.reads == []


@pytest.mark.parametrize("operation", ("greater", "greater_equal", "less", "less_equal"))
def test_ordered_complex_comparisons_retain_python_scalar_errors(operation):
    flow = Flow()
    result = getattr(flow, operation)(flow.input(2 + 1j), flow.input(1 + 2j))
    assert result.trace().complete
    with pytest.raises(TypeError):
        result.value()


@pytest.mark.parametrize("operation,expected", [("equal", True), ("not_equal", False)])
def test_complex_equality_does_not_require_ordering(operation, expected):
    flow = Flow()
    value = flow.input(2 + 1j)
    assert getattr(flow, operation)(value, value).value() is expected
