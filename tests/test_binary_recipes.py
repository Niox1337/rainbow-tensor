"""Ordered binary recipes retain occurrence paths and plan reads before evaluation."""

import pytest

from rainbow_tensor import Flow, ValueBudgetExceeded


class ScalarSource:
    """A live scalar that makes premature or repeated reads observable."""

    shape = ()

    def __init__(self, value):
        self.value = value
        self.reads = 0

    def __getitem__(self, coordinate):
        assert coordinate == ()
        self.reads += 1
        return self.value


def binary_recipe(flow, operation, left, right):
    """Construct a scalar recipe to exercise the query engine independently of views."""
    return flow._node(operation, (left, right), (),
                      lambda coordinate: iter(((left.ref(()), right.ref(())),)),
                      binary_operator=operation)


@pytest.mark.parametrize("operation,expected", [
    ("add", 10), ("subtract", 6), ("multiply", 16), ("divide", 4),
])
def test_binary_recipe_evaluates_ordered_operands_and_distinct_occurrences(operation, expected):
    flow = Flow()
    a, b = ScalarSource(8), ScalarSource(2)
    left, right = flow.input(a), flow.input(b)
    result = binary_recipe(flow, operation, left, right)
    trace = result.trace()
    step = trace.steps[0]
    assert step.binary_operator == operation
    assert step.terms == ()
    assert step.operands == (1, 2)
    assert [trace.steps[i].reference for i in step.operands] == [left.ref(()), right.ref(())]
    assert trace.complete
    assert (a.reads, b.reads) == (0, 0)
    assert result.value() == expected
    assert (a.reads, b.reads) == (1, 1)


def test_reused_source_keeps_two_paths_with_one_value_read():
    flow = Flow()
    array = ScalarSource(3)
    source = flow.input(array)
    result = binary_recipe(flow, "divide", source, source)
    trace = result.trace()
    assert trace.steps[0].operands == (1, 2)
    assert trace.steps[1].reference == trace.steps[2].reference
    assert trace.roots[0].count == 2
    assert result.value() == 1
    assert array.reads == 1


@pytest.mark.parametrize("limits,error", [
    ({"max_terms": 0}, ValueError), ({"max_total_terms": 3}, ValueBudgetExceeded),
])
def test_binary_invalid_or_exceeded_budget_rejects_entire_plan_before_reads(limits, error):
    flow = Flow()
    array = ScalarSource(5)
    source = flow.input(array)
    result = binary_recipe(flow, "multiply", source, source)
    with pytest.raises(error):
        result.value(**limits)
    assert array.reads == 0


def test_binary_recipe_honors_per_output_limits_on_its_dependencies():
    flow = Flow()
    array = ScalarSource(5)
    array.shape = (2,)
    reduced = flow.sum(flow.input(array))
    result = binary_recipe(flow, "multiply", reduced, reduced)
    with pytest.raises(ValueBudgetExceeded, match="max_terms"):
        result.value(max_terms=1)
    assert array.reads == 0


@pytest.mark.parametrize("limits,reason", [
    ({"max_edges": 1}, "max_edges"),
    ({"max_nodes": 2}, "max_nodes"),
    ({"max_depth": 0}, "max_depth"),
])
def test_truncated_binary_expression_never_exposes_an_unpaired_operand(limits, reason):
    flow = Flow()
    array = ScalarSource(5)
    source = flow.input(array)
    result = binary_recipe(flow, "subtract", source, source)
    trace = result.trace(**limits)
    assert not trace.complete
    assert trace.steps[0].operands == ()
    assert trace.steps[0].terms == ()
    assert trace.roots == ()
    assert reason in trace.truncated_reasons
    assert array.reads == 0


def test_binary_evaluation_preserves_live_values_and_zero_division_errors():
    flow = Flow()
    a, b = ScalarSource(8), ScalarSource(2)
    result = binary_recipe(flow, "divide", flow.input(a), flow.input(b))
    assert result.value() == 4
    b.value = 4
    assert result.value() == 2
    b.value = 0
    with pytest.raises(ZeroDivisionError):
        result.value()
