"""Contribution lessons preserve paths and share a bounded numerical plan."""

import math

import numpy as np
import pytest

from rainbow_tensor import Flow
from rainbow_tensor.provenance.lesson import build_lesson


def test_matmul_walkthrough_keeps_ordered_products_and_running_numerator():
    flow = Flow()
    a = flow.input(np.array([[2, 3, 4]]))
    b = flow.input(np.array([[5], [6], [7]]))
    lesson = build_lesson(flow.matmul(a, b), (0, 0), term=1)
    assert lesson.factor_values == (3, 6)
    assert lesson.term_value == 18
    assert lesson.subtotal == 28
    assert lesson.output_value == 56
    assert lesson.available_terms == 3
    assert lesson.trace.complete
    assert lesson.numeric_complete


def test_nested_means_keep_the_local_divisor_and_numerator():
    flow = Flow()
    x = flow.input(np.arange(1, 7).reshape(2, 3))
    result = flow.mean(flow.mean(x, axis=1))
    lesson = build_lesson(result, (), term=1)
    assert lesson.factor_values == (5,)
    assert lesson.subtotal == 7
    assert lesson.step.divisor == 2
    assert lesson.output_value == 3.5
    inner = build_lesson(result, (), occurrence=lesson.factor_occurrences[0], term=2)
    assert inner.step.divisor == 3
    assert inner.subtotal == 15
    assert inner.output_value == 5


def test_repeated_sampling_keeps_distinct_occurrence_numbers():
    flow = Flow()
    x = flow.input(np.array([2, 7]))
    result = flow.sum(flow.index(x, ([1, 0, 1],)))
    lesson = build_lesson(result)
    positions = [
        i for i, step in enumerate(lesson.trace.steps)
        if step.operation == "input" and step.reference.coordinate == (1,)
    ]
    assert len(positions) == 2
    first = build_lesson(result, occurrence=positions[0])
    last = build_lesson(result, occurrence=positions[1])
    assert first.occurrence != last.occurrence
    assert first.step.reference == last.step.reference
    assert first.output_value == last.output_value == 7
    assert first.term is None


def test_truncated_terms_report_a_prefix_without_losing_exact_output_value():
    flow = Flow()
    result = flow.sum(flow.input(np.arange(1, 21)))
    lesson = build_lesson(result, max_nodes=4, term=2)
    assert lesson.available_terms == 3
    assert lesson.step.term_count == 20
    assert not lesson.trace.complete
    assert "max_nodes" in lesson.trace.truncated_reasons
    assert lesson.subtotal == 6
    assert lesson.output_value == 210
    with pytest.raises(IndexError, match="term"):
        build_lesson(result, max_nodes=4, term=3)


class CountingArray:
    """Track backend reads so budget checks cannot hide partial evaluation."""

    shape = (3,)

    def __init__(self):
        self.reads = []

    def __getitem__(self, coordinate):
        self.reads.append(coordinate)
        return coordinate[0] + 1


def test_shared_inputs_are_read_once_across_preview_factors_and_running_total():
    source = CountingArray()
    flow = Flow()
    x = flow.input(source)
    result = flow.sum(flow.index(x, ([2, 0, 2],)))
    lesson = build_lesson(result, term=2, preview_references=(x.ref((1,)),))
    assert sorted(source.reads) == [(0,), (1,), (2,)]
    assert lesson.subtotal == lesson.output_value == 7


@pytest.mark.parametrize("limits", [{"max_terms": 2}, {"max_total_terms": 5}])
def test_budget_failure_reads_nothing_and_hides_all_numerical_fields(limits):
    source = CountingArray()
    flow = Flow()
    lesson = build_lesson(flow.sum(flow.input(source)), term=1, **limits)
    assert source.reads == []
    assert lesson.values == {}
    assert lesson.factor_values == ()
    assert lesson.term_value is lesson.subtotal is lesson.output_value is None
    assert lesson.available_terms == 3
    assert not lesson.numeric_complete


def test_empty_output_scalar_input_and_empty_groups_remain_distinct():
    flow = Flow()
    empty = flow.input(np.empty((2, 0, 3)))
    empty_output = build_lesson(flow.sum(empty, axis=2))
    assert empty_output.occurrence is None
    assert empty_output.term is None
    assert empty_output.trace.root is None
    assert empty_output.available_terms == 0
    zero = build_lesson(flow.sum(empty, axis=1), (0, 0))
    undefined = build_lesson(flow.mean(empty, axis=1), (0, 0))
    assert zero.term is None
    assert zero.subtotal == zero.output_value == 0
    assert undefined.step.divisor == 0
    assert math.isnan(undefined.output_value)
    scalar = build_lesson(flow.sum(flow.input(np.array(7))))
    assert scalar.trace.root.coordinate == ()
    assert scalar.factor_values == (7,)
    assert scalar.output_value == 7


def test_refresh_reads_changed_values_and_rejects_changed_input_shape():
    array = np.arange(3)
    flow = Flow()
    result = flow.sum(flow.input(array))
    assert build_lesson(result).output_value == 3
    array[0] = 10
    assert build_lesson(result).output_value == 13
    array.shape = (1, 3)
    with pytest.raises(ValueError, match="changed shape"):
        build_lesson(result)


@pytest.mark.parametrize("selection", [
    {"occurrence": True}, {"occurrence": -1}, {"occurrence": 20},
    {"term": True}, {"term": -1}, {"term": 3},
])
def test_invalid_selections_do_not_read_values(selection):
    source = CountingArray()
    flow = Flow()
    result = flow.sum(flow.input(source))
    with pytest.raises((TypeError, IndexError)):
        build_lesson(result, **selection)
    assert source.reads == []


def test_snapshot_values_are_read_only():
    flow = Flow()
    lesson = build_lesson(flow.sum(flow.input(np.arange(3))))
    with pytest.raises(TypeError):
        lesson.values[lesson.trace.root] = 99


def test_binary_occurrences_preserve_left_right_roles_without_product_subtotal():
    flow = Flow()
    left = flow.input(np.array([9]))
    right = flow.input(np.array([2]))
    result = flow._node(
        "subtract", (left, right), (1,),
        lambda coordinate: iter(((left.ref(coordinate), right.ref(coordinate)),)),
        binary_operator="subtract",
    )
    lesson = build_lesson(result)
    assert lesson.binary_operator == "subtract"
    assert lesson.factor_values == (9, 2)
    assert lesson.term_value == lesson.output_value == 7
    assert lesson.subtotal is None
    assert lesson.available_terms == 1
