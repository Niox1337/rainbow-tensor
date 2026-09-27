"""Walkthrough controls select contributions without losing numerical boundaries."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.explanations import t
from rainbow_tensor.walkthrough import walkthrough


@pytest.fixture
def lessons():
    """Release all lesson communication channels, including after failed assertions."""
    created = []

    def make(*args, **kwargs):
        lesson = rt.walkthrough(*args, **kwargs)
        created.append(lesson)
        return lesson

    yield make
    for lesson in created:
        lesson.close()


def test_term_buttons_change_highlights_numerator_and_mean_divisor(lessons):
    lesson = lessons(rt.mean, np.arange(1, 7).reshape(2, 3), axis=1)
    before = lesson.visual.svg
    assert lesson.snapshot.factor_values == (1,)
    assert lesson.previous_button.disabled
    lesson.next_button.click()
    assert lesson.snapshot.term == 1
    assert lesson.snapshot.factor_values == (2,)
    assert lesson.snapshot.subtotal == 3
    assert lesson.snapshot.step.divisor == 3
    assert lesson.snapshot.output_value == 2
    assert "Term 2 of 3" in lesson.explorer.figure.description
    assert "X[0, 1]" in lesson.explorer.figure.description
    assert lesson.visual.svg != before
    lesson.next_term()
    assert lesson.snapshot.subtotal == 6
    assert lesson.next_button.disabled
    lesson.previous_term()
    assert lesson.snapshot.term == 1


def test_browser_focus_resets_occurrence_and_term_controls(lessons):
    lesson = lessons(rt.sum, np.arange(1, 7).reshape(2, 3), axis=1)
    lesson.select_occurrence(1)
    assert lesson.snapshot.step.operation == "input"
    lesson.explorer.figure._handle_custom_msg({
        "type": "focus", "revision": lesson.explorer.figure.revision,
        "coordinate": json.dumps([1]),
    }, [])
    assert lesson.focus == (1,)
    assert lesson.occurrences.value == lesson.snapshot.occurrence == 0
    assert lesson.snapshot.term == 0
    assert lesson.snapshot.factor_values == (4,)


def test_matmul_and_repeated_flow_occurrences_remain_inspectable(lessons):
    product = lessons(rt.matmul, np.array([[2, 3]]), np.array([[5], [7]]))
    product.next_term()
    assert product.snapshot.factor_values == (3, 7)
    assert product.snapshot.term_value == 21
    assert product.snapshot.subtotal == 31
    flow = rt.Flow()
    source = flow.input(np.array([2, 7]))
    result = flow.sum(flow.index(source, ([1, 0, 1],)))
    lesson = lessons(result)
    ids = [
        i for i, step in enumerate(lesson.snapshot.trace.steps)
        if step.operation == "input" and step.reference.coordinate == (1,)
    ]
    assert len(ids) == 2
    lesson.occurrences.value = ids[0]
    assert lesson.snapshot.occurrence == ids[0]
    lesson.occurrences.value = ids[1]
    assert lesson.snapshot.occurrence == ids[1]
    assert lesson.snapshot.output_value == 7


def test_invalid_term_and_shape_change_preserve_the_current_lesson(lessons):
    array = np.arange(6).reshape(2, 3)
    lesson = lessons(rt.sum, array, axis=1)
    before = lesson.visual
    with pytest.raises(IndexError, match="term"):
        lesson.select_term(3)
    assert lesson.visual is before
    array.shape = (3, 2)
    lesson.next_button.click()
    assert lesson.visual is before
    assert lesson.snapshot.term == 0
    assert "changed shape" in lesson.explorer.status.value


def test_scalar_and_empty_lessons_have_valid_disabled_controls(lessons):
    scalar = lessons(rt.sum, np.array(7))
    assert scalar.focus == ()
    assert scalar.snapshot.factor_values == (7,)
    assert scalar.next_button.disabled
    empty = lessons(rt.sum, np.empty((2, 0, 3)), axis=2)
    assert empty.focus is None
    assert empty.snapshot.occurrence is None
    assert empty.occurrences.disabled
    assert empty.previous_button.disabled and empty.next_button.disabled
    assert empty.explorer.update_button.disabled
    ET.fromstring(empty.visual.svg)


def test_budget_and_trace_limits_do_not_expand_during_navigation(lessons):
    lesson = lessons(rt.sum, np.arange(20), max_nodes=4, max_terms=10)
    assert not lesson.snapshot.numeric_complete
    assert lesson.snapshot.available_terms == 3
    lesson.select_term(2)
    assert lesson.next_button.disabled
    assert lesson.snapshot.term_value is None
    assert lesson.snapshot.evaluation["max_terms"] == 10
    assert "max_nodes" in lesson.snapshot.trace.truncated_reasons
    assert "Showing 3 of 20" in lesson.visual.text


def test_refresh_reads_each_input_once_for_snapshot_and_figure(lessons):
    class Source:
        shape = (3,)

        def __init__(self):
            self.reads = []

        def __getitem__(self, coordinate):
            self.reads.append(coordinate)
            return coordinate[0] + 1

    source = Source()
    lesson = lessons(rt.sum, source)
    assert sorted(source.reads) == [(0,), (1,), (2,)]
    source.reads.clear()
    lesson.next_term()
    assert sorted(source.reads) == [(0,), (1,), (2,)]


def test_static_export_omits_widget_annotations_and_close_is_idempotent(lessons, tmp_path):
    lesson = lessons(rt.sum, np.arange(3))
    path = tmp_path / "term.svg"
    lesson.visual.save(path)
    assert "data-rt-coordinate" not in path.read_text(encoding="utf-8")
    lesson.close()
    lesson.close()
    with pytest.raises(RuntimeError, match="closed"):
        lesson.next_term()
    assert all(widget.comm is None for widget in lesson._owned_widgets)


@pytest.mark.parametrize("operation,args,options", [
    (rt.shape, ((2, 3),), {}),
    (rt.sum, (), {}),
    (rt.matmul, ((2, 3),), {}),
    (rt.sum, ((2, 3),), {"axis": 3}),
])
def test_invalid_lesson_inputs_fail_before_opening_controls(operation, args, options):
    with pytest.raises((TypeError, ValueError, IndexError)):
        walkthrough(operation, *args, **options)


def test_binary_division_keeps_operand_order_and_preserves_view_on_zero(lessons):
    denominator = np.array([2])
    flow = rt.Flow()
    result = flow.divide(flow.input(np.array([8])), flow.input(denominator))
    lesson = lessons(result)
    assert lesson.snapshot.factor_values == (8, 2)
    assert lesson.snapshot.term_value == 4
    assert lesson.snapshot.subtotal is None
    assert "8 / 2 = 4" in lesson.visual.text
    before = lesson.visual
    denominator[0] = 0
    with pytest.raises(ZeroDivisionError):
        lesson.set_focus((0,))
    assert lesson.visual is before
    lesson.occurrences.value = 1
    assert lesson.visual is before
    assert lesson.occurrences.value == 0
    assert "zero" in lesson.explorer.status.value


def test_depth_limited_lesson_does_not_claim_omitted_terms_are_absent(lessons):
    lesson = lessons(rt.sum, np.arange(3), max_depth=0)
    assert lesson.snapshot.term is None
    assert lesson.snapshot.step.term_count == 3
    assert "Showing 0 of 3" in lesson.term_label.value
    assert "no contributing terms" not in lesson.visual.text
    assert "max_depth" in lesson.visual.text


def test_walkthrough_discloses_python_arithmetic_instead_of_backend_overflow(lessons):
    a = np.array([[250]], dtype=np.uint8)
    b = np.array([[2]], dtype=np.uint8)
    lesson = lessons(rt.matmul, a, b)
    assert lesson.snapshot.output_value == 500
    assert (a @ b).item() == 244
    assert lesson.visual.metadata["numeric_semantics"] == {
        "arithmetic": "python_scalar", "scope": "recorded_operations",
        "operand_sources": ["array_values"],
    }
    assert t("numeric.backend_difference") in lesson.visual.text


def test_walkthrough_labels_generated_shape_values_even_on_a_truncated_trace(lessons):
    lesson = lessons(rt.sum, (2, 3), axis=1, max_depth=0)
    assert lesson.snapshot.output_value == 3
    assert lesson.visual.metadata["numeric_semantics"]["operand_sources"] == ["generated_values"]
    assert t("flow.generated") in lesson.visual.text
    assert t("numeric.model") in lesson.visual.text
