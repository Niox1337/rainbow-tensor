"""Role-labelled paths preserve ordered and repeated uses of each source."""

import json

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor import explanations
from rainbow_tensor.interactive.paths import calculation_paths


@pytest.fixture
def lessons():
    """Close all path controls, even when an interaction assertion fails."""
    created = []

    def make(*args, **options):
        lesson = rt.walkthrough(*args, **options)
        created.append(lesson)
        return lesson

    yield make
    for lesson in created:
        lesson.close()


def test_normalization_paths_name_numerator_denominator_and_sum_sources(lessons):
    flow = rt.Flow()
    source = flow.input(np.array([[3., 6., 9.]]), name="Scores")
    totals = flow.sum(source, axis=1, keepdims=True, name="Totals")
    output = flow.divide(source, totals, name="Normalized")
    lesson = lessons(output, focus=(0, 2))
    denominator = next(path for path in lesson.paths if path.role == "denominator")
    numerator = next(path for path in lesson.paths if path.role == "numerator")
    assert denominator.parent == numerator.parent == 0
    lesson.select_path(denominator.occurrence)
    assert lesson.snapshot.step.operation == "sum"
    assert lesson.snapshot.output_value == 18
    assert "Normalized[0, 2] ← denominator ← Totals[0, 0]" in lesson.visual.text
    summand = next(
        path for path in lesson.paths if path.parent == denominator.occurrence and path.term == 2
    )
    lesson.calculation_paths.value = summand.occurrence
    assert lesson.snapshot.output_value == 9
    assert "denominator ← Totals[0, 0] ← summand 3 ← Scores[0, 2]" in lesson.visual.text
    assert lesson.focus == (0, 2)


def test_repeated_equal_sources_keep_different_paths_and_selection_ids(lessons):
    flow = rt.Flow()
    source = flow.input(np.array([2, 7]), name="Source")
    output = flow.sum(flow.index(source, ([1, 0, 1],), name="Picked"), name="Total")
    lesson = lessons(output)
    paths = [path for path in lesson.paths if (
        lesson.snapshot.trace.steps[path.occurrence].reference == source.ref((1,))
    )]
    assert len(paths) == 2
    assert paths[0].occurrence != paths[1].occurrence
    assert paths[0].parent != paths[1].parent
    lesson.select_path(paths[0].occurrence)
    first = lesson.path_description.value
    lesson.select_path(paths[1].occurrence)
    assert first != lesson.path_description.value
    assert "summand 1" in first
    assert "summand 3" in lesson.path_description.value
    assert lesson.snapshot.output_value == 7


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "greater"])
def test_binary_and_comparison_paths_keep_left_and_right_order(operation, lessons):
    flow = rt.Flow()
    output = getattr(flow, operation)(flow.input(8), flow.input(2))
    lesson = lessons(output)
    assert [path.role for path in lesson.paths] == ["result", "left", "right"]
    lesson.select_path(lesson.paths[2].occurrence)
    assert lesson.snapshot.output_value == 2


def test_matmul_paths_keep_term_and_factor_positions(lessons):
    lesson = lessons(rt.matmul, np.array([[2, 3]]), np.array([[5], [7]]))
    factors = [path for path in lesson.paths if path.role == "factor"]
    assert [(path.term, path.factor) for path in factors] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    lesson.select_path(factors[-1].occurrence)
    assert "summand 2, factor 2" in lesson.visual.text
    assert lesson.snapshot.output_value == 7


def test_broadcast_and_mapping_paths_explain_expansion_without_new_values(lessons):
    flow = rt.Flow()
    source = flow.input(np.array([5]), name="Source")
    expanded, _ = flow.broadcast(source, flow.input(np.zeros(3)))
    lesson = lessons(expanded, focus=(2,))
    assert lesson.paths[1].role == "broadcast"
    lesson.select_path(1)
    assert "broadcast source" in lesson.visual.text
    assert lesson.snapshot.output_value == 5


def test_structural_path_building_never_reads_values():
    class Source:
        shape = (3,)

        def __getitem__(self, coordinate):
            raise AssertionError("structural paths must not read values")

    flow = rt.Flow()
    output = flow.sum(flow.input(Source()))
    paths = calculation_paths(output.trace(()))
    assert len(paths) == 4
    assert [path.role for path in paths[1:]] == ["summand", "summand", "summand"]


def test_where_paths_distinguish_candidates_from_the_evaluated_choice(lessons):
    flow = rt.Flow()
    result = flow.where(flow.input(False, name="Condition"), flow.input(3, name="A"),
                        flow.input(9, name="B"), name="Chosen")
    lesson = lessons(result)
    assert [path.role for path in lesson.paths] == [
        "result", "condition", "when_true", "when_false",
    ]
    assert lesson.snapshot.subtotal is None
    assert "Chosen[()] selects B[()]" in lesson.visual.text
    assert "candidate dependencies" in lesson.visual.text
    assert "False * 3 * 9" not in lesson.visual.text
    json.dumps(lesson.visual.metadata["value_evaluation"])
    json.dumps(lesson.visual.metadata["evaluated_selections"])
    assert lesson.visual.metadata["evaluated_selections"][0]["position"] == 2


def test_empty_and_truncated_paths_never_offer_missing_occurrences(lessons):
    empty = lessons(rt.sum, np.empty((0, 2)), axis=1)
    assert empty.paths == ()
    assert empty.calculation_paths.disabled
    bounded = lessons(rt.sum, np.arange(20), max_nodes=4, max_terms=3)
    assert len(bounded.paths) == 4
    assert "Omitted paths are not selectable" in bounded.path_description.value
    before = bounded.visual
    with pytest.raises(IndexError):
        bounded.select_path(4)
    assert bounded.visual is before
    assert bounded.snapshot.evaluation["max_terms"] == 3


def test_path_control_failure_restores_the_last_successful_selection(lessons, monkeypatch):
    lesson = lessons(rt.sum, np.arange(3))
    before = lesson.visual
    old_html = lesson.path_description.value
    messages = dict(explanations.MESSAGES)
    messages["en"] = dict(messages["en"], **{"lesson.path": "{path:invalid}"})
    monkeypatch.setattr(explanations, "MESSAGES", messages)
    lesson.calculation_paths.value = 1
    assert lesson.visual is before
    assert lesson.calculation_paths.value == lesson.snapshot.occurrence == 0
    assert lesson.path_description.value == old_html
    assert "format" in lesson.explorer.status.value


def test_closing_path_controls_releases_widgets_and_rejects_python_navigation(lessons):
    lesson = lessons(rt.sum, np.arange(3))
    lesson.close()
    assert all(widget.comm is None for widget in lesson._owned_widgets)
    with pytest.raises(RuntimeError, match="closed"):
        lesson.select_path(1)
