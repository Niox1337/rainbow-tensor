"""Capture freezes explicit states and preserves numerical and structural limits."""

import json

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.lessons import LessonRecording, capture_lesson
from rainbow_tensor.visual import TensorVisual


@pytest.fixture
def controllers():
    created = []
    yield created
    for controller in created:
        controller.close()


def test_generator_copies_each_focus_and_retains_no_live_array_dependency(controllers):
    array = np.arange(6).reshape(2, 3)
    explorer = rt.explore(rt.sum, array, axis=1)
    controllers.append(explorer)

    def states():
        yield explorer
        explorer.set_focus((1,))
        yield explorer

    recording = capture_lesson(states())
    before = recording.to_json()
    assert [state["focus"] for state in recording.to_dict()["states"]] == [["0"], ["1"]]
    array[:] = 90
    explorer.set_focus((0,))
    assert recording.to_json() == before
    assert recording.state_count == 2


def test_capture_does_not_reread_values_or_change_controller_focus(controllers):
    class Counted:
        shape = (2, 3)

        def __init__(self):
            self.reads = 0

        def __getitem__(self, coordinate):
            self.reads += 1
            return coordinate[0] * 3 + coordinate[1]

    source = Counted()
    lesson = rt.walkthrough(rt.sum, source, axis=1, focus=(1,))
    controllers.append(lesson)
    lesson.select_term(1)
    before = (source.reads, lesson.focus, lesson.snapshot)
    recording = capture_lesson([lesson])
    assert (source.reads, lesson.focus, lesson.snapshot) == before
    state = recording.to_dict()["states"][0]
    assert state["lesson"]["term"] == 1
    assert state["lesson"]["factor_values"] == [4]
    assert state["lesson"]["subtotal"] == 7
    assert state["lesson"]["output_value"] == 12
    assert state["completeness"] == dict.fromkeys(("trace", "ancestry", "values"), "complete")


def test_capture_preserves_separate_trace_and_value_limits(controllers):
    flow = rt.Flow()
    source = flow.input(np.arange(12))
    lesson = rt.walkthrough(flow.sum(source), max_nodes=2, max_total_terms=1)
    controllers.append(lesson)
    state = capture_lesson([lesson]).to_dict()["states"][0]
    assert state["completeness"]["ancestry"] == "truncated"
    assert state["completeness"]["values"] == "truncated"
    assert state["metadata"]["provenance"]["truncated_reasons"] == ["max_nodes"]
    assert state["lesson"]["output_value"] is None


def test_prediction_capture_keeps_answer_and_visibility_without_revealing(controllers):
    playground = rt.reduction_playground(rt.sum, np.arange(6).reshape(2, 3), axis=1)
    controllers.append(playground)
    playground.prediction.value = "(3,)"
    hidden = capture_lesson([playground]).to_dict()["states"][0]
    assert not playground.revealed
    assert hidden["prediction"]["answer"] == "(3,)"
    assert not hidden["prediction"]["revealed"]
    assert hidden["prediction"]["expression"] == "np.sum(x, axis=1, keepdims=False)"
    assert hidden["prediction"]["source_svg"] == playground.input_visual.svg
    playground.reveal()
    revealed = capture_lesson([playground]).to_dict()["states"][0]
    assert revealed["prediction"]["revealed"]
    assert revealed["metadata"]["prediction_feedback"]["predicted"] == [3]
    assert "<" not in revealed["prediction"]["prompt"]
    assert not hidden["prediction"]["revealed"]


def test_unknown_metadata_never_evaluates_backend_arrays_or_custom_objects():
    class Unknown:
        def __iter__(self):
            raise AssertionError("unknown objects must not be traversed")

        def __repr__(self):
            raise AssertionError("unknown objects must not be formatted")

    visual = TensorVisual("<svg/>", (), metadata={"custom": Unknown(), "array": np.ones(4)})
    state = capture_lesson([visual]).to_dict()["states"][0]
    assert state["metadata"]["custom"] == {"status": "unknown", "type": "Unknown"}
    assert state["metadata"]["array"] == {"status": "unknown", "type": "ndarray"}
    assert state["completeness"] == dict.fromkeys(("trace", "ancestry", "values"), "unknown")


def test_capture_preserves_selection_metadata_and_nonfinite_values_without_invalid_json():
    metadata = {
        "evaluated_selections": [{"operation": "max", "position": 1, "reason": "first_tie"}],
        "large": 2**63 + 1, "nan": float("nan"),
    }
    visual = TensorVisual("<svg/>", (), metadata=metadata)
    recording = capture_lesson([visual])
    state = json.loads(recording.to_json())["states"][0]
    assert state["metadata"]["evaluated_selections"] == metadata["evaluated_selections"]
    assert state["metadata"]["large"] == "9223372036854775809"
    assert state["metadata"]["nan"] == {"kind": "nonfinite", "value": "nan"}
    assert LessonRecording.from_json(recording.to_json()).to_json() == recording.to_json()


def test_capture_rejects_an_infinite_state_sequence_after_one_extra_item():
    visual = TensorVisual("<svg/>", ())
    consumed = []

    def states():
        while True:
            consumed.append(1)
            yield visual

    with pytest.raises(ValueError, match="max_states=2"):
        capture_lesson(states(), max_states=2)
    assert len(consumed) == 3


def test_capture_enforces_exact_document_byte_limit():
    visual = TensorVisual("<svg/>", (), explanation=["λ 😀"])
    before = capture_lesson([visual]).to_json()
    size = len(before.encode("utf-8"))
    assert capture_lesson([visual], max_bytes=size).to_json() == before
    with pytest.raises(ValueError, match="max_bytes"):
        capture_lesson([visual], max_bytes=size - 1)


@pytest.mark.parametrize("states,error", [([], ValueError), ([None], TypeError),
                                            (["visual"], TypeError)])
def test_capture_requires_at_least_one_supported_state(states, error):
    with pytest.raises(error):
        capture_lesson(states)


def test_capture_rejects_non_svg_renderers_and_non_string_title():
    visual = TensorVisual("text", (), mime_type="text/plain")
    with pytest.raises(ValueError, match="SVG"):
        capture_lesson([visual])
    with pytest.raises(TypeError, match="title"):
        capture_lesson([], title=1)
