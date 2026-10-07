"""Prediction feedback distinguishes axis rules from malformed learner input."""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor import explanations
from rainbow_tensor.interactive.prediction import diagnose_prediction, prediction_messages


@pytest.mark.parametrize("answer", ["(False,)", "(2.0,)", "(-1,)", "2", "[2]", "unknown()"])
def test_malformed_predictions_do_not_become_dimension_comparisons(answer):
    feedback = diagnose_prediction(answer, (2, 3), (1,), False)
    assert feedback.code == "malformed"
    assert feedback.predicted is None
    assert "shape tuple of nonnegative integers" in " ".join(prediction_messages(feedback))


@pytest.mark.parametrize("answer, keepdims, code", [
    ("(3,)", False, "matches"),
    ("(4,)", False, "dimension"),
    ("(1, 3)", False, "rank"),
    ("(3,)", True, "keepdims"),
    ("(1, 3, 1)", True, "matches"),
    ("", False, "missing"),
])
def test_tuple_axis_predictions_identify_the_type_of_mistake(answer, keepdims, code):
    feedback = diagnose_prediction(answer, (2, 3, 4), (0, 2), keepdims)
    assert feedback.code == code
    assert feedback.expected == ((1, 3, 1) if keepdims else (3,))
    messages = " ".join(prediction_messages(feedback))
    assert "(0, 2)" in messages
    assert f"Source axis 1 becomes output axis {1 if keepdims else 0} with length 3" in messages


def test_scalar_shape_explanation_does_not_describe_a_one_element_axis():
    feedback = diagnose_prediction("(1,)", (2, 3), (0, 1), False)
    assert feedback.code == "rank"
    assert feedback.expected == ()
    assert "scalar with one value" in " ".join(prediction_messages(feedback))


def test_empty_outputs_and_empty_reduction_groups_have_different_feedback():
    output = diagnose_prediction("(2, 3)", (2, 0, 3), (2,), False)
    groups = diagnose_prediction("(2, 3)", (2, 0, 3), (1,), False)
    assert output.code == "dimension"
    assert output.expected == (2, 0)
    assert "no output elements to select" in " ".join(prediction_messages(output))
    assert groups.code == "matches"
    assert "6 elements, but every reduction group is empty" in " ".join(prediction_messages(groups))


def test_no_axes_and_scalar_inputs_keep_their_shapes():
    unchanged = diagnose_prediction("(2, 3)", (2, 3), (), True)
    scalar = diagnose_prediction("()", (), (), True)
    assert unchanged.code == scalar.code == "matches"
    assert "No source axes are reduced" in " ".join(prediction_messages(unchanged))
    assert scalar.expected == ()


def test_reveal_exposes_diagnosis_and_parameter_changes_clear_it():
    playground = rt.reduction_playground(rt.sum, np.arange(24).reshape(2, 3, 4), axis=(0, 2))
    try:
        assert playground.feedback is None
        playground.prediction.value = "(4,)"
        playground.reveal()
        assert playground.feedback.code == "dimension"
        assert "Compare output axis 0: predicted 4, expected 3" in playground.status.value
        playground.set_parameters(keepdims=True)
        assert playground.feedback is None
        assert not playground.revealed
        playground.prediction.value = "(3,)"
        playground.reveal()
        assert playground.feedback.code == "keepdims"
        playground.hide()
        assert playground.feedback is None
    finally:
        playground.close()


def test_failed_feedback_translation_preserves_hidden_result_and_prior_feedback(monkeypatch):
    playground = rt.reduction_playground(rt.sum, np.arange(6).reshape(2, 3), axis=1)
    try:
        playground.prediction.value = "(3,)"
        before = playground.status.value
        messages = dict(explanations.MESSAGES)
        messages["en"] = dict(messages["en"], **{
            "playground.feedback.dimension": "{predicted:invalid}",
        })
        monkeypatch.setattr(explanations, "MESSAGES", messages)
        with pytest.raises(ValueError):
            playground.reveal()
        assert not playground.revealed
        assert playground.feedback is None
        assert playground.result_box.children == ()
        assert playground.status.value == before
    finally:
        playground.close()
