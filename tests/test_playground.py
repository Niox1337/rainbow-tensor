"""Reduction lessons reveal results and replace controls safely across shape changes."""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.playground import reduction_playground


@pytest.fixture
def playgrounds():
    """Close the active explorer and every playground control after each example."""
    created = []

    def make(*args, **kwargs):
        playground = reduction_playground(*args, **kwargs)
        created.append(playground)
        return playground

    yield make
    for playground in created:
        playground.close()


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("axis,keepdims", [
    (None, False), (None, True), ((), False), ((0, 2), True), (-1, False),
])
def test_shapes_and_control_axes_match_numpy(playgrounds, operation, axis, keepdims):
    array = np.arange(24).reshape(2, 3, 4)
    playground = playgrounds(operation, array, axis=axis, keepdims=keepdims)
    numpy_operation = np.sum if operation is rt.sum else np.mean
    assert playground.explorer.result_shape == numpy_operation(
        array, axis=axis, keepdims=keepdims,
    ).shape
    assert playground.keepdims_control.value is keepdims
    assert not playground.revealed
    assert playground.result_box.children == ()
    assert "<svg" in playground.source_figure.value


def test_predict_reveal_checks_shape_and_parameter_change_starts_a_new_prediction(playgrounds):
    playground = playgrounds(rt.sum, np.arange(6).reshape(2, 3), axis=1)
    playground.prediction.value = "(2,)"
    playground.reveal_button.click()
    assert playground.revealed
    assert playground.result_box.children == (playground.explorer.widget,)
    assert "matches" in playground.status.value
    assert playground.prediction.disabled
    playground.set_parameters(axis=0)
    assert not playground.revealed
    assert playground.result_box.children == ()
    assert playground.prediction.value == ""
    playground.prediction.value = "(99,)"
    playground.reveal()
    assert "Compare" in playground.status.value
    playground.hide()
    assert not playground.revealed


def test_axis_changes_keep_valid_focus_and_reset_invalid_focus(playgrounds):
    playground = playgrounds(rt.sum, np.arange(6).reshape(2, 3), axis=1, predict=False)
    old = playground.explorer
    old.set_focus((1,))
    playground.set_parameters(axis=0)
    assert playground.explorer.focus == (1,)
    assert old.widget.comm is None
    assert all(widget.comm is None for widget in old._owned_widgets)
    assert playground.revealed
    playground.set_parameters(keepdims=True)
    assert playground.explorer.focus == (0, 0)
    assert playground.explorer.result_shape == (1, 3)
    assert "axis=0" in playground.expression
    assert "keepdims=True" in playground.expression


def test_apply_controls_cover_tuple_all_and_no_axes(playgrounds):
    playground = playgrounds(rt.mean, np.arange(24).reshape(2, 3, 4))
    playground.axis_mode.value = "selected"
    playground.axes.value = (0, 2)
    playground.keepdims_control.value = True
    playground.apply_button.click()
    assert playground.axis == (0, 2)
    assert playground.explorer.result_shape == (1, 3, 1)
    assert playground.explorer.visual.trace.divisor == 8
    playground.axis_mode.value = "none"
    playground.apply_button.click()
    assert playground.axis == ()
    assert playground.explorer.result_shape == (2, 3, 4)
    assert playground.explorer.visual.trace.divisor == 1
    playground.axis_mode.value = "all"
    playground.keepdims_control.value = False
    playground.apply_button.click()
    assert playground.axis is None
    assert playground.explorer.focus == ()


def test_empty_output_becomes_real_empty_reduction_groups(playgrounds):
    playground = playgrounds(rt.sum, np.empty((2, 0, 3)), axis=2)
    assert playground.explorer.focus is None
    assert playground.explorer.update_button.disabled
    playground.reveal()
    assert playground.revealed
    playground.set_parameters(axis=1)
    assert playground.explorer.result_shape == (2, 3)
    assert playground.explorer.focus == (0, 0)
    assert playground.visual.trace.term_count == 0
    assert not playground.explorer.update_button.disabled


def test_scalar_axes_and_empty_axis_tuple_are_supported(playgrounds):
    playground = playgrounds(rt.sum, np.array(7), axis=0)
    assert playground.explorer.focus == ()
    assert playground.axes.options == ()
    playground.set_parameters(axis=(), keepdims=True)
    assert playground.explorer.result_shape == ()
    assert playground.visual.trace.divisor == 1


@pytest.mark.parametrize("options", [
    {"axis": 3}, {"axis": (0, 0)}, {"axis": [0]}, {"keepdims": 1},
])
def test_invalid_parameter_changes_preserve_explorer_and_recipe(playgrounds, options):
    playground = playgrounds(rt.sum, np.arange(6).reshape(2, 3), axis=1)
    old = playground.explorer
    before = playground.visual
    with pytest.raises((TypeError, ValueError, IndexError)):
        playground.set_parameters(**options)
    assert playground.explorer is old
    assert playground.visual is before
    assert playground.axis == 1
    assert old.widget.comm is not None


def test_failed_replacement_keeps_the_previous_widget_live(playgrounds):
    class Source:
        shape = (2, 3)
        broken = False

        def __getitem__(self, coordinate):
            if self.broken:
                raise ValueError("source is unavailable")
            return coordinate[0] * 3 + coordinate[1]

    source = Source()
    playground = playgrounds(rt.sum, source, axis=1)
    old = playground.explorer
    source.broken = True
    with pytest.raises(ValueError, match="source is unavailable"):
        playground.set_parameters(axis=0)
    assert playground.explorer is old
    assert old.widget.comm is not None


def test_budget_options_survive_parameter_replacement(playgrounds):
    playground = playgrounds(rt.sum, np.arange(6).reshape(2, 3), axis=1, max_terms=1)
    assert playground.visual.metadata["value_evaluation"]["status"] == "skipped"
    playground.set_parameters(axis=0)
    assert playground.visual.metadata["value_evaluation"]["max_terms"] == 1
    assert playground.visual.metadata["value_evaluation"]["status"] == "skipped"


def test_shape_change_error_and_close_preserve_lifecycle_contract(playgrounds):
    array = np.arange(6).reshape(2, 3)
    playground = playgrounds(rt.sum, array, axis=1)
    before = playground.visual
    array.shape = (3, 2)
    playground.apply_button.click()
    assert playground.visual is before
    assert "source shape changed" in playground.status.value
    playground.close()
    playground.close()
    assert all(widget.comm is None for widget in playground._owned_widgets)
    with pytest.raises(RuntimeError, match="closed"):
        playground.set_parameters(axis=0)
    with pytest.raises(RuntimeError, match="closed"):
        playground.reveal()


@pytest.mark.parametrize("prediction", ["(False,)", "(2.0,)", "2", "not a tuple", "(999,)"])
def test_invalid_predictions_are_feedback_not_python_execution(playgrounds, prediction):
    playground = playgrounds(rt.sum, np.arange(6).reshape(2, 3), axis=1)
    playground.prediction.value = prediction
    playground.reveal()
    assert "Compare" in playground.status.value
