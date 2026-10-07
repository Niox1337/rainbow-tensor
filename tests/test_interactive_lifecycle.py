"""Failed notebook initialization and refresh must release models and preserve snapshots."""

import importlib
import json

import ipywidgets as widgets
import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor import explanations
from rainbow_tensor.interactive import FocusExplorer
from rainbow_tensor.playground import ReductionPlayground
from rainbow_tensor.walkthrough import Walkthrough


@pytest.fixture
def widget_models():
    """Observe new communication models and close leftovers after a failing assertion."""
    registry = importlib.import_module("ipywidgets.widgets.widget")._instances
    before = set(registry)
    yield lambda: {key for key in registry if key not in before}
    for key in tuple(set(registry) - before):
        registry[key].close()


@pytest.fixture
def bad_catalog(tmp_path, monkeypatch):
    """Load a valid catalog whose format specifier is incompatible with a runtime value."""
    monkeypatch.setattr(explanations, "MESSAGES", dict(explanations.MESSAGES))

    def install(key, template):
        path = tmp_path / "zz.json"
        path.write_text(json.dumps({key: template}), encoding="utf-8")
        assert rt.load_translations(path) == ("zz",)
        rt.set_language("zz")

    return install


def create_controller(kind, *, predict=False):
    """Exercise each public constructor with a nonempty vector result."""
    array = np.arange(6).reshape(2, 3)
    if kind == "explorer":
        return rt.explore(rt.sum, array, axis=1)
    if kind == "walkthrough":
        return rt.walkthrough(rt.sum, array, axis=1)
    return rt.reduction_playground(rt.sum, array, axis=1, predict=predict)


def explorer_state(explorer):
    """Capture the public snapshot and synchronized figure traits together."""
    return (
        explorer.visual, explorer._panels, explorer._clickable, explorer.focus,
        tuple((control.value, control.description) for control in explorer.coordinates),
        explorer.figure.value, explorer.figure.description, explorer.figure.label,
        explorer.figure.revision, explorer.explanation.value, explorer.status.value,
        explorer.update_button.description,
    )


def lesson_state(lesson):
    """Include occurrence selection and contribution controls in a lesson snapshot."""
    return (
        explorer_state(lesson.explorer), lesson.snapshot, lesson._selection,
        lesson.occurrences.options, lesson.occurrences.value, lesson.occurrences.description,
        lesson.previous_button.description, lesson.previous_button.disabled,
        lesson.next_button.description, lesson.next_button.disabled,
        lesson.calculation_paths.options, lesson.calculation_paths.value,
        lesson.calculation_paths.description, lesson.calculation_paths.disabled,
        lesson.path_description.value,
        lesson.term_label.value,
    )


def playground_state(playground):
    """Include the replacement explorer and every parameter-dependent display value."""
    return (
        playground.explorer, playground.input_visual, playground.axis, playground.keepdims,
        playground.axis_mode.value, playground.axes.value, playground.axes.disabled,
        playground.keepdims_control.value, playground.source_figure.value,
        playground.expression_view.value, playground.prediction.value,
        playground.prediction.disabled, playground.revealed,
        playground.result_box.children, playground.reveal_button.description,
        playground.feedback, playground.feedback_view.value,
        playground.status.value,
    )


@pytest.mark.parametrize("kind,widget_type,fail_at", [
    ("explorer", "Button", 1), ("walkthrough", "Button", 2),
    ("playground", "Button", 2),
    ("explorer", "HBox", 1), ("walkthrough", "HBox", 1),
    ("playground", "HBox", 2),
    ("walkthrough", "HTML", 1),
    ("playground", "HTML", 1),
])
def test_partial_allocation_closes_controls_styles_and_explicit_layouts(
    kind, widget_type, fail_at, monkeypatch, widget_models,
):
    original = getattr(widgets, widget_type)
    calls = 0

    def allocate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("injected allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(widgets, widget_type, allocate)
    with pytest.raises(RuntimeError, match="injected allocation failure"):
        create_controller(kind)
    assert calls == fail_at
    assert not widget_models()


@pytest.mark.parametrize("kind,controller,method", [
    ("explorer", FocusExplorer, "_show"),
    ("walkthrough", FocusExplorer, "_show"),
    ("playground", ReductionPlayground, "_show_result"),
])
def test_initial_display_failure_closes_all_allocated_models(
    kind, controller, method, monkeypatch, widget_models,
):
    def fail(*args):
        raise ValueError("injected display failure")

    monkeypatch.setattr(controller, method, fail)
    with pytest.raises(ValueError, match="injected display failure"):
        create_controller(kind)
    assert not widget_models()


@pytest.mark.parametrize("kind,key,template", [
    ("explorer", "interactive.showing", "{coordinate:d}"),
    ("walkthrough", "interactive.showing", "{coordinate:d}"),
    ("playground", "playground.shape", "{shape:d}"),
])
def test_catalog_format_failure_during_construction_leaks_no_models(
    kind, key, template, bad_catalog, widget_models,
):
    bad_catalog(key, template)
    with pytest.raises(TypeError):
        create_controller(kind)
    assert not widget_models()


@pytest.mark.parametrize("kind", ["explorer", "walkthrough"])
def test_catalog_format_failure_preserves_focus_visual_and_control_snapshot(
    kind, bad_catalog, widget_models,
):
    controller = create_controller(kind)
    state = explorer_state if kind == "explorer" else lesson_state
    before = state(controller)
    models = widget_models()
    bad_catalog("interactive.showing", "{coordinate:d}")
    with pytest.raises(TypeError):
        controller.set_focus((1,))
    assert state(controller) == before
    assert widget_models() == models
    controller.close()
    assert not widget_models()


def test_lesson_control_preparation_failure_does_not_commit_candidate(
    monkeypatch, widget_models,
):
    lesson = create_controller("walkthrough")
    before = lesson_state(lesson)

    def fail(*args):
        raise ValueError("injected lesson label failure")

    monkeypatch.setattr(Walkthrough, "_prepare_sync", fail)
    with pytest.raises(ValueError, match="injected lesson label failure"):
        lesson.set_focus((1,))
    assert lesson_state(lesson) == before
    with pytest.raises(ValueError, match="injected lesson label failure"):
        lesson.select_occurrence(1)
    assert lesson_state(lesson) == before
    lesson.close()
    assert not widget_models()


def test_lesson_failed_dropdown_restores_prepared_controls_without_retranslation(
    bad_catalog, widget_models,
):
    lesson = create_controller("walkthrough")
    before = lesson_state(lesson)
    bad_catalog("interactive.showing", "{coordinate:d}")
    lesson.occurrences.value = 1
    assert lesson.snapshot is before[1]
    assert lesson._selection == before[2]
    assert lesson.occurrences.value == 0
    assert lesson.term_label.value == before[-1]
    assert lesson.explorer.figure.value == before[0][5]
    assert "format" in lesson.explorer.status.value
    lesson.close()
    assert not widget_models()


def test_playground_failed_replacement_closes_candidate_and_preserves_prior_state(
    bad_catalog, widget_models,
):
    playground = create_controller("playground")
    playground.explorer.set_focus((1,))
    playground.prediction.value = "(2,)"
    before = playground_state(playground)
    explorer_before = explorer_state(playground.explorer)
    models = widget_models()
    bad_catalog("playground.shape", "{shape:d}")
    with pytest.raises(TypeError):
        playground.set_parameters(axis=0, keepdims=True)
    assert playground_state(playground) == before
    assert explorer_state(playground.explorer) == explorer_before
    assert not playground.explorer._closed
    assert widget_models() == models
    playground.close()
    assert not widget_models()


@pytest.mark.parametrize("callback", [False, True])
def test_failed_reveal_preserves_hidden_result_and_prediction(
    callback, bad_catalog, widget_models,
):
    playground = create_controller("playground", predict=True)
    playground.prediction.value = "(2,)"
    before = playground_state(playground)
    bad_catalog("playground.shape", "{shape:d}")
    if callback:
        playground.reveal_button.click()
        assert "format" in playground.status.value
        assert playground_state(playground)[:-1] == before[:-1]
    else:
        with pytest.raises(TypeError):
            playground.reveal()
        assert playground_state(playground) == before
    playground.close()
    assert not widget_models()
