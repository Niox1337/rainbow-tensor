"""Notebook controls keep static focus behavior and computation limits intact."""

import builtins
import subprocess
import sys

import numpy as np
import pytest

import rainbow_tensor as rt


@pytest.fixture
def explorers():
    pytest.importorskip("ipywidgets")
    created = []

    def make(operation, *args, **kwargs):
        explorer = rt.explore(operation, *args, **kwargs)
        created.append(explorer)
        return explorer

    yield make
    for explorer in created:
        explorer.close()


def test_static_import_does_not_load_widgets():
    code = (
        "import sys\nimport rainbow_tensor as rt\n"
        "rt.shape((2, 3))\nassert 'ipywidgets' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_missing_widgets_gives_install_instruction(monkeypatch):
    original = builtins.__import__

    def without_widgets(name, *args, **kwargs):
        if name == "ipywidgets":
            raise ModuleNotFoundError("ipywidgets")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_widgets)
    with pytest.raises(ImportError, match=r"rainbow-tensor\[interactive\]"):
        rt.explore(rt.sum, (2, 3), axis=1)


def test_unsupported_operation_is_rejected():
    with pytest.raises(ValueError, match="supports"):
        rt.explore(rt.shape, (2, 3))


def test_button_changes_focus_and_preserves_static_export(explorers, tmp_path):
    a = np.arange(6).reshape(2, 3)
    b = np.arange(12).reshape(3, 4)
    explorer = explorers(rt.matmul, a, b)
    explorer.coordinates[0].value = 1
    explorer.coordinates[1].value = 2
    explorer.update_button.click()
    assert explorer.focus == (1, 2)
    assert explorer.visual.trace.output_coord == (1, 2)
    assert explorer.visual.svg == rt.matmul(a, b, focus=(1, 2)).svg
    path = tmp_path / "focused.svg"
    explorer.visual.save(path)
    assert path.read_text(encoding="utf-8") == explorer.visual.svg


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_negative_focus_and_mutated_values_are_refreshed(explorers, operation):
    array = np.arange(6).reshape(2, 3)
    explorer = explorers(operation, array, axis=1)
    old = explorer.visual.svg
    array[1] = [10, 20, 30]
    visual = explorer.set_focus((-1,))
    assert explorer.focus == (1,)
    assert visual.svg == operation(array, axis=1, focus=(1,)).svg
    assert visual.svg != old


def test_scalar_result_has_no_coordinate_controls(explorers):
    explorer = explorers(rt.einsum, "i,i->", (3,), (3,))
    assert explorer.coordinates == ()
    explorer.update_button.click()
    assert explorer.visual.trace.output_coord == ()


def test_budget_is_not_relaxed_by_interaction(explorers):
    explorer = explorers(rt.matmul, (2, 1_000_000), (1_000_000, 2), max_terms=10)
    visual = explorer.set_focus((1, 1))
    assert visual.metadata["value_evaluation"]["status"] == "skipped"
    assert visual.metadata["value_evaluation"]["max_terms"] == 10
    assert len(visual.trace.terms) == 8


def test_total_budget_is_preserved_when_focus_changes(explorers):
    explorer = explorers(rt.sum, (2, 3), axis=1, max_total_terms=3)
    explorer.coordinates[0].value = 1
    explorer.update_button.click()
    evaluation = explorer.visual.metadata["value_evaluation"]
    assert evaluation["status"] == "skipped"
    assert evaluation["reason"] == "max_total_terms"
    assert evaluation["max_total_terms"] == 3
    assert evaluation["total_terms"] == 6
    assert explorer.visual.trace.output_coord == (1,)


def test_invalid_python_update_preserves_visual(explorers):
    explorer = explorers(rt.sum, (2, 3), axis=1)
    before = explorer.visual
    with pytest.raises(IndexError):
        explorer.set_focus((2,))
    assert explorer.visual is before
    assert explorer.focus == (0,)


def test_button_error_preserves_visual(explorers):
    array = np.arange(6).reshape(2, 3)
    explorer = explorers(rt.sum, array, axis=1)
    before = explorer.visual
    array.shape = (3, 2)
    explorer.update_button.click()
    assert explorer.visual is before
    assert "output shape changed" in explorer.status.value


def test_closed_explorer_rejects_updates(explorers):
    explorer = explorers(rt.sum, (2, 3), axis=1)
    layout = explorer.widget.layout
    style = explorer.update_button.style
    explorer.close()
    explorer.close()
    assert layout.comm is None
    assert style.comm is None
    with pytest.raises(RuntimeError, match="closed"):
        explorer.set_focus((1,))


def test_changed_default_renderer_cannot_replace_svg_with_plain_text(explorers):
    class TextRenderer:
        name = "interactive-test-text"
        mime_type = "text/plain"

        def render_tensor(self, **kwargs):
            return "plain tensor"

        def render_panels(self, **kwargs):
            return "plain panels"

    explorer = explorers(rt.sum, (2, 3), axis=1)
    before = explorer.visual
    original = rt.get_default_renderer()
    try:
        rt.set_default_renderer(TextRenderer())
        with pytest.raises(ValueError, match="SVG renderer"):
            explorer.set_focus((1,))
        assert explorer.visual is before
        assert explorer.focus == (0,)
    finally:
        rt.set_default_renderer(original)


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_kept_reduction_controls_refresh_focus_and_values(explorers, operation):
    array = np.arange(24).reshape(2, 3, 4)
    explorer = explorers(operation, array, axis=(0, 2), keepdims=True)
    assert explorer.result_shape == (1, 3, 1)
    assert tuple(control.min for control in explorer.coordinates) == (0, 0, 0)
    assert tuple(control.max for control in explorer.coordinates) == (0, 2, 0)

    explorer.coordinates[1].value = 2
    explorer.update_button.click()
    assert explorer.focus == (0, 2, 0)
    assert explorer.visual.trace.output_coord == (0, 2, 0)
    assert {
        term[0].coordinate for term in explorer.visual.trace.terms
    } == {(i, 2, k) for i in range(2) for k in range(4)}
    before = explorer.visual.svg
    array[:, 2, :] += 100
    visual = explorer.set_focus((-1, -1, -1))
    assert explorer.focus == (0, 2, 0)
    assert tuple(control.value for control in explorer.coordinates) == (0, 2, 0)
    assert visual.svg == operation(array, axis=(0, 2), keepdims=True, focus=(0, 2, 0)).svg
    assert visual.svg != before


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("keepdims", [False, True])
def test_all_axis_reduction_controls_follow_result_rank(explorers, operation, keepdims):
    array = np.arange(6).reshape(2, 3)
    explorer = explorers(operation, array, keepdims=keepdims)
    expected_shape = (1, 1) if keepdims else ()
    expected_focus = (0, 0) if keepdims else ()
    assert explorer.result_shape == expected_shape
    assert tuple(control.max for control in explorer.coordinates) == expected_focus
    explorer.update_button.click()
    assert explorer.visual.trace.output_coord == expected_focus
    assert explorer.visual.trace.term_count == 6
    assert explorer.visual.svg == operation(array, keepdims=keepdims, focus=expected_focus).svg


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("keepdims", [False, True])
def test_empty_axis_tuple_preserves_coordinate_controls(explorers, operation, keepdims):
    array = np.arange(6).reshape(2, 3)
    explorer = explorers(operation, array, axis=(), keepdims=keepdims)
    assert explorer.result_shape == array.shape
    assert tuple(control.max for control in explorer.coordinates) == (1, 2)
    visual = explorer.set_focus((-1, -1))
    assert explorer.focus == (1, 2)
    assert visual.trace.output_coord == (1, 2)
    assert visual.trace.term_count == 1
    assert visual.trace.divisor == 1
    assert visual.trace.terms[0][0].coordinate == (1, 2)
    assert visual.svg == operation(array, axis=(), keepdims=keepdims, focus=(1, 2)).svg


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_invalid_kept_axis_focus_preserves_visual_and_controls(explorers, operation):
    explorer = explorers(operation, (2, 3, 4), axis=(0, 2), keepdims=True, focus=(0, 1, 0))
    before = explorer.visual
    with pytest.raises(IndexError):
        explorer.set_focus((1, 1, 0))
    assert explorer.visual is before
    assert explorer.focus == (0, 1, 0)
    assert tuple(control.value for control in explorer.coordinates) == (0, 1, 0)


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("limits, reason", [
    ({"max_terms": 7}, "max_terms"),
    ({"max_terms": 8, "max_total_terms": 23}, "max_total_terms"),
])
def test_kept_reduction_focus_preserves_both_budgets(explorers, operation, limits, reason):
    explorer = explorers(operation, (2, 3, 4), axis=(0, 2), keepdims=True, **limits)
    before = dict(explorer.visual.metadata["value_evaluation"])
    explorer.coordinates[1].value = 2
    explorer.update_button.click()
    evaluation = explorer.visual.metadata["value_evaluation"]
    assert evaluation == before
    assert evaluation["status"] == "skipped"
    assert evaluation["reason"] == reason
    for key, limit in limits.items():
        assert evaluation[key] == limit
    assert evaluation["term_count"] == 8
    assert evaluation["output_count"] == 3
    assert evaluation["total_terms"] == 24
    assert explorer.visual.trace.output_coord == (0, 2, 0)


@pytest.mark.parametrize("operation, args, kwargs", [
    (rt.sum, ((0, 3),), {"axis": 1}),
    (rt.mean, ((2, 0, 3),), {"axis": 2, "keepdims": True}),
    (rt.matmul, ((0, 2), (2, 3)), {}),
    (rt.einsum, ("ij,jk->ik", (0, 2), (2, 3)), {}),
])
def test_empty_results_have_no_focus_or_active_controls(explorers, operation, args, kwargs):
    explorer = explorers(operation, *args, **kwargs)
    before = explorer.visual
    assert explorer.focus is None
    assert explorer.coordinates == ()
    assert explorer.update_button.disabled
    assert explorer.visual.trace is None
    assert "Empty result" in explorer.status.value
    explorer.update_button.click()
    assert explorer.visual is before
    with pytest.raises(IndexError, match="empty"):
        explorer.set_focus(())
    assert explorer.visual is before


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_empty_groups_still_allow_focusing_real_output_cells(explorers, operation):
    explorer = explorers(operation, (2, 0, 3), axis=1)
    assert not explorer.update_button.disabled
    assert tuple(control.max for control in explorer.coordinates) == (1, 2)
    visual = explorer.set_focus((1, 2))
    assert visual.trace.output_coord == (1, 2)
    assert visual.trace.term_count == 0


def test_scalar_source_can_refresh_its_single_value(explorers):
    array = np.array(3)
    explorer = explorers(rt.sum, array)
    before = explorer.visual.svg
    assert explorer.focus == ()
    assert explorer.coordinates == ()
    assert not explorer.update_button.disabled
    array[...] = 9
    explorer.update_button.click()
    assert explorer.visual.svg != before
    assert explorer.visual.trace.terms[0][0].coordinate == ()
