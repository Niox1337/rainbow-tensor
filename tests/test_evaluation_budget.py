"""Math previews bound expensive contractions without presenting partial values."""

from importlib import import_module

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.layout import build_layout

OPERATIONS = ("sum", "mean", "matmul", "einsum")


class CountingArray:
    """Expose a shape and constant values while recording each scalar read."""

    def __init__(self, shape):
        self.shape = shape
        self.reads = 0

    def __getitem__(self, coordinate):
        assert len(coordinate) == len(self.shape)
        assert all(0 <= index < size for index, size in zip(coordinate, self.shape))
        self.reads += 1
        return 1


class PanelValuesRenderer:
    """Read precisely the source and result cells requested by bounded layouts."""

    name = "panel-values"
    mime_type = "text/plain"

    def render_tensor(self, **kwargs):
        raise AssertionError("math operations should render panels")

    def render_panels(self, *, panels, theme, **kwargs):
        self.panels = []
        for panel in panels:
            layout = build_layout(
                panel["shape"],
                selected=panel.get("selected"),
                value_fn=panel.get("value_fn"),
                theme=panel.get("theme", theme),
            )
            self.panels.append({
                cell.coord: cell.value for cell in layout.cells if not cell.ellipsis
            })
        return "captured panel values"


def _arrays(operation, inner):
    if operation in ("sum", "mean"):
        return [CountingArray((2, inner))]
    return [CountingArray((2, inner)), CountingArray((inner, 3))]


def _render(operation, arrays, **kwargs):
    if operation in ("sum", "mean"):
        return getattr(rt, operation)(arrays[0], axis=1, **kwargs)
    if operation == "matmul":
        return rt.matmul(*arrays, **kwargs)
    return rt.einsum("ij,jk->ik", *arrays, **kwargs)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_default_budget_skips_million_term_outputs_but_keeps_a_bounded_preview(operation):
    arrays = _arrays(operation, 1_000_000)
    renderer = PanelValuesRenderer()

    visual = _render(operation, arrays, renderer=renderer)

    assert set(renderer.panels[-1].values()) == {"?"}
    assert visual.result_shape == ((2,) if operation in ("sum", "mean") else (2, 3))
    assert all(0 < array.reads <= rt.LIGHT.max_visible_cells for array in arrays)
    assert visual.metadata["value_evaluation"] == {
        "status": "skipped",
        "term_count": 1_000_000,
        "max_terms": 10_000,
        "scope": "per_output_cell",
    }
    assert "1,000,000" in visual.text
    assert "max_terms=10,000" in visual.text
    assert "max_terms=None" in visual.text
    assert visual.trace.term_count == 1_000_000
    assert len(visual.trace.terms) == 8
    assert not visual.trace.complete
    if operation in ("matmul", "einsum"):
        assert "highlights sample 8" in visual.text


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("max_terms", [3, np.int64(3), None])
def test_threshold_equality_and_unbounded_opt_in_compute_complete_values(operation, max_terms):
    arrays = _arrays(operation, 3)
    renderer = PanelValuesRenderer()

    visual = _render(operation, arrays, renderer=renderer, max_terms=max_terms)

    expected = 1 if operation == "mean" else 3
    assert set(renderer.panels[-1].values()) == {expected}
    assert visual.metadata["value_evaluation"]["status"] == "evaluated"
    assert visual.metadata["value_evaluation"]["term_count"] == 3
    assert visual.metadata["value_evaluation"]["max_terms"] == max_terms
    assert visual.trace.complete
    assert "were not computed" not in visual.text


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("max_terms", [0, -1, True, False, 3.0, "3", np.bool_(True)])
def test_invalid_budget_is_rejected_before_backend_reads(operation, max_terms):
    arrays = _arrays(operation, 3)

    with pytest.raises(ValueError, match="max_terms must be a positive integer or None"):
        _render(operation, arrays, max_terms=max_terms)

    assert all(array.reads == 0 for array in arrays)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_small_over_budget_outputs_are_unknown_instead_of_partial_sums(operation):
    arrays = _arrays(operation, 3)
    renderer = PanelValuesRenderer()

    visual = _render(operation, arrays, renderer=renderer, max_terms=2)

    assert set(renderer.panels[-1].values()) == {"?"}
    assert visual.metadata["value_evaluation"]["status"] == "skipped"
    assert visual.trace.term_count == 3
    assert visual.trace.complete
    assert "were not computed" in visual.text


@pytest.mark.parametrize("operation", OPERATIONS)
def test_skipped_evaluation_only_enumerates_the_eight_trace_terms(operation, monkeypatch):
    module = import_module(
        "rainbow_tensor.views.einsum" if operation == "einsum"
        else "rainbow_tensor.views.reductions"
    )
    helper = {
        "sum": "reduce_source_coords",
        "mean": "reduce_source_coords",
        "matmul": "iter_matmul_source_terms",
        "einsum": "einsum_source_terms",
    }[operation]
    original = getattr(module, helper)
    generated = 0

    def bounded_terms(*args, **kwargs):
        nonlocal generated
        for term in original(*args, **kwargs):
            generated += 1
            assert generated <= 8, "a skipped preview must not enumerate hidden terms"
            yield term

    monkeypatch.setattr(module, helper, bounded_terms)
    arrays = _arrays(operation, 1_000_000)

    visual = _render(operation, arrays)

    assert generated == 8
    assert visual.metadata["value_evaluation"]["status"] == "skipped"


@pytest.mark.parametrize("operation", OPERATIONS)
def test_over_budget_focus_keeps_the_requested_output_and_source_coordinates(operation):
    arrays = _arrays(operation, 1_000_000)
    renderer = PanelValuesRenderer()
    focus = (1,) if operation in ("sum", "mean") else (1, 2)

    visual = _render(operation, arrays, focus=focus, renderer=renderer)

    assert visual.trace.output_coord == focus
    assert visual.trace.term_count == 1_000_000
    assert len(visual.trace.terms) == 8
    assert renderer.panels[-1][focus] == "?"
    for position, term in enumerate(visual.trace.terms):
        assert term[0].coordinate == (1, position)
        if operation in ("matmul", "einsum"):
            assert term[1].coordinate == (position, 2)
    assert all(array.reads <= rt.LIGHT.max_visible_cells for array in arrays)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_scalar_outputs_obey_the_same_budget_without_losing_their_trace(operation):
    left = CountingArray((3,))
    right = CountingArray((3,))
    renderer = PanelValuesRenderer()
    options = {"focus": (), "max_terms": 2, "renderer": renderer}

    if operation in ("sum", "mean"):
        visual = getattr(rt, operation)(left, axis=0, **options)
    elif operation == "matmul":
        visual = rt.matmul(left, right, **options)
    else:
        visual = rt.einsum("i,i->", left, right, **options)

    assert visual.result_shape == ()
    assert renderer.panels[-1] == {(0,): "?"}
    assert visual.trace.output_coord == ()
    assert visual.trace.term_count == 3
    assert visual.trace.complete
