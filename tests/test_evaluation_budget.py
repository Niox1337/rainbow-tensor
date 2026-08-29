"""Math previews bound expensive contractions without presenting partial values."""

import gc
import weakref
from importlib import import_module

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.evaluation import budgeted_values
from rainbow_tensor.layout import build_layout

OPERATIONS = ("sum", "mean", "matmul", "einsum")


@pytest.mark.parametrize("status, retained", [("skipped", 0), ("evaluated", 2)])
def test_unknown_requests_do_not_retain_unbounded_coordinate_objects(status, retained):
    class Coordinate:
        """A weakly observable key for checking callback retention."""

    @budgeted_values({"status": status, "output_count": 2})
    def value(coordinate):
        return 1

    references = []
    for _ in range(1000):
        coordinate = Coordinate()
        references.append(weakref.ref(coordinate))
        value(coordinate)
    del coordinate
    gc.collect()
    assert sum(reference() is not None for reference in references) == retained


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
        "reason": "max_terms",
        "term_count": 1_000_000,
        "max_terms": 10_000,
        "max_total_terms": 100_000,
        "output_count": len(renderer.panels[-1]),
        "total_terms": 1_000_000 * len(renderer.panels[-1]),
        "scope": "per_output_cell_and_preview",
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


@pytest.mark.parametrize("operation", OPERATIONS)
def test_total_limit_skips_every_output_without_partial_calculations(operation):
    arrays = _arrays(operation, 3)
    renderer = PanelValuesRenderer()
    visual = _render(operation, arrays, renderer=renderer, max_total_terms=3)

    evaluation = visual.metadata["value_evaluation"]
    assert evaluation["status"] == "skipped"
    assert evaluation["reason"] == "max_total_terms"
    assert evaluation["total_terms"] == 3 * len(renderer.panels[-1])
    assert set(renderer.panels[-1].values()) == {"?"}
    assert sum(array.reads for array in arrays) == sum(
        len(panel) for panel in renderer.panels[:-1]
    )
    assert "max_total_terms=3" in visual.text
    assert visual.trace.complete


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("unbounded", [False, True])
def test_total_threshold_equality_and_explicit_opt_out(operation, unbounded):
    arrays = _arrays(operation, 3)
    renderer = PanelValuesRenderer()
    output_count = 2 if operation in ("sum", "mean") else 6
    limit = None if unbounded else np.int64(3 * output_count)

    visual = _render(operation, arrays, renderer=renderer, max_total_terms=limit)

    assert set(renderer.panels[-1].values()) == {1 if operation == "mean" else 3}
    assert visual.metadata["value_evaluation"]["status"] == "evaluated"
    assert visual.metadata["value_evaluation"]["total_terms"] == 3 * output_count
    source_reads = sum(len(panel) for panel in renderer.panels[:-1])
    assert sum(array.reads for array in arrays) == source_reads + 3 * output_count * len(arrays)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("limit", [0, -1, True, False, 3.0, "3", np.bool_(True)])
def test_invalid_total_limit_fails_before_reading_array_values(operation, limit):
    arrays = _arrays(operation, 3)
    with pytest.raises(ValueError, match="max_total_terms must be a positive integer or None"):
        _render(operation, arrays, max_total_terms=limit)
    assert all(array.reads == 0 for array in arrays)


def test_default_total_limit_prevents_many_individually_affordable_reductions():
    array = CountingArray((20, 10_000))
    renderer = PanelValuesRenderer()
    visual = rt.sum(array, axis=1, renderer=renderer)

    evaluation = visual.metadata["value_evaluation"]
    assert evaluation["term_count"] == evaluation["max_terms"]
    assert evaluation["total_terms"] > evaluation["max_total_terms"]
    assert evaluation["reason"] == "max_total_terms"
    assert array.reads == len(renderer.panels[0])
    assert set(renderer.panels[-1].values()) == {"?"}


def test_total_cost_counts_only_visible_outputs_and_keeps_a_distant_focus():
    array = CountingArray((100, 40, 40))
    renderer = PanelValuesRenderer()
    visual = rt.sum(
        array, axis=0, focus=(20, 20), renderer=renderer,
        max_terms=100, max_total_terms=25_000,
    )

    evaluation = visual.metadata["value_evaluation"]
    assert evaluation["status"] == "evaluated"
    assert evaluation["output_count"] == len(renderer.panels[-1]) < 40 * 40
    assert evaluation["total_terms"] == 100 * len(renderer.panels[-1])
    assert renderer.panels[-1][(20, 20)] == 100
    assert visual.trace.output_coord == (20, 20)
    assert array.reads == len(renderer.panels[0]) + evaluation["total_terms"]


def test_renderer_repeated_requests_are_cached_and_unplanned_outputs_are_unknown():
    class RepeatingRenderer(PanelValuesRenderer):
        """Probe repeated and hidden output requests after the normal layout."""

        def render_panels(self, *, panels, theme, **kwargs):
            result = super().render_panels(panels=panels, theme=theme, **kwargs)
            for coord, expected in self.panels[-1].items():
                assert panels[-1]["value_fn"](coord) == expected
                assert panels[-1]["value_fn"](coord) == expected
            hidden = next((i,) for i in range(1000) if (i,) not in self.panels[-1])
            assert panels[-1]["value_fn"](hidden) == "?"
            return result

    array = CountingArray((1000, 3))
    renderer = RepeatingRenderer()
    visual = rt.sum(array, axis=1, renderer=renderer, max_total_terms=100)
    evaluation = visual.metadata["value_evaluation"]
    assert evaluation["status"] == "evaluated"
    assert array.reads == len(renderer.panels[0]) + evaluation["total_terms"]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_scalar_output_can_hit_the_total_budget_exactly(operation):
    arrays = [CountingArray((3,)), CountingArray((3,))]
    renderer = PanelValuesRenderer()
    options = {"focus": (), "renderer": renderer, "max_total_terms": 3}
    if operation in ("sum", "mean"):
        visual = getattr(rt, operation)(arrays[0], axis=0, **options)
    elif operation == "matmul":
        visual = rt.matmul(*arrays, **options)
    else:
        visual = rt.einsum("i,i->", *arrays, **options)
    assert visual.result_shape == ()
    assert renderer.panels[-1] == {(): 1 if operation == "mean" else 3}
    assert visual.metadata["value_evaluation"]["output_count"] == 1
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
    assert renderer.panels[-1] == {(): "?"}
    assert visual.trace.output_coord == ()
    assert visual.trace.term_count == 3
    assert visual.trace.complete
