"""Real CPU tensor reads preserve comparison and selection contracts."""

import numpy as np
import pytest

from rainbow_tensor.provenance.flow import Flow
from rainbow_tensor.provenance.query import ValueBudgetExceeded
from rainbow_tensor.views import elementwise, selection
from test_flow_backends import RecordingRenderer
from test_flow_backends import backend as backend

COMPARISONS = ("greater", "greater_equal", "less", "less_equal", "equal", "not_equal")
EXTREMA = ("min", "max", "argmin", "argmax")


@pytest.mark.parametrize("operation", COMPARISONS)
def test_cpu_comparisons_match_numpy_and_keep_ordered_origins(backend, operation):
    left_reference = np.array([[np.nan], [2], [np.inf]], dtype="float32")
    right_reference = np.array([2, np.inf], dtype="float32")
    left = backend.array(left_reference)
    right = backend.array(right_reference)
    expected = getattr(np, operation)(left_reference, right_reference)
    renderer = RecordingRenderer()
    visual = getattr(elementwise, operation)(left, right, focus=(1, 0), renderer=renderer)
    np.testing.assert_array_equal(renderer.panels[-1], expected)
    assert [(ref.operand, ref.coordinate) for ref in visual.trace.expression.operands] == [
        (0, (1, 0)), (1, (0,)),
    ]
    flow = Flow()
    result = getattr(flow, operation)(flow.input(left), flow.input(right))
    assert result.trace((1, 0)).steps[0].binary_operator == operation
    assert isinstance(result.value((1, 0)), bool)
    assert result.value((1, 0)) == bool(expected[1, 0])


def test_cpu_where_broadcasts_conditions_and_preserves_chosen_coordinates(backend):
    condition_reference = np.array([[True], [False]])
    left_reference = np.array([1, 2, 3], dtype="int32")
    right_reference = np.array(-4.5, dtype="float32")
    renderer = RecordingRenderer()
    visual = selection.where(
        backend.array(condition_reference), backend.array(left_reference),
        backend.array(right_reference), focus=(1, 2), renderer=renderer,
    )
    expected = np.where(condition_reference, left_reference, right_reference)
    np.testing.assert_array_equal(renderer.panels[-1], expected)
    assert visual.shape == (2, 1)
    assert visual.result_shape == (2, 3)
    decision = next(item for item in visual.metadata["evaluated_selections"]
                    if item["output"]["coordinate"] == (1, 2))
    assert decision["source"]["coordinate"] == ()
    assert decision["position"] == 2
    assert decision["reason"] == "condition_false"


@pytest.mark.parametrize("operation", EXTREMA)
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_cpu_extrema_match_real_nan_and_first_tie_semantics(backend, operation, axis, keepdims):
    reference = np.array([[2, 2, -1], [np.nan, np.nan, 3]], dtype="float32")
    array = backend.transpose(backend.array(reference.T))
    renderer = RecordingRenderer()
    visual = getattr(selection, operation)(
        array, axis=axis, keepdims=keepdims, renderer=renderer,
    )
    expected = getattr(np, operation)(reference, axis=axis, keepdims=keepdims)
    np.testing.assert_equal(renderer.panels[-1], expected)
    assert visual.shape == reference.shape
    assert visual.result_shape == expected.shape
    assert visual.trace.terms == ()
    assert visual.metadata["evaluated_selections"]
    flow = Flow()
    result = getattr(flow, operation)(flow.input(array), axis=axis, keepdims=keepdims)
    assert result.trace().steps[0].selection_operator == operation
    np.testing.assert_equal(result.value(), expected[(0,) * expected.ndim])


@pytest.mark.parametrize("operation", (*COMPARISONS, "where", *EXTREMA))
def test_cpu_new_operations_accept_scalars_and_empty_surviving_outputs(backend, operation):
    scalar = backend.array(np.array(7, dtype="float32"))
    empty = backend.array(np.empty((0, 3), dtype="float32"))
    renderer = RecordingRenderer()
    if operation in COMPARISONS:
        scalar_visual = getattr(elementwise, operation)(scalar, 3, renderer=renderer)
        expected = getattr(np, operation)(7, 3)
        assert bool(renderer.panels[-1]) == bool(expected)
        empty_visual = getattr(elementwise, operation)(empty, 3, renderer=renderer)
        empty_shape = (0, 3)
    elif operation == "where":
        scalar_visual = selection.where(
            backend.array(np.array(True)), scalar, 0, renderer=renderer,
        )
        assert renderer.panels[-1].item() == 7
        empty_visual = selection.where(True, empty, 0, renderer=renderer)
        empty_shape = (0, 3)
    else:
        scalar_visual = getattr(selection, operation)(scalar, axis=0, renderer=renderer)
        assert renderer.panels[-1].item() == (0 if operation.startswith("arg") else 7)
        empty_visual = getattr(selection, operation)(empty, axis=1, renderer=renderer)
        empty_shape = (0,)
    assert scalar_visual.result_shape == ()
    assert empty_visual.result_shape == empty_shape
    assert empty_visual.trace is None
    assert renderer.panels[-1].size == 0


@pytest.mark.parametrize("operation", EXTREMA)
def test_cpu_extrema_reject_complex_values_after_value_free_trace(backend, operation):
    flow = Flow()
    result = getattr(flow, operation)(
        flow.input(backend.array(np.array([1 + 2j], dtype="complex64"))),
    )
    assert result.trace().complete
    with pytest.raises(TypeError, match="real scalar"):
        result.value()


def test_cpu_conditional_extremum_chain_plans_before_reads_and_reuses_source_cells(
    backend, monkeypatch,
):
    import rainbow_tensor.visual as visual_module

    reference = np.array([[1, 5, 3], [6, 2, 4]], dtype="float32")
    array = backend.transpose(backend.array(reference.T))
    original = visual_module.value_at_coordinate
    reads = []

    def scalar_read(source, coordinate):
        if source is array:
            reads.append(tuple(coordinate))
        return original(source, coordinate)

    monkeypatch.setattr(visual_module, "value_at_coordinate", scalar_read)
    flow = Flow()
    source = flow.input(array)
    condition = flow.greater(source, flow.input(2))
    clipped = flow.where(condition, source, flow.input(0))
    result = flow.max(clipped, axis=1)
    trace = result.trace((0,))
    assert trace.complete
    assert reads == []
    with pytest.raises(ValueBudgetExceeded):
        result.value((0,), max_total_terms=1)
    assert reads == []
    assert result.value((0,)) == 5
    assert reads == [(0, 0), (0, 1), (0, 2)]
    reads.clear()
    renderer = RecordingRenderer()
    visual = result.visualize((0,), renderer=renderer)
    np.testing.assert_equal(renderer.panels[-1], np.where(reference > 2, reference, 0).max(axis=1))
    assert len(reads) == len(set(reads)) == reference.size
    operations = {item["operation"] for item in visual.metadata["evaluated_selections"]}
    assert operations == {"where", "max"}
