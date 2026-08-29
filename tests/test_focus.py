"""Focused-output provenance without eager backend reads."""

import tracemalloc
from dataclasses import FrozenInstanceError
from math import prod

import numpy as np
import pytest

import rainbow_tensor as rt


class RecordingRenderer:
    name = "focus-recording"
    mime_type = "text/plain"

    def __init__(self):
        self.calls = []

    def render_tensor(self, **kwargs):
        self.calls.append(kwargs)
        return "tensor"

    def render_panels(self, **kwargs):
        self.calls.append(kwargs)
        return "panels"


class CountingArray:
    def __init__(self, shape):
        self.shape = shape
        self.reads = []

    def __getitem__(self, coordinate):
        self.reads.append(coordinate)
        return sum(coordinate) + 1


def trace_value(trace, arrays):
    """Evaluate the recorded expression independently of the view helpers."""
    assert trace.complete
    return sum(
        prod(arrays[ref.operand][ref.coordinate] for ref in term)
        for term in trace.terms
    ) / trace.divisor


def test_focus_equations_use_the_same_names_as_the_visible_panels():
    """A learner can match every factor in the equation to a labelled panel."""
    matrix = rt.matmul((2, 3), (3, 4), focus=(1, 2))
    reduction = rt.mean((2, 3), 1, focus=(1,))
    contraction = rt.einsum("ij,jk->ik", (2, 3), (3, 4), focus=(1, 2))

    assert "A[1, 0] * B[0, 2]" in matrix.text
    assert "(source[1, 0] + source[1, 1] + source[1, 2]) / 3" in reduction.text
    assert "operand 0[1, 0] * operand 1[0, 2]" in contraction.text


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("axis", [1, -2])
def test_reduction_focus_tracks_sources_and_preserves_group_tints(operation, axis):
    array = np.arange(24).reshape(2, 3, 4)
    renderer = RecordingRenderer()
    visual = operation(array, axis, focus=(np.int64(-1), -1), renderer=renderer)
    source, result = renderer.calls[0]["panels"]

    assert visual.trace.operation == operation.__name__
    assert visual.trace.output_coord == (1, 3)
    assert visual.trace.term_count == 3
    assert visual.trace.divisor == (3 if operation is rt.mean else 1)
    assert visual.selected == [(1, k, 3) for k in range(3)]
    assert source["selected"] == visual.selected
    assert result["selected"] == [(1, 3)]
    assert source["cell_tint"]((1, 0, 3)) is None
    assert result["cell_tint"]((1, 3)) is None
    assert source["cell_tint"]((0, 1, 2)) == result["cell_tint"]((0, 2))
    expected = getattr(np, operation.__name__)(array, axis=axis)[1, 3]
    assert trace_value(visual.trace, [array]) == pytest.approx(expected)
    assert result["value_fn"]((1, 3)) == pytest.approx(expected)
    assert "output (1, 3)" in visual.text


@pytest.mark.parametrize(
    "a_shape, b_shape, focus",
    [
        ((2, 3), (3, 4), (1, -1)),
        ((1, 2, 3), (4, 3, 5), (3, 1, 4)),
        ((3,), (2, 3, 4), (-1, 2)),
        ((2, 3, 4), (4,), (1, 2)),
        ((3,), (3,), ()),
    ],
)
def test_matmul_focus_matches_numpy_including_vector_and_batch_rules(a_shape, b_shape, focus):
    a = np.arange(prod(a_shape)).reshape(a_shape)
    b = np.arange(prod(b_shape)).reshape(b_shape)
    renderer = RecordingRenderer()
    visual = rt.matmul(a, b, focus=focus, renderer=renderer)
    trace = visual.trace
    panels = renderer.calls[0]["panels"]
    expected = np.matmul(a, b)[trace.output_coord]

    assert trace.operation == "matmul"
    assert trace.term_count == a_shape[-1]
    assert trace_value(trace, [a, b]) == expected
    assert panels[-1]["selected"] == [trace.output_coord]
    assert panels[-1]["value_fn"](trace.output_coord) == expected
    for operand in range(2):
        assert set(panels[operand]["selected"]) == {
            term[operand].coordinate for term in trace.terms
        }


@pytest.mark.parametrize(
    "subscripts, shapes, focus",
    [
        ("...ij,...jk->...ik", [(1, 2, 3), (4, 3, 5)], (3, 1, 4)),
        ("ij,j->i", [(2, 1), (3,)], (1,)),
        ("ij,ij->ij", [(1, 3), (2, 3)], (1, 2)),
        ("ii->i", [(3, 3)], (-1,)),
        ("ij->", [(2, 3)], ()),
        ("ij,jk,kl->il", [(2, 2), (2, 3), (3, 4)], (1, 2)),
    ],
)
def test_einsum_focus_tracks_ordered_terms_and_highlights_sources(subscripts, shapes, focus):
    arrays = [np.arange(prod(shape)).reshape(shape) for shape in shapes]
    renderer = RecordingRenderer()
    visual = rt.einsum(subscripts, *arrays, focus=focus, renderer=renderer)
    trace = visual.trace
    panels = renderer.calls[0]["panels"]
    expected = np.einsum(subscripts, *arrays)[trace.output_coord]

    assert trace.operation == "einsum"
    assert trace_value(trace, arrays) == expected
    assert panels[-1]["selected"] == [trace.output_coord]
    assert panels[-1]["value_fn"](trace.output_coord) == expected
    for operand in range(len(arrays)):
        expected_sources = {term[operand].coordinate for term in trace.terms}
        assert set(panels[operand]["selected"]) == expected_sources
        assert set(visual.selected[operand]) == expected_sources


def test_einsum_trace_preserves_reused_broadcast_factors():
    visual = rt.einsum("ij,j->i", (2, 1), (3,), focus=(1,), renderer=RecordingRenderer())

    assert [term[0].coordinate for term in visual.trace.terms] == [(1, 0)] * 3
    assert [term[1].coordinate for term in visual.trace.terms] == [(0,), (1,), (2,)]


def test_einsum_trace_orders_multiple_contracted_labels_in_input_order():
    visual = rt.einsum(
        "ij,jk,kl->il", (2, 2), (2, 3), (3, 4), focus=(1, 2), renderer=RecordingRenderer()
    )

    assert [term[1].coordinate for term in visual.trace.terms] == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
    ]


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_trace_is_bounded_and_does_not_read_backend_values(operation):
    array = CountingArray((20,))
    renderer = RecordingRenderer()
    visual = operation(array, 0, focus=(), renderer=renderer)

    assert array.reads == []
    assert visual.trace.output_coord == ()
    assert visual.trace.term_count == 20
    assert len(visual.trace.terms) == 8
    assert not visual.trace.complete
    assert "12 omitted" in visual.text
    assert [term[0].coordinate for term in visual.trace.terms] == [(k,) for k in range(8)]


def test_trace_and_operand_references_are_immutable():
    visual = rt.sum((2, 3), 0, focus=(1,), renderer=RecordingRenderer())

    with pytest.raises(FrozenInstanceError):
        visual.trace.output_coord = (2,)
    with pytest.raises(FrozenInstanceError):
        visual.trace.terms[0][0].coordinate = (1, 2)


@pytest.mark.parametrize("operation", ["matmul", "einsum"])
def test_contraction_trace_is_bounded_without_reading_values(operation):
    a = CountingArray((2, 20))
    b = CountingArray((20, 3))
    renderer = RecordingRenderer()
    if operation == "matmul":
        visual = rt.matmul(a, b, focus=(1, 2), renderer=renderer)
    else:
        visual = rt.einsum("ij,jk->ik", a, b, focus=(1, 2), renderer=renderer)

    assert a.reads == b.reads == []
    assert visual.trace.term_count == 20
    assert len(visual.trace.terms) == 8
    assert not visual.trace.complete
    assert "12 omitted" in visual.text
    assert [term[0].coordinate for term in visual.trace.terms] == [(1, k) for k in range(8)]
    assert [term[1].coordinate for term in visual.trace.terms] == [(k, 2) for k in range(8)]


@pytest.mark.parametrize("operation", ["sum", "mean", "matmul", "einsum"])
@pytest.mark.parametrize(
    "focus, error",
    [
        ([0], TypeError),
        ((True,), TypeError),
        ((np.bool_(False),), TypeError),
        ((1.0,), TypeError),
        ((), ValueError),
        ((0, 0), ValueError),
        ((999,), IndexError),
        ((-999,), IndexError),
    ],
)
def test_invalid_focus_fails_before_backend_reads_or_rendering(operation, focus, error):
    array = CountingArray((2, 3))
    renderer = RecordingRenderer()
    with pytest.raises(error, match="focus"):
        if operation in ("sum", "mean"):
            getattr(rt, operation)(array, 0, focus=focus, renderer=renderer)
        elif operation == "matmul":
            rt.matmul(array, (3,), focus=focus, renderer=renderer)
        else:
            rt.einsum("ij->j", array, focus=focus, renderer=renderer)

    assert array.reads == []
    assert renderer.calls == []


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_scalar_outputs_require_an_empty_focus_tuple(operation):
    with pytest.raises(ValueError, match="focus"):
        operation((3,), 0, focus=(0,))


def test_default_focus_keeps_einsum_panels_and_explanation_unchanged():
    renderer = RecordingRenderer()
    visual = rt.einsum("ij,jk->ik", (2, 3), (3, 4), renderer=renderer)

    assert visual.trace.output_coord == (0, 0)
    assert visual.trace.complete
    assert all("selected" not in panel for panel in renderer.calls[0]["panels"])
    assert "Focus:" not in visual.text


def test_tracing_only_consumes_the_stored_term_prefix():
    from rainbow_tensor.tracing import _build_trace

    consumed = []

    def terms():
        for k in range(1000):
            consumed.append(k)
            yield ((k,),)

    trace = _build_trace("sum", (), 1000, terms())

    assert consumed == list(range(8))
    assert len(trace.terms) == 8
    assert trace.term_count == 1000
    assert not trace.complete


def test_einsum_trace_does_not_pool_a_large_contraction_axis():
    from rainbow_tensor.ops.einsum import einsum_source_terms
    from rainbow_tensor.tracing import _build_trace

    tracemalloc.start()
    try:
        terms = einsum_source_terms((("i",),), (), [(1_000_000,)], ())
        trace = _build_trace("einsum", (), 1_000_000, terms)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert trace.terms[-1][0].coordinate == (7,)
    assert not trace.complete
    assert peak < 1_000_000
