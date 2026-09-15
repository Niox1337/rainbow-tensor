"""A focused result preserves exact source identity across shape and copy operations."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.tracing import OperandRef, _trace_explanation
from rainbow_tensor.views._focus import FIRST_OUTPUT


class RecordingRenderer:
    """Keep the panel contract observable without triggering any backend reads."""

    name = "mapped-focus-recording"
    mime_type = "text/plain"

    def __init__(self):
        self.calls = []

    def render_tensor(self, **kwargs):
        self.calls.append(kwargs)
        return "tensor"

    def render_panels(self, **kwargs):
        self.calls.append(kwargs)
        self.panels = kwargs["panels"]
        return "panels"


class CountingArray:
    """Provide bounded scalar access for a logical array without allocating its size."""

    def __init__(self, shape):
        self.shape = shape
        self.reads = []

    def __getitem__(self, coordinate):
        assert len(coordinate) == len(self.shape)
        assert all(0 <= position < size for position, size in zip(coordinate, self.shape))
        self.reads.append(coordinate)
        return 7


MAPPED_CASES = [
    ("reshape", (2, 3, 4), ((4, 6),), {}, (2, 3), (1, 0, 3)),
    ("transpose", (2, 3, 4), (), {}, (3, 2, 1), (1, 2, 3)),
    ("transpose", (2, 3, 4), ((1, 2, 0),), {}, (2, 3, 1), (1, 2, 3)),
    ("swapaxes", (2, 3, 4), (0, 2), {}, (3, 2, 1), (1, 2, 3)),
    ("moveaxis", (2, 3, 4), ((0, 2), (2, 0)), {}, (3, 2, 1), (1, 2, 3)),
    ("squeeze", (1, 2, 1, 3), (), {}, (1, 2), (0, 1, 0, 2)),
    ("squeeze", (1, 1), (), {}, (), (0, 0)),
    ("expand_dims", (2, 3), ((0, -1),), {}, (0, 1, 2, 0), (1, 2)),
    ("repeat", (2, 3), ([2, 0, 3],), {"axis": 1}, (1, 4), (1, 2)),
    ("repeat", (), (3,), {}, (2,), ()),
    ("take", (2, 3), ([2, 0, 2],), {"axis": 1}, (1, 2), (1, 2)),
    ("take", (3,), (-1,), {}, (), (2,)),
    ("take", (), (0,), {}, (), ()),
]


@pytest.mark.parametrize("name,shape,args,kwargs,focus,source", MAPPED_CASES)
def test_mapped_result_and_source_match_numpy(name, shape, args, kwargs, focus, source):
    array = np.arange(np.prod(shape, dtype=int)).reshape(shape)
    expected = getattr(np, name)(array, *args, **kwargs)
    renderer = RecordingRenderer()
    negative = tuple(position - size for position, size in zip(focus, expected.shape))
    visual = getattr(rt, name)(array, *args, **kwargs, focus=negative, renderer=renderer)

    assert visual.result_shape == expected.shape
    assert visual.trace.operation == name
    assert visual.trace.output_coord == focus
    assert visual.trace.terms == ((OperandRef(0, source),),)
    assert visual.trace.term_count == visual.trace.divisor == 1
    assert visual.trace.complete
    assert renderer.panels[0]["selected"] == [source]
    assert renderer.panels[1]["selected"] == [focus]
    assert renderer.panels[1]["value_fn"](focus) == expected[focus] == array[source]
    assert visual.metadata["output_panel_indices"] == (1,)
    assert visual.metadata["focused_output_panel"] == 1
    assert visual.metadata["trace_in_explanation"] is True
    assert visual.explanation.count(_trace_explanation(visual.trace)[0]) == 1
    assert len(renderer.calls) == 1
    assert visual.renderer == renderer.name
    assert visual.mime_type == renderer.mime_type


@pytest.mark.parametrize(
    "operation,axis,focus",
    [
        ("concatenate", 0, (4, 2)),
        ("concatenate", 1, (1, 8)),
        ("stack", 0, (2, 1, 2)),
        ("stack", -1, (1, 2, 2)),
    ],
)
def test_multiple_inputs_retain_original_operand_number(operation, axis, focus):
    arrays = [np.arange(6).reshape(2, 3) + 10 * operand for operand in range(3)]
    expected = getattr(np, operation)(arrays, axis=axis)
    renderer = RecordingRenderer()
    visual = getattr(rt, operation)(arrays, axis=axis, focus=focus, renderer=renderer)
    reference = visual.trace.terms[0][0]
    assert reference.operand == 2
    assert renderer.panels[2]["selected"] == [reference.coordinate]
    assert not renderer.panels[0].get("selected")
    assert not renderer.panels[1].get("selected")
    assert renderer.panels[3]["selected"] == [focus]
    assert renderer.panels[3]["value_fn"](focus) == expected[focus]
    assert arrays[2][reference.coordinate] == expected[focus]
    assert visual.metadata["output_panel_indices"] == (3,)


@pytest.mark.parametrize("operand,source", [(0, (1, 0)), (1, (0, 2))])
def test_broadcast_selects_the_requested_output_port(operand, source):
    arrays = [np.arange(2).reshape(2, 1), np.arange(3).reshape(1, 3) + 10]
    renderer = RecordingRenderer()
    visual = rt.broadcast(*arrays, focus=(1, 2), focus_operand=operand, renderer=renderer)
    assert visual.trace.terms == ((OperandRef(operand, source),),)
    assert renderer.panels[2 * operand]["selected"] == [source]
    assert renderer.panels[2 * operand + 1]["selected"] == [(1, 2)]
    assert not renderer.panels[2 * (1 - operand)].get("selected")
    assert not renderer.panels[2 * (1 - operand) + 1].get("selected")
    assert renderer.panels[2 * operand + 1]["value_fn"]((1, 2)) == arrays[operand][source]
    assert visual.metadata["output_panel_indices"] == (1, 3)
    assert visual.metadata["focused_output_panel"] == 2 * operand + 1
    assert visual.metadata["focus_operand"] == operand


@pytest.mark.parametrize(
    "operand,error",
    [
        (True, TypeError),
        (np.bool_(True), TypeError),
        (0.5, TypeError),
        (-1, ValueError),
        (2, ValueError),
    ],
)
def test_invalid_broadcast_operand_fails_before_reads(operand, error):
    array = CountingArray((2, 3))
    renderer = RecordingRenderer()
    with pytest.raises(error, match="focus_operand"):
        rt.broadcast(array, array, focus=(0, 0), focus_operand=operand, renderer=renderer)
    assert array.reads == renderer.calls == []


@pytest.mark.parametrize("name,shape,args,kwargs,focus,source", MAPPED_CASES)
def test_none_and_omitted_focus_keep_static_output(name, shape, args, kwargs, focus, source):
    omitted = getattr(rt, name)(shape, *args, **kwargs, theme=rt.LIGHT)
    explicit = getattr(rt, name)(shape, *args, **kwargs, theme=rt.LIGHT, focus=None)
    assert omitted.svg == explicit.svg
    assert omitted.explanation == explicit.explanation
    assert omitted.trace is explicit.trace is None
    assert omitted.metadata == explicit.metadata
    assert omitted.metadata["trace_in_explanation"] is False


@pytest.mark.parametrize(
    "focus,error",
    [
        ([0], TypeError),
        ((True,), TypeError),
        ((0.5,), TypeError),
        ((), ValueError),
        ((-7,), IndexError),
        ((6,), IndexError),
    ],
)
@pytest.mark.parametrize(
    "name,args,kwargs",
    [
        ("reshape", ((6,),), {}),
        ("repeat", (1,), {}),
        ("take", (list(range(6)),), {}),
    ],
)
def test_invalid_focus_is_rejected_before_source_reads(name, args, kwargs, focus, error):
    array = CountingArray((6,))
    renderer = RecordingRenderer()
    with pytest.raises(error):
        getattr(rt, name)(array, *args, **kwargs, focus=focus, renderer=renderer)
    assert array.reads == renderer.calls == []


@pytest.mark.parametrize(
    "name,shape,args,kwargs",
    [
        ("reshape", (0, 3), ((3, 0),), {}),
        ("transpose", (0, 3), (), {}),
        ("squeeze", (1, 0), (), {}),
        ("expand_dims", (0,), (0,), {}),
        ("repeat", (3,), (0,), {}),
        ("take", (3,), ([],), {}),
    ],
)
def test_empty_first_output_has_no_trace_and_explicit_focus_fails(name, shape, args, kwargs):
    array = CountingArray(shape)
    renderer = RecordingRenderer()
    visual = getattr(rt, name)(array, *args, **kwargs, focus=FIRST_OUTPUT, renderer=renderer)
    assert visual.trace is None
    assert not any(panel.get("selected") for panel in renderer.panels)
    assert visual.metadata["trace_in_explanation"] is False
    assert len(renderer.calls) == 1
    assert array.reads == []
    with pytest.raises(IndexError, match="empty"):
        getattr(rt, name)(array, *args, **kwargs, focus=(0,) * len(visual.result_shape))
    assert array.reads == []


@pytest.mark.parametrize("shape,target,focus", [((2, 3), (3, 2), (0, 0)), ((), (), ())])
def test_first_output_is_selected_in_one_render(shape, target, focus):
    array = CountingArray(shape)
    renderer = RecordingRenderer()
    visual = rt.reshape(array, target, focus=FIRST_OUTPUT, renderer=renderer)
    assert visual.trace.output_coord == focus
    assert len(renderer.calls) == 1
    assert array.reads == []


def test_huge_repeat_pins_one_source_and_result_without_materializing_repeats():
    array = CountingArray((3,))
    size = 10**30
    position = 2 * size + size // 2
    visual = rt.repeat(
        array,
        size,
        focus=(position,),
        theme=rt.LIGHT.variant(max_cells=6, max_visible_cells=12),
    )
    assert visual.result_shape == (3 * size,)
    assert visual.trace.terms == ((OperandRef(0, (2,)),),)
    root = ET.fromstring(visual.svg)
    selected = [node for node in root.iter() if node.get("fill") == rt.LIGHT.surface_selected]
    assert len(selected) >= 2
    assert len(array.reads) < 30
    assert (2,) in array.reads


def test_squeeze_focus_preserves_removed_and_surviving_axis_colours():
    renderer = RecordingRenderer()
    rt.squeeze((1, 2, 1, 3), focus=(1, 2), theme=rt.LIGHT, renderer=renderer)
    assert renderer.panels[0]["theme"].axis_color(0) == rt.LIGHT.surface_selected
    assert renderer.panels[0]["theme"].axis_color(2) == rt.LIGHT.surface_selected
    assert renderer.panels[1]["theme"].axis_color(0) == rt.LIGHT.axis_color(1)
    assert renderer.panels[1]["theme"].axis_color(1) == rt.LIGHT.axis_color(3)
