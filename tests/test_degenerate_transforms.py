"""Degenerate transforms preserve NumPy shapes, values, and bounded empty previews."""

import tracemalloc

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.ops.combining import repeat_result_shape, repeat_source_positions


class ResultRenderer:
    """Evaluate result callbacks at logical coordinates without display-shape aliases."""

    name = "degenerate-transform-results"
    mime_type = "text/plain"

    def render_panels(self, *, panels, **kwargs):
        result = panels[-1]
        self.values = np.asarray([
            result["value_fn"](coord) for coord in np.ndindex(result["shape"])
        ]).reshape(result["shape"])
        return "recorded transform"

    def render_tensor(self, **kwargs):
        raise AssertionError("expected a comparison")


@pytest.mark.parametrize("shape, operation, args", [
    ((), "reshape", ((1, 1),)),
    ((1, 1), "reshape", ((),)),
    ((0,), "reshape", ((2, 0, 3),)),
    ((2, 0, 3), "reshape", ((-1, 3),)),
    ((), "transpose", ()),
    ((2, 0, 3), "transpose", ((2, 0, 1),)),
    ((2, 0, 3), "swapaxes", (0, 1)),
    ((2, 0, 3), "moveaxis", (1, -1)),
    ((), "squeeze", ()),
    ((), "squeeze", (0,)),
    ((), "squeeze", (-1,)),
    ((1, 0, 1), "squeeze", ()),
    ((1, 1), "squeeze", ()),
    ((), "expand_dims", (0,)),
    ((0,), "expand_dims", ((0, 2),)),
    ((), "repeat", (3,)),
    ((), "repeat", (0,)),
    ((0,), "repeat", (2,)),
    ((2, 0, 3), "repeat", (2, 0)),
    ((2, 0, 3), "repeat", (2, 1)),
    ((3,), "repeat", (0,)),
    ((3,), "repeat", ([0, 2, 0],)),
    ((), "take", (0,)),
    ((), "take", (-1,)),
    ((), "take", ([0, -1],)),
    ((), "take", ([],)),
    ((0,), "take", ([],)),
    ((2, 0, 3), "take", (1, 0)),
    ((2, 0, 3), "take", ([], 1)),
    ((1,), "take", (0,)),
])
def test_transform_values_match_numpy(shape, operation, args):
    array = np.arange(np.prod(shape, dtype=int)).reshape(shape)
    renderer = ResultRenderer()
    visual = getattr(rt, operation)(array, *args, renderer=renderer)
    if operation in {"repeat", "take"} and len(args) == 1:
        expected = getattr(np, operation)(array, *args, axis=0)
    else:
        expected = getattr(np, operation)(array, *args)
    assert visual.result_shape == expected.shape
    assert renderer.values.shape == expected.shape
    np.testing.assert_array_equal(renderer.values, expected)


@pytest.mark.parametrize("operation, shapes, kwargs", [
    ("stack", [(), ()], {}),
    ("stack", [(0,), (0,)], {}),
    ("stack", [(2, 0, 3), (2, 0, 3)], {"axis": -1}),
    ("concatenate", [(0, 3), (2, 3)], {}),
    ("concatenate", [(2, 0), (2, 0)], {"axis": 1}),
])
def test_combining_values_match_numpy(operation, shapes, kwargs):
    arrays = [np.arange(np.prod(s, dtype=int)).reshape(s) for s in shapes]
    renderer = ResultRenderer()
    visual = getattr(rt, operation)(arrays, renderer=renderer, **kwargs)
    expected = getattr(np, operation)(arrays, **kwargs)
    assert visual.result_shape == expected.shape
    np.testing.assert_array_equal(renderer.values, expected)


@pytest.mark.parametrize("shapes", [((), ()), ((), (0,)), ((1, 3), (2, 0, 3))])
def test_broadcast_shapes_and_right_operand_values(shapes):
    arrays = [np.ones(s) for s in shapes]
    renderer = ResultRenderer()
    visual = rt.broadcast(*arrays, renderer=renderer)
    expected = np.broadcast_arrays(*arrays)[-1]
    assert visual.result_shape == expected.shape
    np.testing.assert_array_equal(renderer.values, expected)


@pytest.mark.parametrize("operation, args", [
    (rt.reshape, ((0,), (0, -1))),
    (rt.reshape, ((0,), ())),
    (rt.squeeze, ((), (0,))),
    (rt.concatenate, ([(), ()],)),
    (rt.take, ((0,), [0])),
    (rt.repeat, ((0,), -1)),
    (rt.repeat, ((0,), [-1])),
])
def test_invalid_empty_and_scalar_transforms_are_rejected(operation, args):
    with pytest.raises(ValueError):
        operation(*args)


def test_empty_repeat_stays_bounded_with_a_huge_other_dimension():
    tracemalloc.start()
    try:
        visual = rt.repeat((10**12, 0), 2, axis=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert visual.result_shape == (2 * 10**12, 0)
    assert "No elements" in visual.svg
    assert peak < 1_000_000
    assert repeat_result_shape((10**12,), 0, 0) == (0,)
    assert repeat_source_positions(10**12, 0) == []


@pytest.mark.parametrize("operation, argument", [
    (rt.repeat, iter([0, 2, 0])),
    (rt.take, iter([2, 0])),
])
def test_count_and_index_generators_are_consumed_once(operation, argument):
    renderer = ResultRenderer()
    visual = operation(np.arange(3), argument, renderer=renderer)
    assert visual.result_shape == (2,)
