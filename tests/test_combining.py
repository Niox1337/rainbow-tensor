"""Combining tensors: repeat, take, concatenate, stack, and broadcast.

Every result shape and coordinate mapping is cross-checked against NumPy, so
the operands and the seam the package draws describe the same operation NumPy
performs.
"""

import itertools

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.ops import (
    broadcast_result_shape,
    broadcast_source_coord,
    broadcast_stretched_axes,
    concatenate_result_shape,
    concatenate_source,
    repeat_result_shape,
    repeat_source_coord,
    repeat_source_positions,
    stack_result_shape,
    stack_source,
    take_axis_and_indices,
    take_result_shape,
    take_source_coord,
)

TAKE_CASES = [
    ((4,), [0, 2, 2, 1], 0),
    ((4,), [3, -1, 0], 0),
    ((3, 4), [2, 0, 1], 1),
    ((3, 4), [0, 0, 2], 0),
    ((3, 4), [1, -1], -1),
    ((2, 3, 4), [3, 1, 1, 0], 2),
    ((2, 3, 4), [1, 0], -2),
]


@pytest.mark.parametrize("shape, indices, axis", TAKE_CASES)
def test_take_matches_numpy(shape, indices, axis):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    expected = np.take(x, indices, axis=axis)
    result = take_result_shape(shape, indices, axis)

    assert result == expected.shape
    norm_axis, resolved = take_axis_and_indices(shape, indices, axis)
    for rc in np.ndindex(result):
        assert expected[rc] == x[take_source_coord(rc, resolved, norm_axis)]


def test_take_rejects_out_of_range():
    with pytest.raises(ValueError):
        take_result_shape((3,), [5], 0)
    with pytest.raises(ValueError):
        take_result_shape((3, 4), [0], 5)


REPEAT_CASES = [
    ((4,), 2, 0),
    ((4,), [1, 2, 0, 3], 0),
    ((3, 4), 2, 0),
    ((3, 4), 3, 1),
    ((3, 4), [2, 0, 1], 0),
    ((3, 4), [1, 2, 1, 0], 1),
    ((2, 3, 4), 2, -1),
    ((2, 3, 4), [1, 2], 0),
]


@pytest.mark.parametrize("shape, repeats, axis", REPEAT_CASES)
def test_repeat_matches_numpy(shape, repeats, axis):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    expected = np.repeat(x, repeats, axis=axis)
    result = repeat_result_shape(shape, repeats, axis)

    assert result == expected.shape
    norm_axis = axis + len(shape) if axis < 0 else axis
    positions = repeat_source_positions(shape[norm_axis], repeats)
    for rc in np.ndindex(result):
        assert expected[rc] == x[repeat_source_coord(rc, positions, norm_axis)]


def test_repeat_rejects_bad_input():
    with pytest.raises(ValueError):
        repeat_result_shape((3,), [1, 2], 0)
    with pytest.raises(ValueError):
        repeat_result_shape((3,), -1, 0)


CONCAT_CASES = [
    ([(3,), (2,)], 0),
    ([(2, 3), (2, 3)], 0),
    ([(2, 3), (2, 3)], 1),
    ([(2, 3), (4, 3)], 0),
    ([(2, 3), (2, 5)], 1),
    ([(2, 3), (2, 3), (2, 3)], 0),
    ([(2, 3, 4), (2, 1, 4)], 1),
    ([(2, 3), (2, 3)], -1),
]


@pytest.mark.parametrize("shapes, axis", CONCAT_CASES)
def test_concatenate_matches_numpy(shapes, axis):
    arrays = [np.arange(int(np.prod(s))).reshape(s) + n * 1000 for n, s in enumerate(shapes)]
    expected = np.concatenate(arrays, axis)
    axis_r = axis + len(shapes[0]) if axis < 0 else axis

    assert concatenate_result_shape(shapes, axis) == expected.shape
    for rc in np.ndindex(expected.shape):
        i, sc = concatenate_source(rc, shapes, axis_r)
        assert arrays[i][sc] == expected[rc]


def test_concatenate_rejects_mismatch():
    with pytest.raises(ValueError):
        concatenate_result_shape([(2, 3), (2, 4)], 0)
    with pytest.raises(ValueError):
        concatenate_result_shape([(2, 3), (2, 3, 4)], 0)


STACK_CASES = [
    ([(3,), (3,)], 0),
    ([(3,), (3,)], 1),
    ([(2, 3), (2, 3)], 0),
    ([(2, 3), (2, 3)], 1),
    ([(2, 3), (2, 3)], 2),
    ([(2, 3), (2, 3), (2, 3)], 0),
    ([(2, 3), (2, 3)], -1),
]


@pytest.mark.parametrize("shapes, axis", STACK_CASES)
def test_stack_matches_numpy(shapes, axis):
    arrays = [np.arange(int(np.prod(s))).reshape(s) + n * 1000 for n, s in enumerate(shapes)]
    expected = np.stack(arrays, axis)
    ndim = len(shapes[0])
    axis_r = axis + ndim + 1 if axis < 0 else axis

    assert stack_result_shape(shapes, axis) == expected.shape
    for rc in np.ndindex(expected.shape):
        i, sc = stack_source(rc, axis_r, ndim)
        assert arrays[i][sc] == expected[rc]


def test_stack_rejects_mismatch():
    with pytest.raises(ValueError):
        stack_result_shape([(2, 3), (2, 4)], 0)


BROADCAST_CASES = [
    [(3, 1), (1, 4)],
    [(1,), (5,)],
    [(2, 3, 4), (4,)],
    [(2, 1, 4), (3, 4)],
    [(5,), (1,)],
    [(3, 1, 5), (1, 4, 5)],
    [(1, 1), (2, 3)],
    [(4,), (4,)],
]


@pytest.mark.parametrize("shapes", BROADCAST_CASES)
def test_broadcast_result_shape_matches_numpy(shapes):
    assert broadcast_result_shape(shapes) == np.broadcast_shapes(*shapes)


@pytest.mark.parametrize("shapes", BROADCAST_CASES)
def test_broadcast_source_coord_and_stretched_match_numpy(shapes):
    result = broadcast_result_shape(shapes)
    for s in shapes:
        arr = np.arange(int(np.prod(s))).reshape(s)
        broadcast = np.broadcast_to(arr, result)
        for coord in itertools.product(*[range(n) for n in result]):
            assert arr[broadcast_source_coord(coord, s)] == broadcast[coord]

        extra = len(result) - len(s)
        expected = list(range(extra)) + [
            extra + i for i, size in enumerate(s) if size == 1 and result[extra + i] != 1
        ]
        assert broadcast_stretched_axes(s, result) == expected


def test_broadcast_incompatible_shapes_raise():
    with pytest.raises(ValueError):
        broadcast_result_shape([(3,), (4,)])


def test_public_repeat_take_render_and_report_shape():
    x = np.arange(12).reshape(3, 4)
    assert rt.take(x, [0, 2, 2], axis=1).result_shape == (3, 3)
    assert rt.take(x, [2, 0], axis=0).result_shape == (2, 4)
    assert rt.repeat(x, 2, axis=0).result_shape == (6, 4)
    assert rt.repeat(x, [1, 2, 1, 0], axis=1).result_shape == (3, 4)
    for visual in (
        rt.take(x, [0, 2, 2], axis=1),
        rt.repeat(x, 2, axis=0),
        rt.repeat(x, [1, 2, 1], axis=0),
    ):
        assert visual.svg.startswith("<svg")


def test_public_concatenate_stack_render_and_report_shape():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(100, 106).reshape(2, 3)
    assert rt.concatenate([a, b], 0).result_shape == (4, 3)
    assert rt.concatenate([a, b], 1).result_shape == (2, 6)
    assert rt.stack([a, b], 0).result_shape == (2, 2, 3)
    assert rt.stack([a, b], 1).result_shape == (2, 2, 3)
    for visual in (rt.concatenate([a, b], 0), rt.stack([a, b], 1)):
        assert visual.svg.startswith("<svg")


def test_combine_render_rejects_mismatch():
    with pytest.raises(ValueError):
        rt.concatenate([np.zeros((2, 3)), np.zeros((2, 4))], 0)
    with pytest.raises(ValueError):
        rt.stack([np.zeros((2, 3)), np.zeros((3, 3))], 0)


def test_public_broadcast_renders_and_reports_shape():
    visual = rt.broadcast((3, 1), (1, 4))
    assert visual.result_shape == (3, 4)
    assert visual.svg.startswith("<svg")


def test_public_broadcast_with_arrays():
    a = np.arange(3).reshape(3, 1)
    b = np.arange(4).reshape(1, 4)
    visual = rt.broadcast(a, b)
    assert visual.result_shape == (3, 4)
