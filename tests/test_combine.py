"""Concatenate and stack.

Every result shape and coordinate mapping is cross-checked against NumPy, so
the operands and the seam the package draws describe the same join NumPy
performs.
"""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.ops import (
    concatenate_result_shape,
    concatenate_source,
    stack_result_shape,
    stack_source,
)

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


def test_public_functions_render_and_report_shape():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(100, 106).reshape(2, 3)
    assert rt.concatenate([a, b], 0).result_shape == (4, 3)
    assert rt.concatenate([a, b], 1).result_shape == (2, 6)
    assert rt.stack([a, b], 0).result_shape == (2, 2, 3)
    assert rt.stack([a, b], 1).result_shape == (2, 2, 3)
    for visual in (rt.concatenate([a, b], 0), rt.stack([a, b], 1)):
        assert visual.svg.startswith("<svg")


def test_render_rejects_mismatch():
    with pytest.raises(ValueError):
        rt.concatenate([np.zeros((2, 3)), np.zeros((2, 4))], 0)
    with pytest.raises(ValueError):
        rt.stack([np.zeros((2, 3)), np.zeros((3, 3))], 0)
