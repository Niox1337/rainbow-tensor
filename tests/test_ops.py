"""Reshape, transpose, and axis reductions.

Every result shape and coordinate mapping is cross-checked against NumPy, so
the figures the package draws describe the same operation NumPy performs.
"""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.ops import (
    reduce_result_shape,
    reduce_source_coords,
    reshape_result_shape,
    reshape_source_coord,
    transpose_axes,
    transpose_result_shape,
    transpose_source_coord,
)

RESHAPE_CASES = [
    ((6,), (2, 3)),
    ((2, 3), (3, 2)),
    ((2, 3), (6,)),
    ((2, 3, 4), (4, 6)),
    ((2, 3, 4), (-1, 4)),
    ((12,), (2, -1, 3)),
]


@pytest.mark.parametrize("old, new", RESHAPE_CASES)
def test_reshape_matches_numpy(old, new):
    x = np.arange(int(np.prod(old))).reshape(old)
    expected = x.reshape(new)
    resolved = reshape_result_shape(old, new)

    assert resolved == expected.shape
    for rc in np.ndindex(resolved):
        assert expected[rc] == x[reshape_source_coord(rc, old, resolved)]


def test_reshape_rejects_bad_size():
    with pytest.raises(ValueError):
        reshape_result_shape((2, 3), (4, 2))
    with pytest.raises(ValueError):
        reshape_result_shape((2, 3), (-1, -1))


TRANSPOSE_CASES = [
    ((2, 3), None),
    ((2, 3), (1, 0)),
    ((2, 3, 4), None),
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4, 5), (0, 2, 1, 3)),
]


@pytest.mark.parametrize("shape, axes", TRANSPOSE_CASES)
def test_transpose_matches_numpy(shape, axes):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    expected = x.transpose() if axes is None else x.transpose(axes)
    perm = transpose_axes(len(shape), axes)

    assert transpose_result_shape(shape, perm) == expected.shape
    for rc in np.ndindex(expected.shape):
        assert expected[rc] == x[transpose_source_coord(rc, perm)]


def test_transpose_rejects_non_permutation():
    with pytest.raises(ValueError):
        transpose_axes(3, (0, 1))
    with pytest.raises(ValueError):
        transpose_axes(2, (0, 0))


REDUCE_CASES = [
    ((4,), 0),
    ((3, 4), 0),
    ((3, 4), 1),
    ((2, 3, 4), 1),
    ((2, 3, 4), -1),
]


@pytest.mark.parametrize("shape, axis", REDUCE_CASES)
def test_reduce_matches_numpy(shape, axis):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    res = reduce_result_shape(shape, axis)

    assert res == np.sum(x, axis).shape
    for rc in np.ndindex(res):
        vals = [x[sc] for sc in reduce_source_coords(rc, shape, axis)]
        assert sum(vals) == np.sum(x, axis)[rc]
        assert np.isclose(sum(vals) / len(vals), np.mean(x, axis)[rc])


def test_reduce_rejects_bad_axis():
    with pytest.raises(ValueError):
        reduce_result_shape((2, 3), 2)


def test_public_functions_render_and_report_shape():
    x = np.arange(12).reshape(3, 4)
    assert rt.reshape(x, (2, 6)).result_shape == (2, 6)
    assert rt.transpose(x).result_shape == (4, 3)
    assert rt.sum(x, 0).result_shape == (4,)
    assert rt.mean(x, 1).result_shape == (3,)
    for visual in (rt.reshape(x, (2, 6)), rt.transpose(x), rt.sum(x, 0), rt.mean(x, 1)):
        assert visual.svg.startswith("<svg")


def test_reduce_to_scalar_renders():
    x = np.arange(4)
    visual = rt.sum(x, 0)
    assert visual.result_shape == ()
    assert visual.svg.startswith("<svg")
