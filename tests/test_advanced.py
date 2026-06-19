"""Advanced indexing: boolean masks and integer index arrays.

Every supported case is cross-checked against NumPy, so the selected
coordinates and the result shape are guaranteed to match real indexing.
"""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.indexing import advanced_index, is_advanced

# (shape, index) cases that must match NumPy exactly.
CASES = [
    ((2, 3), np.array([[True, False, True], [False, True, False]])),  # boolean mask
    ((4,), ([0, 2, 3],)),  # 1D integer array
    ((3, 4), ([0, 2], [1, 3])),  # paired arrays gather points
    ((3, 4), ([0, 2, 2], [1, 3, 0])),  # repeated coordinates
    ((3, 4), ([0, 2], [1])),  # length-1 array broadcasts
    ((3, 4), ([0, 2], slice(1, 3))),  # array contiguous with a slice
    ((3, 4, 5), ([0, 2], [1, 3], slice(None))),  # adjacent arrays + slice
    ((3, 4, 5), ([0, 2], 1, [0, 4])),  # an int between arrays does not separate
    ((4,), ([-1, -2],)),  # negative array values
    ((3, 4, 5), ([0, 2], slice(None), [1, 3])),  # slice between arrays: front moved
    ((2, 3, 4, 5), (slice(None), [0, 1], slice(None), [2, 4])),  # front moved, slices kept
    ((4,), (np.array([[0, 2], [3, 1]]),)),  # 2D index array, result keeps its shape
    ((3, 4), ([[0, 2], [1, 2]], [[1, 3], [0, 2]])),  # paired 2D arrays gather a grid
    ((3, 4), ([[0], [2]], [1, 3])),  # (2,1) and (2,) broadcast to (2, 2)
    ((3, 4, 5), ([[0, 2]], slice(None), [[1, 3]])),  # 2D arrays, slice moves block front
    ((5,), (np.array([[0, 1, 2], [2, 3, 4]]),)),  # 2x3 block on a vector
]


@pytest.mark.parametrize("shape, index", CASES)
def test_matches_numpy(shape, index):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    selected, result, _ = advanced_index(shape, index)

    ref = np.zeros(shape, bool)
    ref[index] = True
    expected = set(map(tuple, np.argwhere(ref)))

    assert set(selected) == expected
    assert result == x[index].shape


def test_index_renders_mask_and_arrays():
    x = np.arange(6).reshape(2, 3)
    mask = x % 2 == 0
    assert rt.index(x, mask).result_shape == (3,)
    assert rt.index(x, ([0, 1], [2, 0])).result_shape == (2,)
    # A 2D index array keeps its own shape in the result.
    assert rt.index(x, ([[0, 1], [1, 0]], [[2, 0], [1, 2]])).result_shape == (2, 2)


def test_detects_advanced():
    assert is_advanced(([0, 1],), (3,))
    assert is_advanced(np.array([True, False, True]), (3,))
    assert not is_advanced((0, slice(None)), (3, 3))


def test_refuses_unsupported():
    # mask shape must match the tensor
    with pytest.raises(IndexError):
        advanced_index((2, 3), np.array([True, False, True]))
    # newaxis with advanced indexing
    with pytest.raises(IndexError):
        advanced_index((3, 4), (None, [0, 1]))
