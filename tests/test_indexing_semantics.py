"""NumPy comparisons for advanced-index ordering and integer validation."""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.indexing import advanced_index


@pytest.mark.parametrize(
    "shape, index",
    [
        ((3, 4, 5), (0, slice(None), [0, 1])),
        ((3, 4, 5), ([0, 1], slice(None), 0)),
        ((6, 3, 4, 5), (slice(None), 0, slice(None), [0, 1])),
        ((6, 3, 4, 5), (slice(None), [0, 1], slice(None), 0)),
        ((6, 3, 4, 5), (slice(None), 0, [0, 1], slice(None))),
        ((6, 3, 4, 5), (slice(None), [0, 1], 0, slice(None))),
        ((6, 3, 4, 5), (slice(None), [0, 1], Ellipsis, [0, 1], slice(None))),
        ((6, 3, 4, 5), (slice(None), 0, Ellipsis, [0, 1], slice(None))),
        ((6, 3, 4, 5), (slice(None), [0, 1], Ellipsis, 0, slice(None))),
        ((6, 3, 4, 5), (slice(None), [0, 1], Ellipsis, [0, 1])),
        ((3, 4, 5), (np.int64(-1), slice(None), np.array([0, 1], dtype=np.int32))),
        ((3, 4, 5), (np.uint64(0), slice(None), [0, 1])),
        ((3, 4, 5), (np.array(0), slice(None), [0, 1])),
        ((3, 4), (np.array([], dtype=np.int64), slice(None))),
        ((3, 4), ([], slice(None))),
        ((3, 4), ([True, 2], slice(None))),
        ((3,), ([[True, False], [1, 0]],)),
    ],
)
def test_advanced_shape_and_selection_match_numpy(shape, index):
    """Scalar positions and even empty ellipses affect advanced axis order."""
    array = np.arange(np.prod(shape)).reshape(shape)
    selected, result, _ = advanced_index(shape, index)

    expected_selection = np.zeros(shape, dtype=bool)
    expected_selection[index] = True

    assert result == array[index].shape
    assert set(selected) == set(map(tuple, np.argwhere(expected_selection)))


@pytest.mark.parametrize(
    "indices",
    [
        [0.8, 1.8],
        [0.0, 1.0],
        [[0, 1.5]],
        [True, 1.5],
        [[True, False], [0, 1.5]],
        np.array([0.0, 1.0]),
        np.empty(0, dtype=float),
        np.empty((0, 2), dtype=float),
        np.array([0, 1], dtype=complex),
        np.array([0, 1], dtype=object),
        np.array(["0", "1"]),
    ],
)
def test_noninteger_index_arrays_are_rejected_like_numpy(indices):
    """Integer-looking values must not make an invalid index dtype valid."""
    with pytest.raises(IndexError):
        np.arange(4)[indices]
    with pytest.raises(IndexError, match="integer"):
        advanced_index((4,), (indices,))


def test_public_index_reports_numpy_order_for_separated_scalar_and_array():
    array = np.arange(60).reshape(3, 4, 5)
    index = (0, slice(None), [0, 1])

    assert rt.index(array, index).result_shape == array[index].shape
