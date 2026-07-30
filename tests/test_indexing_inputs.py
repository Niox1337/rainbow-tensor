"""Index inputs preserve integer protocols, rectangularity and empty masks."""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.indexing import advanced_index, result_shape, validate_index
from rainbow_tensor.shape import validate_shape


class IndexValue:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class CountingArray:
    shape = (4, 3)

    def __init__(self):
        self.reads = 0

    def __getitem__(self, coordinate):
        self.reads += 1
        return 0


@pytest.mark.parametrize("value", [np.int64(2), np.uint32(2), IndexValue(2)])
def test_shape_dimensions_follow_the_integer_index_protocol(value):
    shape = validate_shape((value, 3))

    assert shape == (2, 3)
    assert all(type(dimension) is int for dimension in shape)
    assert rt.shape((value, 3)).shape == (2, 3)


@pytest.mark.parametrize("value", [np.int64(-1), np.uint32(1), IndexValue(1)])
def test_basic_and_advanced_scalar_indices_use_the_same_integer_protocol(value):
    array = np.arange(12).reshape(3, 4)
    basic = (value, slice(None))
    advanced = (value, [0, 2])

    tokens = validate_index(basic, array.shape)
    assert type(tokens[0]) is int
    assert result_shape(array.shape, basic) == array[basic].shape
    assert rt.index(array, basic).result_shape == array[basic].shape
    assert rt.index(array, advanced).result_shape == array[advanced].shape


@pytest.mark.parametrize("value", [True, np.bool_(True), 2.0, np.float64(2.0)])
def test_shapes_and_basic_indices_still_reject_boolean_and_float_scalars(value):
    with pytest.raises(TypeError):
        validate_shape((value, 3))
    with pytest.raises(TypeError):
        validate_index((value, slice(None)), (4, 3))


@pytest.mark.parametrize(
    "indices",
    [
        [[0], [1, 2]],
        [[0, 1], [2]],
        [[0], 1],
        [[], [0]],
        [[[0]], [[1, 2]]],
    ],
)
def test_ragged_index_lists_are_rejected_before_source_values_are_read(indices):
    array = CountingArray()
    with pytest.raises(ValueError):
        np.zeros(array.shape)[indices, :]
    with pytest.raises(ValueError, match="ragged|rectangular"):
        rt.index(array, (indices, slice(None)))

    assert array.reads == 0


@pytest.mark.parametrize(
    "shape, index",
    [
        ((3, 4), (np.empty(0, dtype=bool),)),
        ((3, 4), (slice(None), np.empty(0, dtype=bool))),
        ((3, 4), (np.empty((0, 4), dtype=bool), Ellipsis)),
        ((3, 4), (np.empty((3, 0), dtype=bool),)),
        ((3, 4), (np.empty((0, 0), dtype=bool),)),
        ((3, 4, 2), (np.empty((0, 4), dtype=bool),)),
        ((3, 4), (np.empty(0, dtype=bool), [1])),
    ],
)
def test_empty_boolean_masks_keep_their_axis_consumption(shape, index):
    array = np.zeros(shape)
    selected, result, _ = advanced_index(shape, index)

    assert result == array[index].shape
    assert selected == []
    assert rt.index(array, index).result_shape == array[index].shape


@pytest.mark.parametrize("mask_shape", [(0, 4), (3, 0), (0, 0)])
def test_standalone_empty_full_rank_boolean_masks_match_numpy(mask_shape):
    array = np.zeros((3, 4))
    mask = np.empty(mask_shape, dtype=bool)

    visual = rt.index(array, mask)

    assert visual.result_shape == array[mask].shape == (0,)
    assert visual.selected == []


@pytest.mark.parametrize("mask_shape", [(0, 5), (2, 0)])
def test_empty_masks_still_reject_nonempty_dimension_mismatches(mask_shape):
    mask = np.empty(mask_shape, dtype=bool)
    with pytest.raises(IndexError):
        np.zeros((3, 4))[mask]
    with pytest.raises(IndexError):
        advanced_index((3, 4), (mask,))
    with pytest.raises(IndexError):
        advanced_index((3, 4), mask)


def test_empty_boolean_mask_still_must_broadcast_with_integer_arrays():
    index = (np.empty(0, dtype=bool), [1, 2])
    with pytest.raises(IndexError):
        np.zeros((3, 4))[index]
    with pytest.raises(IndexError):
        advanced_index((3, 4), index)


@pytest.mark.parametrize("dtype", [float, complex, object])
def test_empty_nonboolean_standalone_arrays_are_not_masks(dtype):
    with pytest.raises((TypeError, IndexError)):
        rt.index((3, 4), np.empty((0, 4), dtype=dtype))


@pytest.mark.parametrize(
    "index",
    [(), (1,), (slice(None),), (1, None), (None,), (np.int64(-1),), (Ellipsis,)],
)
def test_basic_indexing_implicitly_keeps_omitted_trailing_axes(index):
    array = np.arange(24).reshape(2, 3, 4)
    visual = rt.index(array, index)

    assert visual.result_shape == array[index].shape
    assert {array[c] for c in visual.selected} == set(array[index].ravel())


@pytest.mark.parametrize(
    "index",
    [([0, 2],), (1, [0, 2]), ([0, 2], Ellipsis), (np.array([True, False, True]),)],
)
def test_advanced_indexing_implicitly_keeps_omitted_trailing_axes(index):
    array = np.arange(60).reshape(3, 4, 5)
    visual = rt.index(array, index)

    assert visual.result_shape == array[index].shape
    assert {array[c] for c in visual.selected} == set(array[index].ravel())


def test_implicit_trailing_axes_do_not_allow_too_many_indices():
    with pytest.raises(IndexError):
        validate_index((0, 0, 0), (2, 3))
    with pytest.raises(IndexError):
        advanced_index((2, 3), ([0], 0, 0))
