"""Tests for result shape, index formatting, and explanation."""

import pytest

from rainbow_tensor.indexing import (
    explain_index,
    format_index,
    format_slice,
    result_shape,
    selected_coordinates,
)

np = pytest.importorskip("numpy")


def test_result_shape_examples():
    assert result_shape((2, 2, 2), (0, slice(None), 1)) == (2,)
    assert result_shape((2, 2, 2), (slice(None), 1, slice(None))) == (2, 2)
    assert result_shape((2, 2, 2), (slice(None), slice(0, 1), slice(None))) == (2, 1, 2)


# Cross-check result shape and selection against NumPy for the new cases.
@pytest.mark.parametrize(
    "shape, index",
    [
        ((4,), (-1,)),
        ((4,), (slice(None, None, -1),)),
        ((5,), (slice(4, 1, -1),)),
        ((2, 3, 2), (0, Ellipsis, 1)),
        ((2, 3), (Ellipsis,)),
        ((3,), (None, slice(None))),
        ((2, 3), (slice(None), None, -1)),
    ],
)
def test_matches_numpy(shape, index):
    arr = np.arange(int(np.prod(shape))).reshape(shape)
    assert result_shape(shape, index) == arr[index].shape
    # Every selected source coordinate must read a value NumPy also selects.
    picked = {arr[coord] for coord in selected_coordinates(shape, index)}
    assert picked == set(np.asarray(arr[index]).ravel().tolist())


def test_format_index_special_tokens():
    assert format_index((-1, Ellipsis, None)) == "-1, ..., None"
    assert format_index((slice(None, None, -1),)) == "::-1"


def test_format_slice():
    assert format_slice(slice(None)) == ":"
    assert format_slice(slice(1, 3)) == "1:3"
    assert format_slice(slice(0, 4, 2)) == "0:4:2"


def test_format_index():
    assert format_index((0, slice(None), 1)) == "0, :, 1"
    assert format_index((slice(None), slice(1, 3))) == ":, 1:3"
    assert format_index((slice(0, 4, 2),)) == "0:4:2"


def test_explain_index():
    lines = explain_index((2, 2, 2), (0, slice(None), 1))
    assert lines[0] == "Original shape: (2, 2, 2)"
    assert lines[1] == "Index: 0, :, 1"
    assert lines[2] == "Result shape: (2,)"
    assert lines[3] == "Axis 0 is removed because integer index 0 is used."
    assert lines[4] == "Axis 1 is kept because slice : is used."
    assert lines[5] == "Axis 2 is removed because integer index 1 is used."


def test_explain_newaxis_names_position():
    lines = explain_index((3,), (None, slice(None), None))
    assert "result position 0" in lines[3]
    assert "result position 2" in lines[5]
