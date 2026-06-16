"""Tests for result shape, index formatting, and explanation."""

from rainbow_tensor.indexing import (
    explain_index,
    format_index,
    format_slice,
    result_shape,
)


def test_result_shape_examples():
    assert result_shape((2, 2, 2), (0, slice(None), 1)) == (2,)
    assert result_shape((2, 2, 2), (slice(None), 1, slice(None))) == (2, 2)
    assert result_shape((2, 2, 2), (slice(None), slice(0, 1), slice(None))) == (2, 1, 2)


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
