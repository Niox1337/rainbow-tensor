"""Tests for shape extraction and validation."""

import pytest

from rainbow_tensor.shape import (
    extract_shape,
    flat_index,
    format_shape,
    format_shape_label,
    generate_values,
    validate_shape,
)


class FakeArray:
    def __init__(self, shape):
        self.shape = shape


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 2)])
def test_valid_shapes(shape):
    assert validate_shape(shape) == shape


def test_extract_shape_from_tuple():
    assert extract_shape((2, 2, 2)) == (2, 2, 2)


def test_extract_shape_from_array_like():
    assert extract_shape(FakeArray((2, 2, 2))) == (2, 2, 2)


def test_array_like_with_invalid_shape_uses_same_validation():
    with pytest.raises(ValueError):
        extract_shape(FakeArray((2, 0)))


def test_reject_empty_shape():
    with pytest.raises(ValueError):
        validate_shape(())


def test_reject_zero_dimension():
    with pytest.raises(ValueError):
        validate_shape((2, 0))


def test_reject_negative_dimension():
    with pytest.raises(ValueError):
        validate_shape((2, -1))


def test_reject_non_integer_dimension():
    with pytest.raises(TypeError):
        validate_shape((2, "3"))


def test_reject_boolean_dimension():
    with pytest.raises(TypeError):
        validate_shape((2, True))


def test_reject_four_dimensions():
    with pytest.raises(ValueError):
        validate_shape((2, 2, 2, 2))


def test_generate_values_for_three_dimensions():
    values = generate_values((2, 2, 2))
    assert values[(0, 0, 0)] == 0
    assert values[(0, 0, 1)] == 1
    assert values[(0, 1, 0)] == 2
    assert values[(0, 1, 1)] == 3
    assert values[(1, 0, 0)] == 4
    assert values[(1, 1, 1)] == 7


def test_flat_index_row_major():
    assert flat_index((1, 0, 1), (2, 2, 2)) == 5


def test_format_shape_uses_tuple_form():
    assert format_shape((2,)) == "(2,)"
    assert format_shape((2, 2, 2)) == "(2, 2, 2)"


def test_format_shape_label_has_no_trailing_comma():
    assert format_shape_label((3,)) == "(3)"
    assert format_shape_label((2, 2, 2)) == "(2, 2, 2)"
