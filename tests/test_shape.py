"""Tests for shape extraction and validation."""

import pytest

from rainbow_tensor.shape import extract_shape, validate_shape


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
