"""Tests for index validation, slice expansion, and selection."""

import pytest

from rainbow_tensor.indexing import (
    expand_slice,
    selected_coordinates,
    validate_index,
)


def test_valid_indices():
    assert validate_index((0,), (2,)) == (0,)
    assert validate_index((slice(None),), (2,)) == (slice(None),)
    assert validate_index((0, slice(None), 1), (2, 2, 2)) == (0, slice(None), 1)


@pytest.mark.parametrize("index", ["0, :, 1", [0, slice(None), 1]])
def test_reject_non_tuple_index(index):
    with pytest.raises(TypeError):
        validate_index(index, (2, 2, 2))


def test_reject_none_entry():
    with pytest.raises(TypeError):
        validate_index((0, None, 1), (2, 2, 2))


def test_reject_wrong_length():
    with pytest.raises(ValueError):
        validate_index((0, slice(None)), (2, 2, 2))


def test_reject_out_of_range_integer():
    with pytest.raises(IndexError):
        validate_index((2, slice(None), 1), (2, 2, 2))
    with pytest.raises(IndexError):
        validate_index((0, slice(None), 2), (2, 2, 2))


def test_reject_negative_integer():
    with pytest.raises(IndexError):
        validate_index((-1, slice(None), 1), (2, 2, 2))


def test_expand_slice():
    assert expand_slice(slice(None), 4) == [0, 1, 2, 3]
    assert expand_slice(slice(1, 3), 4) == [1, 2]
    assert expand_slice(slice(0, 4, 2), 4) == [0, 2]


def test_selected_coordinates():
    selected = selected_coordinates((2, 2, 2), (0, slice(None), 1))
    assert selected == [(0, 0, 1), (0, 1, 1)]


def test_selected_coordinates_one_dimension():
    assert selected_coordinates((3,), (slice(1, 3),)) == [(1,), (2,)]
