"""Elementwise arithmetic preserves ordered operands and bounded scalar reads."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rainbow_tensor.ops.elementwise import evaluate_binary, normalize_operand


@pytest.mark.parametrize("operation,left,right,expected", [
    ("add", 7, 3, 10),
    ("subtract", 7, 3, 4),
    ("subtract", 3, 7, -4),
    ("multiply", 7, 3, 21),
    ("divide", 7, 2, 3.5),
    ("divide", 2, 8, 0.25),
    ("add", 2 + 3j, 4 - 2j, 6 + 1j),
    ("multiply", 10**100, 10**100, 10**200),
])
def test_binary_scalar_rules_preserve_order_and_python_precision(operation, left, right, expected):
    assert evaluate_binary(operation, left, right) == expected


@pytest.mark.parametrize("numerator", [0, 1, -1, 3.5, 2 + 1j])
@pytest.mark.parametrize("denominator", [0, 0.0, -0.0, 0j])
def test_binary_division_rejects_every_zero_denominator(numerator, denominator):
    with pytest.raises(ZeroDivisionError):
        evaluate_binary("divide", numerator, denominator)


def test_binary_rules_reject_an_unknown_operation():
    with pytest.raises(ValueError, match="unsupported binary operation"):
        evaluate_binary("power", 2, 3)


@pytest.mark.parametrize("literal", [0, 2, -3, 2.5, 1 + 2j, True])
def test_numeric_literals_are_immutable_scalar_operands(literal):
    operand = normalize_operand(literal)
    assert operand.shape == ()
    assert operand[()] == literal
    assert type(operand[()]) is type(literal)
    with pytest.raises(IndexError, match="coordinate"):
        operand[(0,)]
    with pytest.raises(FrozenInstanceError):
        operand.value = 99


@pytest.mark.parametrize("operand", [(2, 3), (), np.array([2, 3]), np.array(4), np.int64(5)])
def test_normalization_preserves_arrays_and_shape_placeholders(operand):
    assert normalize_operand(operand) is operand


def test_operand_normalization_never_reads_array_values():
    class UnreadableArray:
        shape = (3,)

        def __getitem__(self, coordinate):
            raise AssertionError("normalizing an operand must not read its values")

    operand = UnreadableArray()
    assert normalize_operand(operand) is operand
