"""Scalar rules shared by elementwise views and recorded operation flows."""

from dataclasses import dataclass
from numbers import Number
from operator import add, mul, sub, truediv

BINARY_SYMBOLS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
_BINARY_OPERATORS = {"add": add, "subtract": sub, "multiply": mul, "divide": truediv}


@dataclass(frozen=True, slots=True)
class _ScalarOperand:
    """Expose a numeric literal as one scalar cell without creating an array."""

    value: Number

    @property
    def shape(self):
        """A scalar has one element and no coordinate axes."""
        return ()

    def __getitem__(self, coordinate):
        """Read the literal only at its scalar coordinate."""
        if coordinate != ():
            raise IndexError("a scalar operand only has coordinate ()")
        return self.value


def normalize_operand(obj):
    """Wrap numeric literals while preserving arrays and shape-only operands.

    Tuple operands keep their existing meaning as logical shapes. Array-like
    objects, including backend scalars with a shape, retain their identity and
    are not read or copied here.
    """
    if isinstance(obj, Number) and not hasattr(obj, "shape"):
        return _ScalarOperand(obj)
    return obj


def evaluate_binary(operation, left, right):
    """Apply one ordered Python scalar operation without backend dispatch.

    Division uses true division. A zero denominator raises ZeroDivisionError,
    including zero divided by zero, rather than adopting a backend's infinity
    or NaN rules. Integer arithmetic retains Python's unbounded precision.
    """
    try:
        function = _BINARY_OPERATORS[operation]
    except KeyError:
        raise ValueError(f"unsupported binary operation: {operation!r}") from None
    return function(left, right)
