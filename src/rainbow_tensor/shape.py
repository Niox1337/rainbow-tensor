"""Shape extraction and validation.

This module turns user input into a validated ``tuple[int, ...]`` shape and
generates the display values used when only a shape is provided.
"""

import itertools
from operator import index as integer_index


def extract_shape(obj):
    """Extract a validated shape from a tuple or an array-like object.

    A tuple is treated as a literal shape. Any other object is expected to
    expose a ``.shape`` attribute (for example a NumPy array). No tensor
    library is imported, so this works for any object with ``.shape``.
    """
    if isinstance(obj, tuple):
        raw = obj
    elif hasattr(obj, "shape"):
        raw = obj.shape
    else:
        raw = obj
    return validate_shape(raw)


def validate_shape(shape):
    """Validate a shape and return it as a ``tuple[int, ...]``.

    The shape must be a non-empty sequence of positive integers. Dimensions
    follow Python's integer index protocol and are normalized to Python ints.
    Boolean and floating-point dimensions remain invalid. Any rank is accepted.
    """
    if not isinstance(shape, tuple):
        try:
            shape = tuple(shape)
        except TypeError as exc:
            raise TypeError(
                "shape must be a tuple of integers or an object with a .shape attribute"
            ) from exc

    if len(shape) == 0:
        raise ValueError("shape must have at least one dimension, got an empty shape")

    normalized = []
    for dim in shape:
        dim = _as_integer(dim, "shape dimensions")
        if dim <= 0:
            raise ValueError(f"shape dimensions must be positive, got {dim!r}")
        normalized.append(dim)

    return tuple(normalized)


def coordinates(shape):
    """Yield every coordinate in row-major order for the given shape."""
    return itertools.product(*[range(dim) for dim in shape])


def flat_index(coord, shape):
    """Return the row-major flat index of ``coord`` inside ``shape``."""
    index = 0
    for value, dim in zip(coord, shape):
        index = index * dim + value
    return index


def _as_integer(value, name):
    """Return an index-protocol integer without accepting booleans or floats.

    NumPy boolean scalars are rejected without importing an array backend.
    Unlike ``int``, the index protocol never truncates floating-point values.
    """
    if isinstance(value, bool) or type(value).__name__.startswith("bool"):
        raise TypeError(f"{name} must be integers, got {value!r}")
    try:
        return integer_index(value)
    except TypeError:
        raise TypeError(f"{name} must be integers, got {value!r}") from None


def _integer_or_sequence(value, name):
    """Normalize an integer or consume an iterable of integers exactly once."""
    try:
        return _as_integer(value, name)
    except TypeError:
        try:
            values = iter(value)
        except TypeError:
            raise TypeError(f"{name} must be an integer or an iterable of integers") from None
        return tuple(_as_integer(item, name) for item in values)


def _check_axis(axis, ndim):
    """Normalize an integer-protocol axis, resolve negatives and check bounds."""
    axis = _as_integer(axis, "axis")
    resolved = axis + ndim if axis < 0 else axis
    if not 0 <= resolved < ndim:
        raise ValueError(f"axis {axis} is out of range for a rank {ndim} tensor")
    return resolved


def generate_values(shape):
    """Map every coordinate of ``shape`` to a sequential value from 0.

    For shape ``(2, 2, 2)`` this produces the values 0 through 7 in
    row-major order.
    """
    return {coord: flat_index(coord, shape) for coord in coordinates(shape)}


def format_shape(shape):
    """Format a shape the way Python prints a tuple, for example ``(2,)``."""
    return str(tuple(shape))


def format_shape_label(shape):
    """Format a shape for an on-screen label without a trailing comma.

    For shape ``(3,)`` this returns ``(3)`` rather than ``(3,)``.
    """
    return "(" + ", ".join(str(dim) for dim in shape) + ")"
