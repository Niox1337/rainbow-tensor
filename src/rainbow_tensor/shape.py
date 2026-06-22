"""Shape extraction and validation.

This module turns user input into a validated ``tuple[int, ...]`` shape and
generates the display values used when only a shape is provided.
"""

import itertools


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

    The shape must be a non-empty sequence of positive integers. Any rank is
    accepted; the layout nests frames to arbitrary depth.
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

    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(f"shape dimensions must be integers, got {dim!r}")
        if dim <= 0:
            raise ValueError(f"shape dimensions must be positive, got {dim!r}")

    return shape


def coordinates(shape):
    """Yield every coordinate in row-major order for the given shape."""
    return itertools.product(*[range(dim) for dim in shape])


def flat_index(coord, shape):
    """Return the row-major flat index of ``coord`` inside ``shape``."""
    index = 0
    for value, dim in zip(coord, shape):
        index = index * dim + value
    return index


def _check_axis(axis, ndim):
    """Resolve a possibly negative axis and bounds check it."""
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
