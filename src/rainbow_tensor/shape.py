"""Shape extraction and validation.

This module turns user input into a validated ``tuple[int, ...]`` shape and
generates the display values used when only a shape is provided.
"""

SUPPORTED_NDIM = (1, 2, 3)


def validate_shape(shape):
    """Validate a shape and return it as a ``tuple[int, ...]``.

    The shape must be a non-empty sequence of positive integers with a
    supported number of dimensions (1D, 2D, or 3D in this version).
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

    if len(shape) not in SUPPORTED_NDIM:
        raise ValueError(
            f"only 1D, 2D, and 3D tensors are supported in this version, "
            f"got {len(shape)} dimensions"
        )

    return shape
