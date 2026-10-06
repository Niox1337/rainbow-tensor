"""Shape contracts and first-occurrence rules for real scalar extrema."""

from numbers import Real

from .reductions import (
    normalize_reduction_axes,
    reduce_result_shape,
    reduce_term_count,
    validate_keepdims,
)

EXTREMA = frozenset(("min", "max", "argmin", "argmax"))


def extrema_spec(shape, axis, keepdims, operation):
    """Validate an extremum before recording a recipe or reading any values.

    Value extrema accept an integer axis, a tuple of axes, or None. Index
    extrema accept only one integer axis or None, whose result is a flattened
    row-major position. A scalar accepts integer axis zero or minus one.
    Empty reduced groups have no identity and are rejected even when another
    dimension is empty. Empty surviving outputs remain valid.
    """
    if operation not in EXTREMA:
        raise ValueError(f"unsupported extremum operation: {operation!r}")
    if operation in {"argmin", "argmax"} and isinstance(axis, tuple):
        raise TypeError("argmin and argmax accept one integer axis or None")
    axes = normalize_reduction_axes(axis, len(shape), allow_scalar_axis=True)
    keepdims = validate_keepdims(keepdims)
    count = reduce_term_count(shape, axes)
    if count == 0:
        raise ValueError(f"{operation} cannot reduce an empty group without an identity")
    return axes, keepdims, reduce_result_shape(shape, axes, keepdims=keepdims), count


def extremum_choice(operation, candidates):
    """Return the output, chosen row-major position, and explicit selection reason.

    Candidates must be real scalar values. Ties retain the first occurrence,
    including the sign of a tied zero. Any NaN wins over real values, with the
    first NaN selected. This makes min/max NaN propagation and argmin/argmax
    positions consistent without claiming backend dtype or reduction kernels.
    The iterator is consumed once and only the current winner is retained.
    """
    if operation not in EXTREMA:
        raise ValueError(f"unsupported extremum operation: {operation!r}")
    minimum = operation in {"min", "argmin"}
    chosen_position = None
    chosen = None
    chosen_nan = False
    for position, value in enumerate(candidates):
        if not isinstance(value, Real):
            raise TypeError("extrema require real scalar values")
        is_nan = bool(value != value)
        if chosen_position is None or (not chosen_nan and (
            is_nan or (value < chosen if minimum else value > chosen)
        )):
            chosen_position, chosen, chosen_nan = position, value, is_nan
    if chosen_position is None:
        raise ValueError(f"{operation} cannot reduce an empty group without an identity")
    reason = "first_nan" if chosen_nan else ("first_min" if minimum else "first_max")
    output = chosen_position if operation in {"argmin", "argmax"} else chosen
    return output, chosen_position, reason
