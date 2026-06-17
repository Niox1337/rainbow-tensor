"""Index validation, slice expansion, and selection computation.

This module validates indexing expressions, expands slices into integer
positions, computes the selected coordinates, and computes the result shape
after indexing. It supports negative integer indices, negative slice bounds
and steps, ``Ellipsis`` (``...``), and ``None`` (newaxis).
"""

import itertools

from .shape import format_shape


def _resolve_int(entry, size, axis):
    """Resolve a possibly negative integer index against an axis size."""
    resolved = entry + size if entry < 0 else entry
    if not 0 <= resolved < size:
        raise IndexError(
            f"axis {axis}: index {entry} is out of range for size {size}"
        )
    return resolved


def validate_index(index, shape):
    """Validate and normalize an index tuple against a tensor shape.

    The index must be a tuple. Entries may be integers (negative allowed),
    slices, a single ``Ellipsis``, or ``None`` (newaxis). The returned tuple
    has the ellipsis expanded into full slices and integers resolved to their
    non-negative position, with ``None`` kept in place to mark an inserted
    size 1 axis. The axis consuming entries (integers and slices) must cover
    the tensor rank exactly.
    """
    if not isinstance(index, tuple):
        raise TypeError(
            f"index must be a tuple, got {type(index).__name__}. "
            f"For a single axis use a one element tuple such as (0,)"
        )

    if sum(e is Ellipsis for e in index) > 1:
        raise IndexError("an index can have at most one ellipsis '...'")

    consuming = sum(
        isinstance(e, slice) or (isinstance(e, int) and not isinstance(e, bool))
        for e in index
    )
    if consuming > len(shape):
        raise IndexError(
            f"too many indices for tensor of rank {len(shape)}: "
            f"{consuming} axes were indexed"
        )
    fill = [slice(None)] * (len(shape) - consuming)

    tokens = []
    axis = 0
    for entry in index:
        if entry is Ellipsis:
            tokens.extend(fill)
            axis += len(fill)
        elif entry is None:
            tokens.append(None)
        elif isinstance(entry, bool):
            raise TypeError(f"axis {axis}: boolean indices are not supported")
        elif isinstance(entry, int):
            tokens.append(_resolve_int(entry, shape[axis], axis))
            axis += 1
        elif isinstance(entry, slice):
            tokens.append(entry)
            axis += 1
        else:
            raise TypeError(
                f"axis {axis}: unsupported index entry {entry!r}. "
                f"Only integers, slices, Ellipsis, and None are supported"
            )

    if axis != len(shape):
        raise ValueError(
            f"index covers {axis} axes but tensor has rank {len(shape)}. "
            f"Use '...' to fill the remaining axes"
        )
    return tuple(tokens)


def expand_slice(sl, size):
    """Expand a slice into the list of integer positions it selects.

    Negative bounds and negative steps are handled by ``slice.indices``.
    """
    return list(range(*sl.indices(size)))


def selected_coordinates(shape, index):
    """Compute every coordinate selected by ``index`` in row-major order.

    ``None`` entries insert a new axis in the result and select nothing in the
    source tensor, so they are skipped here.
    """
    tokens = validate_index(index, shape)
    axes = []
    axis = 0
    for tok in tokens:
        if tok is None:
            continue
        size = shape[axis]
        axis += 1
        if isinstance(tok, int):
            axes.append([tok])
        else:
            axes.append(expand_slice(tok, size))
    return list(itertools.product(*axes))


def result_shape(shape, index):
    """Compute the result shape after indexing.

    Integer entries remove their axis. Slice entries keep their axis with a
    size equal to the number of selected positions. ``None`` inserts a size 1
    axis where it appears.
    """
    tokens = validate_index(index, shape)
    out = []
    axis = 0
    for tok in tokens:
        if tok is None:
            out.append(1)
            continue
        size = shape[axis]
        axis += 1
        if isinstance(tok, slice):
            out.append(len(expand_slice(tok, size)))
    return tuple(out)


def format_slice(sl):
    """Format a slice the way it would appear in indexing notation."""
    if sl.start is None and sl.stop is None and sl.step is None:
        return ":"
    start = "" if sl.start is None else str(sl.start)
    stop = "" if sl.stop is None else str(sl.stop)
    if sl.step is None:
        return f"{start}:{stop}"
    return f"{start}:{stop}:{sl.step}"


def format_token(entry):
    """Format a single raw index entry as it would appear in notation."""
    if isinstance(entry, slice):
        return format_slice(entry)
    if entry is None:
        return "None"
    if entry is Ellipsis:
        return "..."
    return str(entry)


def format_index(index):
    """Format an index tuple as a readable string such as ``0, :, 1``."""
    return ", ".join(format_token(entry) for entry in index)


def explain_index(shape, index):
    """Build the lines explaining how an index transforms a shape."""
    tokens = validate_index(index, shape)
    lines = [
        f"Original shape: {format_shape(shape)}",
        f"Index: {format_index(index)}",
        f"Result shape: {format_shape(result_shape(shape, index))}",
    ]
    axis = 0  # source axis
    out = 0  # result axis position
    for tok in tokens:
        if tok is None:
            lines.append(f"A new size 1 axis is inserted by None at result position {out}.")
            out += 1
            continue
        if isinstance(tok, int):
            lines.append(
                f"Axis {axis} is removed because integer index {tok} is used."
            )
        else:
            lines.append(
                f"Axis {axis} is kept because slice {format_slice(tok)} is used."
            )
            out += 1
        axis += 1
    return lines
