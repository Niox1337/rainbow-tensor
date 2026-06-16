"""Index validation, slice expansion, and selection computation.

This module validates indexing expressions, expands slices into integer
positions, computes the selected coordinates, and computes the result shape
after indexing.
"""

import itertools

from .shape import format_shape


def validate_index(index, shape):
    """Validate an index tuple against a tensor shape.

    The index must be a tuple whose length matches the tensor rank. Each
    entry must be a non-negative in-range integer or a slice. Negative
    integers and other entry types are rejected in this version.
    """
    if not isinstance(index, tuple):
        raise TypeError(
            f"index must be a tuple, got {type(index).__name__}. "
            f"For a single axis use a one element tuple such as (0,)"
        )

    if len(index) != len(shape):
        raise ValueError(
            f"index length {len(index)} does not match tensor rank {len(shape)}"
        )

    for axis, (entry, size) in enumerate(zip(index, shape)):
        if isinstance(entry, bool):
            raise TypeError(f"axis {axis}: boolean indices are not supported")
        if isinstance(entry, int):
            if entry < 0:
                raise IndexError(
                    f"axis {axis}: negative indices are not supported, got {entry}"
                )
            if entry >= size:
                raise IndexError(
                    f"axis {axis}: integer index {entry} is out of range for size {size}"
                )
        elif isinstance(entry, slice):
            continue
        else:
            raise TypeError(
                f"axis {axis}: unsupported index entry {entry!r}. "
                f"Only integers and slices are supported"
            )

    return index


def expand_slice(sl, size):
    """Expand a slice into the list of integer positions it selects."""
    return list(range(*sl.indices(size)))


def selected_coordinates(shape, index):
    """Compute every coordinate selected by ``index`` in row-major order."""
    validate_index(index, shape)
    axes = []
    for entry, size in zip(index, shape):
        if isinstance(entry, int):
            axes.append([entry])
        else:
            axes.append(expand_slice(entry, size))
    return list(itertools.product(*axes))


def result_shape(shape, index):
    """Compute the result shape after indexing.

    Integer entries remove their axis. Slice entries keep their axis with a
    size equal to the number of selected positions.
    """
    validate_index(index, shape)
    out = []
    for entry, size in zip(index, shape):
        if isinstance(entry, slice):
            out.append(len(expand_slice(entry, size)))
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


def format_index(index):
    """Format an index tuple as a readable string such as ``0, :, 1``."""
    parts = []
    for entry in index:
        if isinstance(entry, slice):
            parts.append(format_slice(entry))
        else:
            parts.append(str(entry))
    return ", ".join(parts)


def explain_index(shape, index):
    """Build the lines explaining how an index transforms a shape."""
    validate_index(index, shape)
    lines = [
        f"Original shape: {format_shape(shape)}",
        f"Index: {format_index(index)}",
        f"Result shape: {format_shape(result_shape(shape, index))}",
    ]
    for axis, entry in enumerate(index):
        if isinstance(entry, int):
            lines.append(
                f"Axis {axis} is removed because integer index {entry} is used."
            )
        else:
            lines.append(
                f"Axis {axis} is kept because slice {format_slice(entry)} is used."
            )
    return lines
