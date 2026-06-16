"""Index validation, slice expansion, and selection computation.

This module validates indexing expressions, expands slices into integer
positions, computes the selected coordinates, and computes the result shape
after indexing.
"""

import itertools


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
