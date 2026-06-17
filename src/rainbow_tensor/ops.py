"""Shape changing and combining operations.

This module computes result shapes and the coordinate mappings for the
operations a learner meets when rearranging, summarising, combining, or
stretching tensors. Everything here is pure Python over shapes and integer
coordinates, so no tensor library is imported. The notebook layer turns these
mappings into values to draw, and the tests cross check every result against
NumPy.
"""

from math import prod

from .shape import flat_index


def unflatten(flat, shape):
    """Return the coordinate at row major flat position ``flat`` in ``shape``."""
    coord = []
    for dim in reversed(shape):
        coord.append(flat % dim)
        flat //= dim
    return tuple(reversed(coord))


# Reshape ------------------------------------------------------------------


def reshape_result_shape(old_shape, new_shape):
    """Resolve ``new_shape`` against ``old_shape`` and validate the element count.

    One axis may be ``-1`` and is inferred from the remaining axes, matching
    the NumPy convention. The total number of elements must stay the same.
    """
    new = tuple(new_shape)
    neg = [i for i, d in enumerate(new) if d == -1]
    if len(neg) > 1:
        raise ValueError("can only specify one unknown dimension with -1")
    for d in new:
        if d < -1 or d == 0:
            raise ValueError(f"invalid reshape dimension {d}")
    total = prod(old_shape)
    if neg:
        rest = prod(d for d in new if d != -1)
        if rest == 0 or total % rest != 0:
            raise ValueError(
                f"cannot reshape size {total} into {new}"
            )
        new = tuple(total // rest if d == -1 else d for d in new)
    if prod(new) != total:
        raise ValueError(
            f"cannot reshape tensor of size {total} into shape {tuple(new_shape)}"
        )
    return new


def reshape_source_coord(result_coord, old_shape, new_shape):
    """Map a destination coordinate back to its source coordinate.

    Reshape keeps row major order, so the element at flat position ``k`` in the
    new layout is the same element at flat position ``k`` in the old layout.
    """
    return unflatten(flat_index(result_coord, new_shape), old_shape)


# Transpose and permute ----------------------------------------------------


def transpose_axes(ndim, axes):
    """Validate ``axes`` and return the permutation as a tuple.

    ``None`` reverses the axis order, matching ``numpy.transpose``.
    """
    if axes is None:
        return tuple(reversed(range(ndim)))
    axes = tuple(axes)
    if sorted(axes) != list(range(ndim)):
        raise ValueError(
            f"axes {axes} is not a permutation of the {ndim} tensor axes"
        )
    return axes


def transpose_result_shape(shape, axes):
    """Return the shape after moving each axis to its permuted position."""
    return tuple(shape[a] for a in axes)


def transpose_source_coord(result_coord, axes):
    """Map a destination coordinate back to its source coordinate.

    Result axis ``r`` is drawn from source axis ``axes[r]``, so the source
    coordinate places ``result_coord[r]`` at source axis ``axes[r]``.
    """
    source = [0] * len(axes)
    for r, a in enumerate(axes):
        source[a] = result_coord[r]
    return tuple(source)


# Axis reductions ----------------------------------------------------------


def reduce_result_shape(shape, axis):
    """Return the shape after collapsing ``axis``."""
    axis = _check_axis(axis, len(shape))
    return tuple(s for i, s in enumerate(shape) if i != axis)


def reduce_source_coords(result_coord, shape, axis):
    """Yield every source coordinate that collapses into ``result_coord``."""
    axis = _check_axis(axis, len(shape))
    for k in range(shape[axis]):
        yield result_coord[:axis] + (k,) + result_coord[axis:]


def _check_axis(axis, ndim):
    """Resolve a possibly negative axis and bounds check it."""
    resolved = axis + ndim if axis < 0 else axis
    if not 0 <= resolved < ndim:
        raise ValueError(f"axis {axis} is out of range for a rank {ndim} tensor")
    return resolved
