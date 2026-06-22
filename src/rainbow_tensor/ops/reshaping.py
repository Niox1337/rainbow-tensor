"""Shape math for rearranging a single tensor.

Pure Python over shapes and integer coordinates for reshape, transpose,
swapaxes, moveaxis, squeeze, and expand_dims. No tensor library is imported,
so the views layer turns these mappings into values to draw and the tests
cross check every result against NumPy.
"""

from math import prod

from ..shape import _check_axis, flat_index


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


# Swap axes ----------------------------------------------------------------


def swapaxes_axes(ndim, axis1, axis2):
    """Return the transpose permutation for swapping two axes."""
    first = _check_axis(axis1, ndim)
    second = _check_axis(axis2, ndim)
    axes = list(range(ndim))
    axes[first], axes[second] = axes[second], axes[first]
    return tuple(axes)


# Move axis ----------------------------------------------------------------


def _normalize_axis_seq(axis, ndim):
    """Resolve an axis or sequence of axes to a tuple of distinct positions."""
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    resolved = tuple(_check_axis(a, ndim) for a in axes)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"repeated axis in {axes}")
    return resolved


def moveaxis_axes(ndim, source, destination):
    """Return the transpose permutation that moves axes to new positions.

    ``source`` and ``destination`` are an axis or a sequence of axes of equal
    length. Each moved axis lands at its destination and the remaining axes keep
    their relative order, matching ``numpy.moveaxis``. The permutation is
    expressed in the same form as :func:`transpose_axes`, so result axis ``r`` is
    drawn from source axis ``order[r]``.
    """
    src = _normalize_axis_seq(source, ndim)
    dst = _normalize_axis_seq(destination, ndim)
    if len(src) != len(dst):
        raise ValueError(
            f"moveaxis source {src} and destination {dst} must have the same length"
        )
    order = [n for n in range(ndim) if n not in src]
    for d, s in sorted(zip(dst, src)):
        order.insert(d, s)
    return tuple(order)


# Squeeze and expand dims --------------------------------------------------


def squeeze_axes(shape, axis=None):
    """Return the source axes squeeze removes, validated against ``shape``.

    With ``axis`` as ``None`` every size one axis is removed. An explicit axis,
    or a tuple of axes, must point at a size one axis, matching ``numpy.squeeze``.
    Negative axes count from the end.
    """
    ndim = len(shape)
    if axis is None:
        return tuple(i for i, size in enumerate(shape) if size == 1)
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    removed = []
    for a in axes:
        r = _check_axis(a, ndim)
        if shape[r] != 1:
            raise ValueError(
                f"cannot select axis {a} to squeeze out which has size {shape[r]}, not one"
            )
        removed.append(r)
    if len(set(removed)) != len(removed):
        raise ValueError("squeeze axes may not repeat")
    return tuple(sorted(removed))


def squeeze_result_shape(shape, axis=None):
    """Return the shape after removing the squeezed size one axes."""
    removed = set(squeeze_axes(shape, axis))
    return tuple(size for i, size in enumerate(shape) if i not in removed)


def squeeze_source_coord(result_coord, removed):
    """Map a result coordinate back to its source coordinate.

    Each removed axis had size one, so its source position is always zero.
    Inserting a zero at every removed axis, lowest first, rebuilds the source
    coordinate.
    """
    coord = list(result_coord)
    for axis in sorted(removed):
        coord.insert(axis, 0)
    return tuple(coord)


def expand_dims_axes(ndim, axis):
    """Return the inserted axis positions in the expanded shape.

    ``axis`` is a single position or a tuple of positions in the result, which
    has ``ndim + len(axis)`` axes. Negative positions count from the end of the
    result, matching ``numpy.expand_dims``.
    """
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    out_ndim = ndim + len(axes)
    inserted = [_check_axis(a, out_ndim) for a in axes]
    if len(set(inserted)) != len(inserted):
        raise ValueError("expand_dims axes may not repeat")
    return tuple(sorted(inserted))


def expand_dims_result_shape(shape, axis):
    """Return the shape after inserting a size one axis at each ``axis``."""
    inserted = set(expand_dims_axes(len(shape), axis))
    out_ndim = len(shape) + len(inserted)
    source = iter(shape)
    return tuple(1 if i in inserted else next(source) for i in range(out_ndim))


def expand_dims_source_coord(result_coord, inserted):
    """Map a result coordinate back to its source coordinate.

    The inserted axes carry no source position, so dropping them leaves the
    surviving coordinates in their original order.
    """
    inserted = set(inserted)
    return tuple(c for i, c in enumerate(result_coord) if i not in inserted)
