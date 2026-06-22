"""Shape math for combining and stretching tensors.

Pure Python over shapes and integer coordinates for repeat, take, concatenate,
stack, and broadcast. No tensor library is imported, so the views layer turns
these mappings into values to draw and the tests cross check every result
against NumPy.
"""

from ..shape import _check_axis

# Repeat -------------------------------------------------------------------


def repeat_source_positions(axis_size, repeats):
    """Return the source index each result position along the axis copies.

    ``repeats`` is a single count applied to every element, or one count per
    element along the axis, matching ``numpy.repeat``. Element ``i`` contributes
    ``repeats[i]`` adjacent result positions, so the returned list reads as the
    materialised copies in order.
    """
    if isinstance(repeats, int):
        counts = [repeats] * axis_size
    else:
        counts = [int(r) for r in repeats]
        if len(counts) != axis_size:
            raise ValueError(
                f"repeats has length {len(counts)} but axis has size {axis_size}"
            )
    positions = []
    for i, count in enumerate(counts):
        if count < 0:
            raise ValueError(f"repeats may not be negative, got {count}")
        positions.extend([i] * count)
    return positions


def repeat_result_shape(shape, repeats, axis):
    """Return the shape after repeating elements along ``axis``."""
    axis = _check_axis(axis, len(shape))
    positions = repeat_source_positions(shape[axis], repeats)
    return shape[:axis] + (len(positions),) + shape[axis + 1:]


def repeat_source_coord(result_coord, source_positions, axis):
    """Map a result coordinate back to the source element it copies.

    Along the repeated axis the result position picks its source index from the
    materialised copy list, and every other axis is copied straight through.
    """
    source = source_positions[result_coord[axis]]
    return result_coord[:axis] + (source,) + result_coord[axis + 1:]


# Take ---------------------------------------------------------------------


def take_axis_and_indices(shape, indices, axis):
    """Resolve the axis and the gather indices for a take.

    The axis is normalised against the rank and each index is normalised against
    the axis size, so negative axes and negative indices both work like
    ``numpy.take``. An out of range index raises a clear error.
    """
    axis = _check_axis(axis, len(shape))
    size = shape[axis]
    resolved = []
    for raw in indices:
        i = int(raw)
        r = i + size if i < 0 else i
        if not 0 <= r < size:
            raise ValueError(
                f"take index {i} is out of range for axis {axis} of size {size}"
            )
        resolved.append(r)
    return axis, tuple(resolved)


def take_result_shape(shape, indices, axis):
    """Return the shape after gathering ``indices`` along ``axis``.

    The chosen axis is replaced by the number of indices, so the rank stays the
    same and only the gathered axis changes length.
    """
    axis, resolved = take_axis_and_indices(shape, indices, axis)
    return shape[:axis] + (len(resolved),) + shape[axis + 1:]


def take_source_coord(result_coord, resolved_indices, axis):
    """Map a result coordinate back to its source coordinate.

    Along the gathered axis the result position selects one source position from
    the resolved index list, and every other axis is copied straight through.
    """
    source = resolved_indices[result_coord[axis]]
    return result_coord[:axis] + (source,) + result_coord[axis + 1:]


# Concatenate --------------------------------------------------------------


def concatenate_result_shape(shapes, axis):
    """Return the shape after joining ``shapes`` along ``axis``.

    Every operand must match on every axis except the joined one.
    """
    if not shapes:
        raise ValueError("need at least one tensor to concatenate")
    ndim = len(shapes[0])
    axis = _check_axis(axis, ndim)
    for s in shapes:
        if len(s) != ndim:
            raise ValueError("all tensors must have the same rank to concatenate")
        for i in range(ndim):
            if i != axis and s[i] != shapes[0][i]:
                raise ValueError(
                    f"all tensors must match on axis {i} to concatenate along "
                    f"axis {axis}, got {s[i]} and {shapes[0][i]}"
                )
    out = list(shapes[0])
    out[axis] = sum(s[axis] for s in shapes)
    return tuple(out)


def concatenate_source(result_coord, shapes, axis):
    """Map a result coordinate to its ``(operand_index, source_coord)``."""
    axis = _check_axis(axis, len(shapes[0]))
    pos = result_coord[axis]
    for i, s in enumerate(shapes):
        if pos < s[axis]:
            coord = result_coord[:axis] + (pos,) + result_coord[axis + 1:]
            return i, coord
        pos -= s[axis]
    raise IndexError("result coordinate is past the end of the joined axis")


# Stack --------------------------------------------------------------------


def stack_result_shape(shapes, axis):
    """Return the shape after stacking ``shapes`` on a new ``axis``.

    Every operand must have the same shape. The new axis may sit anywhere from
    ``0`` to the operand rank.
    """
    if not shapes:
        raise ValueError("need at least one tensor to stack")
    base = shapes[0]
    for s in shapes:
        if tuple(s) != tuple(base):
            raise ValueError(
                f"all tensors must have the same shape to stack, got {tuple(s)} "
                f"and {tuple(base)}"
            )
    axis = _check_axis(axis, len(base) + 1)
    out = list(base)
    out.insert(axis, len(shapes))
    return tuple(out)


def stack_source(result_coord, axis, ndim):
    """Map a result coordinate to its ``(operand_index, source_coord)``."""
    axis = _check_axis(axis, ndim + 1)
    i = result_coord[axis]
    coord = result_coord[:axis] + result_coord[axis + 1:]
    return i, coord


# Broadcasting -------------------------------------------------------------


def broadcast_result_shape(shapes):
    """Return the broadcast shape of ``shapes`` using the NumPy rules.

    Axes are aligned from the right. On each axis the sizes must be equal or
    one of them must be ``1``, which stretches to match the other.
    """
    ndim = max(len(s) for s in shapes)
    result = []
    for offset in range(1, ndim + 1):
        sizes = [s[-offset] for s in shapes if offset <= len(s)]
        big = [n for n in sizes if n != 1]
        if len(set(big)) > 1:
            raise ValueError(
                f"operands could not be broadcast together: axis sizes {sizes}"
            )
        result.append(big[0] if big else 1)
    return tuple(reversed(result))


def broadcast_source_coord(result_coord, operand_shape):
    """Map a result coordinate back to ``operand_shape``.

    Leading axes the operand does not have are dropped, and a stretched axis of
    size one always reads position zero.
    """
    extra = len(result_coord) - len(operand_shape)
    coord = []
    for i, size in enumerate(operand_shape):
        pos = result_coord[extra + i]
        coord.append(0 if size == 1 else pos)
    return tuple(coord)


def broadcast_stretched_axes(operand_shape, result_shape):
    """Return the result axes where ``operand_shape`` is stretched.

    This is every leading axis the operand gains, plus every size one axis that
    grows to match the result.
    """
    extra = len(result_shape) - len(operand_shape)
    stretched = list(range(extra))
    for i, size in enumerate(operand_shape):
        if size == 1 and result_shape[extra + i] != 1:
            stretched.append(extra + i)
    return stretched
