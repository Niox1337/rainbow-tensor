"""Shape math for combining and stretching tensors.

Pure Python over shapes and integer coordinates for repeat, take, concatenate,
stack, and broadcast. No tensor library is imported, so the views layer turns
these mappings into values to draw and the tests cross check every result
against NumPy.
"""

from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate

from ..shape import _as_integer, _check_axis, _integer_or_sequence

# Repeat -------------------------------------------------------------------


def _repeat_spec(axis_size, repeats):
    """Validate repeat counts without allocating a list for every source element."""
    counts = _integer_or_sequence(repeats, "repeats")
    if not isinstance(counts, int):
        if len(counts) == 1:
            counts = counts[0]
        elif len(counts) != axis_size:
            raise ValueError(
                f"repeats has length {len(counts)} but axis has size {axis_size}"
            )
    for count in (counts,) if isinstance(counts, int) else counts:
        if count < 0:
            raise ValueError(f"repeats may not be negative, got {count}")
    return counts


def repeat_source_positions(axis_size, repeats):
    """Return the source index each result position along the axis copies.

    ``repeats`` is a single count applied to every element, or one count per
    element along the axis, matching ``numpy.repeat``. Element ``i`` contributes
    ``repeats[i]`` adjacent result positions, so the returned list reads as the
    materialised copies in order. Counts follow the integer index protocol,
    so booleans and floats are rejected instead of silently coerced.
    """
    repeats = _repeat_spec(axis_size, repeats)
    if repeats == 0:
        return []
    counts = (repeats for _ in range(axis_size)) if isinstance(repeats, int) else repeats
    positions = []
    for i, count in enumerate(counts):
        positions.extend([i] * count)
    return positions


@dataclass(frozen=True, slots=True)
class _RepeatSourceLookup:
    """Resolve repeat positions without storing each copy's source index.

    Uniform repetition needs only an integer count. Variable repetition stores
    one cumulative end per input element, so memory depends on the input axis
    rather than the number of copies. ``count`` remains usable when a logical
    output is larger than Python's platform-sized ``len`` limit.
    """

    count: int
    _fixed_count: int | None
    _ends: tuple[int, ...] = ()

    def __len__(self):
        """Return the output axis size when it fits Python's length protocol."""
        return self.count

    def __getitem__(self, position):
        """Return a source index, accepting integer and negative positions."""
        position = _as_integer(position, "repeat result position")
        if position < 0:
            position += self.count
        if not 0 <= position < self.count:
            raise IndexError("repeat result position is out of range")
        if self._fixed_count is not None:
            return position // self._fixed_count
        return bisect_right(self._ends, position)


def repeat_source_lookup(axis_size, repeats):
    """Build a bounded-memory mapping from repeat results to source positions.

    The lookup accepts a single repeat count or one count per input element,
    including zero counts. Read ``lookup[position]`` to resolve one output
    coordinate and ``lookup.count`` for its exact output size. Scalar counts
    use constant memory and variable counts use memory proportional to the
    input axis. No list proportional to the repeated output is constructed.
    """
    axis_size = _as_integer(axis_size, "axis size")
    if axis_size < 0:
        raise ValueError(f"axis size must be nonnegative, got {axis_size}")
    counts = _repeat_spec(axis_size, repeats)
    if isinstance(counts, int):
        return _RepeatSourceLookup(axis_size * counts, counts)
    ends = tuple(accumulate(counts))
    return _RepeatSourceLookup(ends[-1] if ends else 0, None, ends)


def repeat_result_shape(shape, repeats, axis):
    """Return the repeat shape, treating a scalar as one element on axis zero."""
    shape = shape or (1,)
    axis = _check_axis(axis, len(shape))
    counts = _repeat_spec(shape[axis], repeats)
    size = counts * shape[axis] if isinstance(counts, int) else sum(counts)
    return shape[:axis] + (size,) + shape[axis + 1:]


def repeat_source_coord(result_coord, source_positions, axis, shape=None):
    """Map a result coordinate back to the source element it copies.

    Along the repeated axis the result position picks its source index from a
    copy list or lazy lookup, and every other axis is copied straight through.
    Passing the original scalar ``shape=()`` maps each copy to coordinate ``()``.
    """
    if shape == ():
        return ()
    source = source_positions[result_coord[axis]]
    return result_coord[:axis] + (source,) + result_coord[axis + 1:]


# Take ---------------------------------------------------------------------


def take_axis_and_indices(shape, indices, axis):
    """Resolve the axis and the gather indices for a take.

    The axis is normalised against the rank and each index is normalised against
    the axis size, so negative axes and negative indices both work like
    ``numpy.take``. Indices follow the integer index protocol, with booleans
    and floats rejected. An out of range index raises a clear error.
    """
    shape = shape or (1,)
    axis = _check_axis(axis, len(shape))
    size = shape[axis]
    indices = _integer_or_sequence(indices, "take indices")
    scalar_index = isinstance(indices, int)
    resolved = []
    for i in (indices,) if scalar_index else indices:
        r = i + size if i < 0 else i
        if not 0 <= r < size:
            raise ValueError(
                f"take index {i} is out of range for axis {axis} of size {size}"
            )
        resolved.append(r)
    return axis, resolved[0] if scalar_index else tuple(resolved)


def take_result_shape(shape, indices, axis):
    """Return the shape after gathering ``indices`` along ``axis``.

    A scalar index removes the chosen axis. A sequence replaces it with the
    number of indices. A scalar input is treated as one element on axis zero.
    """
    axis, resolved = take_axis_and_indices(shape, indices, axis)
    shape = shape or (1,)
    if isinstance(resolved, int):
        return shape[:axis] + shape[axis + 1:]
    return shape[:axis] + (len(resolved),) + shape[axis + 1:]


def take_source_coord(result_coord, resolved_indices, axis, shape=None):
    """Map a result coordinate back to its source coordinate.

    Along the gathered axis the result position selects one source position from
    the resolved index list, and every other axis is copied straight through.
    A scalar index inserts its source position at the removed axis. Passing
    the original scalar ``shape=()`` returns the source coordinate ``()``.
    """
    if shape == ():
        return ()
    if isinstance(resolved_indices, int):
        return result_coord[:axis] + (resolved_indices,) + result_coord[axis:]
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
    if ndim == 0:
        raise ValueError("zero-dimensional tensors cannot be concatenated")
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
