"""Shape math for matrix multiplication and axis reductions.

Pure Python over shapes and integer coordinates for matmul, sum, and mean.
Matmul builds on the broadcast rules in :mod:`rainbow_tensor.ops.combining`.
No tensor library is imported, so the views layer turns these mappings into
values to draw and the tests cross check every result against NumPy.
"""

from math import prod

from ..shape import _check_axis
from .combining import broadcast_result_shape, broadcast_source_coord

# Matmul -------------------------------------------------------------------


def _matmul_promote(a_shape, b_shape):
    """Return the rank lifted operand shapes and the 1-D promotion flags.

    A 1-D first operand gains a leading axis and a 1-D second operand gains a
    trailing axis, matching how ``numpy.matmul`` treats vectors. The lifted
    shapes always have at least two axes.
    """
    if len(a_shape) == 0 or len(b_shape) == 0:
        raise ValueError("matmul operands must be at least 1-D")
    a1 = len(a_shape) == 1
    b1 = len(b_shape) == 1
    aa = (1,) + tuple(a_shape) if a1 else tuple(a_shape)
    bb = tuple(b_shape) + (1,) if b1 else tuple(b_shape)
    if aa[-1] != bb[-2]:
        raise ValueError(
            f"matmul inner dimensions do not match: {aa[-1]} and {bb[-2]}"
        )
    return aa, bb, a1, b1


def matmul_result_shape(a_shape, b_shape):
    """Return the shape of ``a_shape @ b_shape``, matching ``numpy.matmul``.

    The batch axes left of the last two broadcast together, the shared inner
    axis contracts, and a 1-D operand drops its promoted axis from the result.
    """
    aa, bb, a1, b1 = _matmul_promote(a_shape, b_shape)
    batch = broadcast_result_shape([aa[:-2], bb[:-2]])
    full = list(batch) + [aa[-2], bb[-1]]
    nb = len(batch)
    drop = []
    if a1:
        drop.append(nb)
    if b1:
        drop.append(nb + 1)
    for d in sorted(drop, reverse=True):
        del full[d]
    return tuple(full)


def matmul_source_terms(out_coord, a_shape, b_shape):
    """Return the ``(a_coord, b_coord)`` pairs that combine into one output.

    The output element at ``out_coord`` is the sum over the shared inner axis of
    the products of these operand elements. The batch part of ``out_coord`` maps
    back through broadcasting, and a promoted vector axis is dropped from the
    returned operand coordinate.
    """
    return list(iter_matmul_source_terms(out_coord, a_shape, b_shape))


def iter_matmul_source_terms(out_coord, a_shape, b_shape):
    """Yield each source-coordinate pair in ascending contraction-index order.

    Unlike ``matmul_source_terms``, this iterator does not materialize the full
    contraction. A trace can therefore inspect its first terms without walking
    the rest, while existing callers can retain the list-returning helper.
    """
    aa, bb, a1, b1 = _matmul_promote(a_shape, b_shape)
    k = aa[-1]
    a_batch, b_batch = aa[:-2], bb[:-2]
    batch = broadcast_result_shape([a_batch, b_batch])
    nb = len(batch)

    full = []
    cursor = iter(out_coord)
    for pos in range(nb + 2):
        if (a1 and pos == nb) or (b1 and pos == nb + 1):
            full.append(0)
        else:
            full.append(next(cursor))
    batch_coord = tuple(full[:nb])
    row, col = full[nb], full[nb + 1]

    a_batch_coord = broadcast_source_coord(batch_coord, a_batch)
    b_batch_coord = broadcast_source_coord(batch_coord, b_batch)
    for j in range(k):
        a_full = a_batch_coord + (row, j)
        b_full = b_batch_coord + (j, col)
        a_coord = a_full[1:] if a1 else a_full
        b_coord = b_full[:-1] if b1 else b_full
        yield a_coord, b_coord


# Axis reductions ----------------------------------------------------------


def normalize_reduction_axes(axis, ndim):
    """Return distinct reduction axes in ascending source-axis order.

    ``None`` selects every axis. An integer follows Python's index protocol,
    and a tuple selects zero or more axes. Negative axes count from the end.
    Booleans, floats and non-tuple containers are rejected. Sorting makes
    source-term order independent of the order in which axes were supplied.
    """
    if axis is None:
        return tuple(range(ndim))
    axes = axis if isinstance(axis, tuple) else (axis,)
    resolved = tuple(_check_axis(value, ndim) for value in axes)
    if len(set(resolved)) != len(resolved):
        raise ValueError("reduction axes may not repeat")
    return tuple(sorted(resolved))


def validate_keepdims(keepdims):
    """Normalize a Python or NumPy boolean scalar without importing NumPy."""
    if not isinstance(keepdims, bool) and type(keepdims).__name__ not in ("bool", "bool_"):
        raise TypeError("keepdims must be a boolean")
    return bool(keepdims)


def reduce_result_shape(shape, axis=None, *, keepdims=False):
    """Return the reduced shape, optionally retaining length-one reduced axes.

    ``axis=None`` reduces every axis, while ``axis=()`` leaves the shape intact.
    The default drops reduced axes, preserving the original single-axis API.
    """
    axes = normalize_reduction_axes(axis, len(shape))
    keepdims = validate_keepdims(keepdims)
    reduced = set(axes)
    if keepdims:
        return tuple(1 if i in reduced else size for i, size in enumerate(shape))
    return tuple(size for i, size in enumerate(shape) if i not in reduced)


def reduce_term_count(shape, axis=None):
    """Count contributions per output without enumerating any source coordinates.

    The product over no reduced axes is one, so an empty axis tuple has one
    source term per output and a mean divisor of one.
    """
    axes = normalize_reduction_axes(axis, len(shape))
    return prod(shape[i] for i in axes)


def reduce_source_index(result_coord, shape, axis=None, *, keepdims=False):
    """Return a compact basic index for one output's contributing source group.

    ``result_coord`` is an already normalized coordinate in the result shape.
    Reduced source axes become full slices, while surviving axes are fixed at
    their output positions. With ``keepdims``, the reduced length-one output
    positions are ignored. The index can construct a compact ``BasicSelection``
    without materializing its Cartesian product.
    """
    axes = normalize_reduction_axes(axis, len(shape))
    keepdims = validate_keepdims(keepdims)
    reduced = set(axes)
    surviving = iter(result_coord)
    return tuple(
        slice(None) if i in reduced else (result_coord[i] if keepdims else next(surviving))
        for i in range(len(shape))
    )


def reduce_source_coords(result_coord, shape, axis=None, *, keepdims=False):
    """Yield one output's source coordinates in deterministic row-major order.

    ``result_coord`` is already normalized in the result shape. An odometer
    advances only reduced axes, with the last source axis changing fastest.
    Memory depends on rank rather than the number of terms, and no axis ranges
    are pooled. An empty axis tuple yields exactly the output coordinate itself.
    """
    axes = normalize_reduction_axes(axis, len(shape))
    source_index = reduce_source_index(result_coord, shape, axes, keepdims=keepdims)
    coordinate = [0 if isinstance(token, slice) else token for token in source_index]
    count = prod(shape[i] for i in axes)
    for _ in range(count):
        yield tuple(coordinate)
        for i in reversed(axes):
            coordinate[i] += 1
            if coordinate[i] < shape[i]:
                break
            coordinate[i] = 0
