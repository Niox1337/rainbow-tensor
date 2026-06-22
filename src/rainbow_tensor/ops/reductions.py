"""Shape math for matrix multiplication and axis reductions.

Pure Python over shapes and integer coordinates for matmul, sum, and mean.
Matmul builds on the broadcast rules in :mod:`rainbow_tensor.ops.combining`.
No tensor library is imported, so the views layer turns these mappings into
values to draw and the tests cross check every result against NumPy.
"""

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
    terms = []
    for j in range(k):
        a_full = a_batch_coord + (row, j)
        b_full = b_batch_coord + (j, col)
        a_coord = a_full[1:] if a1 else a_full
        b_coord = b_full[:-1] if b1 else b_full
        terms.append((a_coord, b_coord))
    return terms


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
