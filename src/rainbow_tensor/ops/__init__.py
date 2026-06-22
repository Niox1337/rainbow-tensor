"""Shape changing and combining operations.

This package computes result shapes and the coordinate mappings for the
operations a learner meets when rearranging, summarising, combining, or
stretching tensors. Everything here is pure Python over shapes and integer
coordinates, so no tensor library is imported. The views layer turns these
mappings into values to draw, and the tests cross check every result against
NumPy.

The math is split by operation family to mirror the views layer. Every name a
caller used from the old flat module stays importable from here.
"""

from .combining import (
    broadcast_result_shape,
    broadcast_source_coord,
    broadcast_stretched_axes,
    concatenate_result_shape,
    concatenate_source,
    repeat_result_shape,
    repeat_source_coord,
    repeat_source_positions,
    stack_result_shape,
    stack_source,
    take_axis_and_indices,
    take_result_shape,
    take_source_coord,
)
from .einsum import (
    einsum_contracted_labels,
    einsum_index_sizes,
    einsum_result_shape,
    einsum_selected_coords,
    parse_einsum_subscripts,
)
from .reductions import (
    matmul_result_shape,
    matmul_source_terms,
    reduce_result_shape,
    reduce_source_coords,
)
from .reshaping import (
    expand_dims_axes,
    expand_dims_result_shape,
    expand_dims_source_coord,
    moveaxis_axes,
    reshape_result_shape,
    reshape_source_coord,
    squeeze_axes,
    squeeze_result_shape,
    squeeze_source_coord,
    swapaxes_axes,
    transpose_axes,
    transpose_result_shape,
    transpose_source_coord,
    unflatten,
)

__all__ = [
    "broadcast_result_shape",
    "broadcast_source_coord",
    "broadcast_stretched_axes",
    "concatenate_result_shape",
    "concatenate_source",
    "einsum_contracted_labels",
    "einsum_index_sizes",
    "einsum_result_shape",
    "einsum_selected_coords",
    "expand_dims_axes",
    "expand_dims_result_shape",
    "expand_dims_source_coord",
    "matmul_result_shape",
    "matmul_source_terms",
    "moveaxis_axes",
    "parse_einsum_subscripts",
    "reduce_result_shape",
    "reduce_source_coords",
    "repeat_result_shape",
    "repeat_source_coord",
    "repeat_source_positions",
    "reshape_result_shape",
    "reshape_source_coord",
    "squeeze_axes",
    "squeeze_result_shape",
    "squeeze_source_coord",
    "stack_result_shape",
    "stack_source",
    "swapaxes_axes",
    "take_axis_and_indices",
    "take_result_shape",
    "take_source_coord",
    "transpose_axes",
    "transpose_result_shape",
    "transpose_source_coord",
    "unflatten",
]
