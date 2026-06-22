"""Notebook display layer.

Each public function returns a small object that renders as SVG in a notebook
and stays inspectable in plain Python, so the package is testable outside a
notebook. The functions are split by operation family (shapes, reshaping,
reductions, combining, einsum) and re-exported here, mirroring the math split
in :mod:`rainbow_tensor.ops`.
"""

from .combining import broadcast, concatenate, repeat, stack, take
from .einsum import einsum
from .reductions import matmul, mean, sum
from .reshaping import expand_dims, moveaxis, reshape, squeeze, swapaxes, transpose
from .shapes import index, shape

__all__ = [
    "shape",
    "index",
    "reshape",
    "transpose",
    "swapaxes",
    "moveaxis",
    "squeeze",
    "expand_dims",
    "matmul",
    "sum",
    "mean",
    "repeat",
    "take",
    "concatenate",
    "stack",
    "broadcast",
    "einsum",
]
