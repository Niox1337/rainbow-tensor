"""rainbow-tensor.

A small package for IPython and Jupyter notebooks that visualises tensor
shape, indexing, and slicing as SVG. It is meant for people learning how a
tensor is structured and how an indexing expression selects elements.

Public API:

    from rainbow_tensor import show_shape, show_index

    show_shape((2, 2, 2))
    show_index((2, 2, 2), (0, slice(None), 1))
"""

from .notebook import show_index, show_shape

__version__ = "0.2.1"
__all__ = ["show_shape", "show_index", "__version__"]
