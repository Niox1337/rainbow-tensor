"""Notebook display layer.

This module exposes the public functions ``show_shape`` and ``show_index``.
The real rendering is added in later changes. For now these are placeholders
so the public API can be imported and wired up.
"""


def show_shape(shape):
    """Visualise the structure of a tensor shape.

    Not implemented yet.
    """
    raise NotImplementedError("show_shape is not implemented yet")


def show_index(tensor_or_shape, index):
    """Visualise how an index selects elements from a tensor.

    Not implemented yet.
    """
    raise NotImplementedError("show_index is not implemented yet")
