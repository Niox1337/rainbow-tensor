"""rainbow-tensor.

A small package for IPython and Jupyter notebooks that visualises tensor
shape, indexing, and slicing as SVG. It is meant for people learning how a
tensor is structured and how an indexing expression selects elements.

Public API:

    import rainbow_tensor as rt

    rt.shape((2, 2, 2))
    rt.index((2, 2, 2), (0, slice(None), 1))

    rt.shape((2, 2, 2), theme="dark")
    rt.set_default_theme("dark")
"""

from .notebook import (
    TensorVisual,
    broadcast,
    concatenate,
    einsum,
    expand_dims,
    index,
    matmul,
    mean,
    reshape,
    shape,
    squeeze,
    stack,
    sum,
    swapaxes,
    take,
    transpose,
)
from .renderers import (
    SVG,
    SvgRenderer,
    get_default_renderer,
    register_renderer,
    resolve_renderer,
    set_default_renderer,
)
from .theme import (
    DARK,
    LIGHT,
    Theme,
    get_default_axis_colors,
    get_default_theme,
    register_theme,
    resolve_theme,
    set_default_axis_colors,
    set_default_theme,
)

__version__ = "0.13.0"
__all__ = [
    "shape",
    "index",
    "reshape",
    "transpose",
    "swapaxes",
    "squeeze",
    "expand_dims",
    "matmul",
    "sum",
    "mean",
    "concatenate",
    "stack",
    "take",
    "broadcast",
    "einsum",
    "TensorVisual",
    "SvgRenderer",
    "SVG",
    "get_default_renderer",
    "set_default_renderer",
    "register_renderer",
    "resolve_renderer",
    "Theme",
    "LIGHT",
    "DARK",
    "get_default_theme",
    "set_default_theme",
    "get_default_axis_colors",
    "set_default_axis_colors",
    "register_theme",
    "resolve_theme",
    "__version__",
]
