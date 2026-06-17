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

from .notebook import TensorVisual, index, shape
from .theme import (
    DARK,
    LIGHT,
    Theme,
    get_default_theme,
    register_theme,
    resolve_theme,
    set_default_theme,
)

__version__ = "0.3.0"
__all__ = [
    "shape",
    "index",
    "TensorVisual",
    "Theme",
    "LIGHT",
    "DARK",
    "get_default_theme",
    "set_default_theme",
    "register_theme",
    "resolve_theme",
    "__version__",
]
