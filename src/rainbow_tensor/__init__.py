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

from .explanations import (
    available_languages,
    get_language,
    get_resolved_language,
    load_translations,
    set_language,
)
from .interactive import FocusExplorer, explore
from .memory import memory
from .provenance import Flow, TrackedTensor, ValueBudgetExceeded
from .renderers import (
    SVG,
    SvgRenderer,
    get_default_renderer,
    register_renderer,
    resolve_renderer,
    set_default_renderer,
)
from .theme import (
    AUTO,
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
from .tracing import OperandRef, OutputTrace
from .views import (
    broadcast,
    concatenate,
    einsum,
    expand_dims,
    index,
    matmul,
    mean,
    moveaxis,
    repeat,
    reshape,
    shape,
    squeeze,
    stack,
    sum,
    swapaxes,
    take,
    transpose,
)
from .visual import TensorVisual

__version__ = "1.3.1"
__all__ = [
    "shape",
    "memory",
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
    "concatenate",
    "stack",
    "take",
    "repeat",
    "broadcast",
    "einsum",
    "TensorVisual",
    "Flow",
    "TrackedTensor",
    "ValueBudgetExceeded",
    "OperandRef",
    "OutputTrace",
    "explore",
    "FocusExplorer",
    "SvgRenderer",
    "SVG",
    "get_default_renderer",
    "set_default_renderer",
    "register_renderer",
    "resolve_renderer",
    "Theme",
    "AUTO",
    "LIGHT",
    "DARK",
    "get_default_theme",
    "set_default_theme",
    "get_default_axis_colors",
    "set_default_axis_colors",
    "register_theme",
    "resolve_theme",
    "set_language",
    "get_language",
    "get_resolved_language",
    "available_languages",
    "load_translations",
    "__version__",
]
