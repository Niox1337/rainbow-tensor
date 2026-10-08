"""Visualise tensor operations and trace where each result element comes from.

Rainbow Tensor renders shapes, indexing, arithmetic, reductions, and recorded
operation chains as SVG for IPython and Jupyter. Its interactive lessons help
learners connect output coordinates to source elements and intermediate steps.

Public API:

    import rainbow_tensor as rt

    rt.shape((2, 2, 2))
    rt.index((2, 2, 2), (0, slice(None), 1))

    rt.shape((2, 2, 2), theme="dark")
    rt.set_default_theme("dark")
"""

from .capabilities import capabilities
from .explanations import (
    available_languages,
    get_language,
    get_resolved_language,
    load_translations,
    set_language,
)
from .interactive import FocusExplorer, explore
from .lessons import LessonRecording, capture_lesson
from .memory import memory
from .playground import ReductionPlayground, reduction_playground
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
from .tracing import BinaryExpression, OperandRef, OutputTrace, SelectionExpression
from .views import (
    add,
    argmax,
    argmin,
    broadcast,
    concatenate,
    divide,
    einsum,
    equal,
    expand_dims,
    greater,
    greater_equal,
    index,
    less,
    less_equal,
    matmul,
    max,
    mean,
    min,
    moveaxis,
    multiply,
    not_equal,
    repeat,
    reshape,
    shape,
    squeeze,
    stack,
    subtract,
    sum,
    swapaxes,
    take,
    transpose,
    where,
)
from .visual import TensorVisual
from .walkthrough import Walkthrough, walkthrough

__version__ = "1.8.0"
__all__ = [
    "capabilities",
    "capture_lesson",
    "LessonRecording",
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
    "add",
    "subtract",
    "multiply",
    "divide",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "where",
    "min",
    "max",
    "argmin",
    "argmax",
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
    "BinaryExpression",
    "SelectionExpression",
    "OutputTrace",
    "explore",
    "FocusExplorer",
    "Walkthrough",
    "walkthrough",
    "ReductionPlayground",
    "reduction_playground",
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
