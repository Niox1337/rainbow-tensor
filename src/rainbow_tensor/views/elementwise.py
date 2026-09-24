"""Broadcast-aware binary arithmetic with an ordered explanation for each output."""

from ..evaluation import (
    DEFAULT_MAX_TERMS,
    DEFAULT_MAX_TOTAL_TERMS,
    budgeted_values,
    evaluation_explanation,
    evaluation_plan,
)
from ..explanations import t
from ..numerics import numeric_explanation, numeric_semantics
from ..ops import broadcast_result_shape, broadcast_source_coord, broadcast_stretched_axes
from ..ops.elementwise import BINARY_SYMBOLS, evaluate_binary, normalize_operand
from ..renderers import resolve_renderer
from ..shape import extract_shape, format_shape
from ..theme import resolve_theme
from ..tracing import (
    BinaryExpression,
    OperandRef,
    OutputTrace,
    _normalize_focus,
    _trace_explanation,
)
from ..visual import (
    _preview_explanation,
    _shape_caption_parts,
    _source_value,
    _value_fn_for,
    _visual,
)


def _elementwise(operation, a, b, theme, precision, renderer, focus, max_terms, max_total_terms):
    """Plan bounded output work before reading either ordered source operand."""
    arrays = (normalize_operand(a), normalize_operand(b))
    shapes = tuple(extract_shape(array) for array in arrays)
    result = broadcast_result_shape(shapes)
    focused = _normalize_focus(focus, result)
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    selected_result = [] if focused is None else [focused]
    evaluation = evaluation_plan(1, max_terms, max_total_terms, result, selected_result, theme)
    source_values = tuple(_source_value(array, shape) for array, shape in zip(arrays, shapes))
    semantics = numeric_semantics(arrays)

    @budgeted_values(evaluation)
    def result_value(coordinate):
        left, right = (
            value(broadcast_source_coord(coordinate, shape))
            for value, shape in zip(source_values, shapes)
        )
        return evaluate_binary(operation, left, right)

    trace = None
    selected_sources = [[], []]
    if focused is not None:
        references = tuple(
            OperandRef(operand, broadcast_source_coord(focused, shape))
            for operand, shape in enumerate(shapes)
        )
        trace = OutputTrace(
            operation, focused, 1, (), True,
            expression=BinaryExpression(operation, references),
        )
        selected_sources = [[reference.coordinate] for reference in references]
    symbol = BINARY_SYMBOLS[operation]
    explanation = [t(
        "elementwise.operands", operation=operation, a=format_shape(shapes[0]),
        b=format_shape(shapes[1]), symbol=symbol,
    )]
    for label, shape in zip(("A", "B"), shapes):
        axes = broadcast_stretched_axes(shape, result)
        explanation.append(t(
            "broadcast.stretches" if axes else "broadcast.matches",
            i=label, shape=format_shape(shape), axes=axes,
        ))
    explanation.extend([
        t("common.result_shape", shape=format_shape(result)),
        t("elementwise.rule"),
        *_trace_explanation(trace),
        *_preview_explanation([*shapes, result], theme),
        *evaluation_explanation(evaluation),
        *numeric_explanation(semantics),
    ])
    if operation == "divide":
        explanation.append(t("elementwise.divide_zero"))
    panels = [
        {
            "shape": shape,
            "value_fn": _value_fn_for(array),
            "selected": selected,
            "caption_parts": _shape_caption_parts(label, shape, theme),
        }
        for label, array, shape, selected in zip(("A", "B"), arrays, shapes, selected_sources)
    ]
    panels.append({
        "shape": result,
        "value_fn": result_value,
        "selected": selected_result,
        "caption_parts": _shape_caption_parts(operation, result, theme),
    })
    content = renderer.render_panels(
        panels=panels, connectors=[symbol, "->"], explanation=explanation,
        theme=theme, precision=precision,
    )
    visual = _visual(
        content, shapes[0], renderer, selected=selected_sources[0], result=result,
        explanation=explanation,
    )
    visual.trace = trace
    visual.metadata.update(
        output_panel_indices=(2,), focused_output_panel=2 if trace is not None else None,
        trace_in_explanation=trace is not None, value_evaluation=evaluation,
        numeric_semantics=semantics,
    )
    return visual



def add(
    a, b, theme=None, precision=2, renderer=None, *, focus=None,
    max_terms=DEFAULT_MAX_TERMS, max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualize broadcast-aware addition ``a + b``.

    Operands can be array-like objects, numeric literals, or shape tuples.
    Shapes follow NumPy broadcasting. Tuples supply generated row-major
    placeholder values, while a numeric literal is a zero-dimensional value.
    Both operand coordinates are retained, even when the same input is used twice.

    The three panels show both inputs and the output. ``focus`` selects one
    output and pins its corresponding source in each input. Omit it to select
    the first output. Scalars use ``()`` and empty results have no trace.
    The trace's binary expression preserves the ordered operands without
    describing subtraction or division as a sum of products.

    Numerical previews use Python scalars, so backend dtype, overflow and
    rounding can differ. Each visible output costs one binary operation.
    ``max_terms`` and ``max_total_terms`` limit per-output and total output
    work. Over-budget output values show ``?`` and bounded input previews
    remain available. ``None`` disables the corresponding calculation limit.
    """
    return _elementwise(
        "add", a, b, theme, precision, renderer, focus, max_terms, max_total_terms,
    )



def subtract(
    a, b, theme=None, precision=2, renderer=None, *, focus=None,
    max_terms=DEFAULT_MAX_TERMS, max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualize ordered elementwise subtraction ``a - b``.

    For example, ``subtract(7, 2)`` displays 5 and ``subtract(2, 7)``
    displays -5. Array operands broadcast to a common shape, while the
    focused expression retains separate left and right source coordinates.
    See :func:`add` for operand types, focus, rendering and preview budgets.
    """
    return _elementwise(
        "subtract", a, b, theme, precision, renderer, focus, max_terms, max_total_terms,
    )



def multiply(
    a, b, theme=None, precision=2, renderer=None, *, focus=None,
    max_terms=DEFAULT_MAX_TERMS, max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualize broadcast-aware elementwise multiplication ``a * b``.

    Each result uses one value from each input. For example,
    ``multiply(array, 2)`` explains doubling each element. Use :func:`matmul`
    to explain a matrix product with products summed over an inner axis.
    See :func:`add` for operand types, focus, rendering and preview budgets.
    """
    return _elementwise(
        "multiply", a, b, theme, precision, renderer, focus, max_terms, max_total_terms,
    )



def divide(
    a, b, theme=None, precision=2, renderer=None, *, focus=None,
    max_terms=DEFAULT_MAX_TERMS, max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualize ordered true division ``a / b`` with broadcasting.

    For example, ``divide(array, 2)`` halves each element. A zero denominator
    raises ZeroDivisionError, including zero divided by zero. This follows
    Python scalar division rather than a backend's infinity or NaN rules.
    Shape tuples generate values starting at zero, so a shape-only denominator
    can trigger the same error. Supply actual nonzero values when appropriate.
    See :func:`add` for operand types, focus, rendering and preview budgets.
    """
    return _elementwise(
        "divide", a, b, theme, precision, renderer, focus, max_terms, max_total_terms,
    )
