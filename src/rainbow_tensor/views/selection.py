"""Visualize value-dependent choices with explicit structural candidate origins."""

from ..evaluation import DEFAULT_MAX_TERMS, DEFAULT_MAX_TOTAL_TERMS
from ..provenance.flow import Flow


def _standalone_result(visual, source):
    """Retain the first input's shape and selection like other standalone views."""
    visual.shape = source.shape
    visual.selected = sorted({
        step.reference.coordinate for step in visual.provenance.steps
        if (step.reference.node_id, step.reference.output_port)
        == (source.node_id, source.output_port)
    })
    return visual


def where(
    condition, x, y, theme=None, precision=2, renderer=None, *, focus=None,
    max_terms=DEFAULT_MAX_TERMS, max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Show how a broadcast condition chooses a value from two alternatives.

    Each input accepts an array, numeric scalar or shape tuple. The condition
    uses Python scalar truth. All three shapes broadcast together. Structural
    traces contain condition, true branch and false branch without reading
    values. Bounded numerical evaluation plans and reads both alternatives,
    then reports the chosen source in ``metadata['evaluated_selections']``.
    Values preserve their Python scalar type without common dtype promotion.
    Budgets cover all previewed inputs and recursive dependencies together.
    """
    flow = Flow()
    operands = tuple(flow.input(value, name=name) for value, name in zip(
        (condition, x, y), ("Condition", "A", "B"),
    ))
    result = flow.where(*operands, name="Y")
    visual = result.visualize(
        focus=focus, theme=theme, precision=precision, renderer=renderer,
        max_terms=max_terms, max_total_terms=max_total_terms,
    )
    return _standalone_result(visual, operands[0])


def _extremum_visual(operation, array, axis, keepdims, theme, precision, renderer,
                     focus, max_terms, max_total_terms):
    """Share recorded candidate rules with standalone extrema displays."""
    flow = Flow()
    source = flow.input(array, name="X")
    result = getattr(flow, operation)(source, axis=axis, keepdims=keepdims, name="Y")
    visual = result.visualize(
        focus=focus, theme=theme, precision=precision, renderer=renderer,
        max_terms=max_terms, max_total_terms=max_total_terms,
    )
    return _standalone_result(visual, source)


def min(
    array, axis=None, theme=None, precision=2, renderer=None, *, keepdims=False,
    focus=None, max_terms=DEFAULT_MAX_TERMS, max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualize a real-valued minimum and its bounded structural candidates.

    Axis accepts None, one integer, or a tuple of distinct axes. An empty axis
    tuple leaves values and shape unchanged. Empty reduced groups raise
    ValueError. Keepdims retains reduced axes with length one. Evaluation
    preserves the first tied value and the first NaN, recording the chosen
    source separately from the value-free trace. Complex values are rejected.
    Budgets cover all previewed inputs and candidate dependencies together.
    """
    return _extremum_visual(
        "min", array, axis, keepdims, theme, precision, renderer,
        focus, max_terms, max_total_terms,
    )


def max(
    array, axis=None, theme=None, precision=2, renderer=None, *, keepdims=False,
    focus=None, max_terms=DEFAULT_MAX_TERMS, max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualize a maximum with the same candidate and first-tie rules as min."""
    return _extremum_visual(
        "max", array, axis, keepdims, theme, precision, renderer,
        focus, max_terms, max_total_terms,
    )


def argmin(
    array, axis=None, theme=None, precision=2, renderer=None, *, keepdims=False,
    focus=None, max_terms=DEFAULT_MAX_TERMS, max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualize the first minimum's position and original source coordinate.

    Axis accepts None or one integer. None returns a flattened row-major
    position. Keepdims retains reduced axes with length one. The first equal
    extremum wins, and the first NaN wins if any candidate is NaN. Empty
    reduced groups and complex values are rejected. Structural traces list
    all candidate dependencies within their limits without reading values.
    """
    return _extremum_visual(
        "argmin", array, axis, keepdims, theme, precision, renderer,
        focus, max_terms, max_total_terms,
    )


def argmax(
    array, axis=None, theme=None, precision=2, renderer=None, *, keepdims=False,
    focus=None, max_terms=DEFAULT_MAX_TERMS, max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualize the first maximum's position using the same rules as argmin."""
    return _extremum_visual(
        "argmax", array, axis, keepdims, theme, precision, renderer,
        focus, max_terms, max_total_terms,
    )
