"""Visualize value-dependent choices with explicit structural candidate origins."""

from ..evaluation import DEFAULT_MAX_TERMS, DEFAULT_MAX_TOTAL_TERMS
from ..provenance.flow import Flow


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
    return result.visualize(
        focus=focus, theme=theme, precision=precision, renderer=renderer,
        max_terms=max_terms, max_total_terms=max_total_terms,
    )
