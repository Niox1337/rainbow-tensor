"""Build one bounded numerical snapshot for a contribution walkthrough."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..ops.elementwise import evaluate_binary
from ..shape import _as_integer
from .model import ElementRef, TrackedTensor
from .query import ProvenanceTrace, ValueBudgetExceeded, evaluate_values


@dataclass(frozen=True, slots=True)
class LessonSnapshot:
    """A selected occurrence and term within one immutable trace revision.

    Occurrence numbers index ``trace.steps`` and must not be reused after the
    output focus or structural limits change. Equal element references can
    occupy different occurrence numbers. ``subtotal`` is the sum of available
    products through the selected term, before a mean's division. It is never
    the total of omitted terms. Numerical values are absent when planning
    exceeds a budget, even though the structural selection remains available.
    """

    trace: ProvenanceTrace
    occurrence: int | None
    term: int | None
    factor_occurrences: tuple[int, ...]
    values: Mapping[ElementRef, object]
    evaluation: Mapping[str, object]
    factor_values: tuple
    term_value: object | None
    subtotal: object | None
    output_value: object | None
    binary_operator: str | None

    @property
    def step(self):
        """Return the selected occurrence, or None for an empty output."""
        return None if self.occurrence is None else self.trace.steps[self.occurrence]

    @property
    def available_terms(self):
        """Count selectable terms without treating an omitted prefix as complete."""
        if self.step is None:
            return 0
        if self.binary_operator is not None:
            return int(bool(getattr(self.step, "operands", ())))
        return len(self.step.terms)

    @property
    def numeric_complete(self):
        """Whether the shared evaluation plan completed without a budget failure."""
        return self.evaluation["status"] == "evaluated"


def _position(value, size, name):
    """Validate an occurrence or term number without accepting booleans."""
    result = _as_integer(value, name)
    if not 0 <= result < size:
        raise IndexError(f"{name} {result} is outside the available range of {size}")
    return result


def _product(values):
    """Multiply a nonempty factor group without converting scalar values."""
    result = values[0]
    for value in values[1:]:
        result = result * value
    return result


def build_lesson(
    node, focus=None, *, occurrence=0, term=None, preview_references=(),
    max_depth=6, max_nodes=80, max_edges=120, max_terms=10_000,
    max_total_terms=100_000,
):
    """Plan trace values and optional preview cells in one numerical request.

    ``preview_references`` lets a renderer share this request rather than read
    arrays again for a figure. Structural limits bound the occurrence tree.
    Numerical limits cover the complete value of the selected output and all
    requested previews, including recursive factors and input reads. Changing
    the selected term never raises either budget. Input values are reread for
    each snapshot, and a planning failure leaves every numerical field absent.
    """
    if not isinstance(node, TrackedTensor):
        raise TypeError("a walkthrough lesson requires a tracked tensor")
    trace = node.trace(focus, max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges)
    if not trace.steps:
        if occurrence != 0 or isinstance(occurrence, bool) or term is not None:
            raise IndexError("an empty output has no occurrence or term")
        selected = None
        step = None
    else:
        selected = _position(occurrence, len(trace.steps), "occurrence")
        step = trace.steps[selected]
    binary = getattr(step, "binary_operator", None)
    groups = () if step is None else step.terms
    if binary is not None:
        operands = getattr(step, "operands", ())
        groups = (operands,) if operands else ()
    position = 0 if term is None and groups else term
    if position is not None:
        position = _position(position, len(groups), "term")
    factors = () if position is None else groups[position]
    references = tuple(item.reference for item in trace.steps) + tuple(preview_references)
    try:
        values, evaluation = evaluate_values(
            node.flow, references, max_terms=max_terms, max_total_terms=max_total_terms,
        )
    except ValueBudgetExceeded as error:
        values = {}
        evaluation = {
            "status": "skipped", "reason": error.reason, "total_terms": error.total_terms,
            "max_terms": max_terms, "max_total_terms": max_total_terms,
            "scope": "recursive_terms_factors_and_inputs",
        }
    factor_values = ()
    term_value = subtotal = output_value = None
    if evaluation["status"] == "evaluated" and step is not None:
        output_value = values[step.reference]
        factor_values = tuple(values[trace.steps[child].reference] for child in factors)
        if position is not None:
            if binary is not None:
                term_value = evaluate_binary(binary, *factor_values)
            else:
                term_value = _product(factor_values)
                subtotal = sum(
                    _product(tuple(values[trace.steps[child].reference] for child in group))
                    for group in groups[:position + 1]
                )
        elif not step.term_count and step.operation != "input":
            subtotal = 0
    return LessonSnapshot(
        trace, selected, position, tuple(factors), MappingProxyType(values),
        MappingProxyType(evaluation), factor_values, term_value, subtotal, output_value, binary,
    )
