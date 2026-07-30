"""Immutable, bounded descriptions of how one tensor output is formed.

Tracing records coordinates rather than values. Building or inspecting a trace
never asks an array backend to evaluate any source or output element.
"""

from dataclasses import dataclass
from itertools import islice
from operator import index as integer_index


@dataclass(frozen=True, slots=True)
class OperandRef:
    """One factor in an output term, identified by operand and source coordinate."""

    operand: int
    coordinate: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OutputTrace:
    """Describe the sum of products that produces one normalized output coordinate.

    Each item in ``terms`` is an ordered tuple of ``OperandRef`` factors to
    multiply. Add those products and divide by ``divisor`` to obtain the output.
    Sum and mean have one factor per term. Mean uses its reduction length as
    the divisor, while the other operations use one. Reused broadcast factors
    remain present in every term that uses them.

    ``term_count`` counts the full expression. Views store its first eight terms
    at most, so ``complete`` is false when later terms have been omitted. The
    ordered tuples and frozen references can be safely inspected or exported
    without retaining backend arrays or mutable intermediate values.
    """

    operation: str
    output_coord: tuple[int, ...]
    term_count: int
    terms: tuple[tuple[OperandRef, ...], ...]
    complete: bool
    divisor: int = 1


def _normalize_focus(focus, result_shape):
    """Validate an optional output coordinate before any backend value reads."""
    if focus is None:
        return (0,) * len(result_shape)
    if not isinstance(focus, tuple):
        raise TypeError("focus must be a tuple of integer output coordinates")
    if len(focus) != len(result_shape):
        raise ValueError(
            f"focus has {len(focus)} coordinates for output rank {len(result_shape)}"
        )
    resolved = []
    for axis, (value, size) in enumerate(zip(focus, result_shape)):
        if isinstance(value, bool) or type(value).__name__.startswith("bool"):
            raise TypeError(f"focus axis {axis} must be an integer, not a boolean")
        try:
            coordinate = integer_index(value)
        except TypeError:
            raise TypeError(f"focus axis {axis} must be an integer, got {value!r}") from None
        coordinate = coordinate + size if coordinate < 0 else coordinate
        if not 0 <= coordinate < size:
            raise IndexError(f"focus axis {axis}: index {value} is out of range for size {size}")
        resolved.append(coordinate)
    return tuple(resolved)


def _build_trace(operation, output_coord, term_count, terms, *, divisor=1):
    """Keep only a bounded ordered term prefix without consuming the remainder."""
    stored = tuple(
        tuple(OperandRef(operand, tuple(coord)) for operand, coord in enumerate(term))
        for term in islice(terms, 8)
    )
    return OutputTrace(
        operation, output_coord, term_count, stored, len(stored) == term_count, divisor
    )


def _trace_explanation(trace):
    """Explain a focus using source coordinates without reading numeric values."""
    plural = "s" if trace.term_count != 1 else ""
    heading = f"Focus: output {trace.output_coord} uses {trace.term_count} term{plural}."
    if not trace.complete:
        omitted = trace.term_count - len(trace.terms)
        heading += f" Showing the first {len(trace.terms)} ({omitted} omitted)."

    def subscript(coordinate):
        return ", ".join(map(str, coordinate)) if coordinate else "()"

    def operand_name(operand):
        if trace.operation == "matmul":
            return ("A", "B")[operand]
        if trace.operation in ("sum", "mean"):
            return "source"
        return f"operand {operand}"

    expression = " + ".join(
        " * ".join(
            f"{operand_name(ref.operand)}[{subscript(ref.coordinate)}]" for ref in term
        )
        for term in trace.terms
    )
    if not trace.complete:
        expression += " + ..."
    if trace.divisor != 1:
        expression = f"({expression}) / {trace.divisor}"
    return [heading, f"output[{subscript(trace.output_coord)}] = {expression}"]
