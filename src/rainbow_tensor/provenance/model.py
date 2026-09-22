"""Immutable coordinate recipes for explicitly recorded tensor operations."""

from dataclasses import dataclass, field
from typing import Any, Callable

from ..tracing import _normalize_focus


@dataclass(frozen=True, slots=True)
class ElementRef:
    """Identify an element without retaining its value or an array view.

    Output ports distinguish the separate results of one broadcast operation.
    References are meaningful within the Flow that created them.
    """

    node_id: str
    output_port: int
    coordinate: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class TrackedTensor:
    """A tensor shape and a lazy recipe, separate from its rendered preview.

    Operation parameters are captured when the node is created. Input array
    values stay live and are read again on each evaluation or visualization.
    This object does not execute a backend kernel or intercept array operators.
    Use methods on its Flow to record each operation explicitly.
    """

    flow: Any = field(repr=False)
    node_id: str
    output_port: int
    name: str
    operation: str
    shape: tuple[int, ...]
    inputs: tuple
    term_count: int
    term_factory: Callable = field(repr=False)
    divisor: int = 1
    source: Any = field(default=None, repr=False)
    origin: Callable | None = field(default=None, repr=False)
    numeric_sources: frozenset[str] = field(init=False)

    def __post_init__(self):
        """Carry value-source semantics through truncation without walking ancestors."""
        if self.operation == "input":
            kind = "array_values" if hasattr(self.source, "shape") else "generated_values"
            kinds = frozenset((kind,))
        elif self.operation == "broadcast":
            kinds = self.inputs[self.output_port].numeric_sources
        else:
            kinds = frozenset(kind for node in self.inputs for kind in node.numeric_sources)
        object.__setattr__(self, "numeric_sources", kinds)

    def ref(self, coordinate):
        """Return a normalized reference, rejecting empty or invalid positions."""
        normalized = _normalize_focus(coordinate, self.shape)
        if normalized is None:
            raise IndexError("an empty tensor has no element references")
        return ElementRef(self.node_id, self.output_port, normalized)

    def trace(self, focus=None, *, max_depth=6, max_nodes=80, max_edges=120):
        """Follow a bounded tree of contributions without reading array values.

        Duplicate paths remain separate occurrences. An incomplete trace lists
        its limits and reports only the root occurrences that were reached.
        Empty outputs produce an empty trace with no fabricated coordinate.
        """
        from .query import trace_output

        return trace_output(
            self, focus, max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges
        )

    def value(self, coordinate=None, *, max_terms=10_000, max_total_terms=100_000):
        """Evaluate one element using Python scalars after planning all work.

        Each operation keeps its own sum, product, and mean division. Shared
        intermediate elements are evaluated once per call. Exceeding a budget
        raises ValueBudgetExceeded before any input values are read.
        """
        from .query import evaluate_values

        reference = self.ref(coordinate)
        values, _ = evaluate_values(
            self.flow, (reference,), max_terms=max_terms, max_total_terms=max_total_terms
        )
        return values[reference]

    def visualize(
        self, focus=None, theme=None, precision=2, renderer=None, *,
        max_depth=6, max_nodes=80, max_edges=120, max_panels=8,
        max_terms=10_000, max_total_terms=100_000,
    ):
        """Render a selected output with its bounded ancestry and local equations.

        The result remains a TensorVisual. Its provenance attribute contains
        the structural trace and its trace attribute describes the final step.
        Unplanned or over-budget values display as question marks.
        """
        from .view import render_flow

        return render_flow(
            self, focus=focus, theme=theme, precision=precision, renderer=renderer,
            max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges,
            max_panels=max_panels, max_terms=max_terms, max_total_terms=max_total_terms,
        )
