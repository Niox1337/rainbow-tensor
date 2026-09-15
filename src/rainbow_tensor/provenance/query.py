"""Bounded ancestry traversal and per-refresh evaluation of recorded recipes."""

from collections import Counter
from dataclasses import dataclass, replace

from ..evaluation import validate_max_terms
from ..shape import _as_integer, extract_shape
from ..tracing import _normalize_focus
from ..visual import _source_value
from .model import ElementRef


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One occurrence in an expression, with ordered child occurrence numbers.

    The same element can appear in several steps. Terms preserve multiplication
    groups and division belongs to this step, so nested means stay explicit.
    """

    reference: ElementRef
    operation: str
    depth: int
    term_count: int
    divisor: int
    terms: tuple[tuple[int, ...], ...] = ()
    complete: bool = False


@dataclass(frozen=True, slots=True)
class RootContribution:
    """Count reached paths to an input element, not numerical coefficients."""

    reference: ElementRef
    count: int


@dataclass(frozen=True, slots=True)
class ProvenanceTrace:
    """A finite expression tree whose omitted work is always explicit.

    The first step is the focused output. Child numbers index steps in this
    tuple. Root counts are exact only when complete is true. Otherwise they
    count the visible portion, without guessing hidden contributions.
    """

    root: ElementRef | None
    steps: tuple[TraceStep, ...]
    roots: tuple[RootContribution, ...]
    complete: bool
    truncated_reasons: tuple[str, ...]
    edge_count: int


def _limit(value, name, *, zero=False):
    """Validate finite structural limits before traversing a recipe."""
    result = _as_integer(value, name)
    if result < (0 if zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return result


def _node(flow, reference):
    return flow._nodes[(reference.node_id, reference.output_port)]


def trace_output(node, focus=None, *, max_depth=6, max_nodes=80, max_edges=120):
    """Expand an occurrence tree breadth first within independent hard limits."""
    depth_limit = _limit(max_depth, "max_depth", zero=True)
    node_limit = _limit(max_nodes, "max_nodes")
    edge_limit = _limit(max_edges, "max_edges")
    coordinate = _normalize_focus(focus, node.shape)
    if coordinate is None:
        return ProvenanceTrace(None, (), (), True, (), 0)
    root = node.ref(coordinate)
    steps = [TraceStep(root, node.operation, 0, node.term_count, node.divisor)]
    roots = Counter()
    reasons = set()
    edge_count = 0
    position = 0
    while position < len(steps):
        step = steps[position]
        current = _node(node.flow, step.reference)
        if current.operation == "input":
            roots[step.reference] += 1
            steps[position] = replace(step, complete=True)
        elif not current.term_count:
            steps[position] = replace(step, complete=True)
        elif step.depth >= depth_limit:
            reasons.add("max_depth")
        else:
            terms = []
            finished = True
            for term in current.term_factory(step.reference.coordinate):
                if len(term) > min(node_limit - len(steps), edge_limit - edge_count):
                    if len(term) > node_limit - len(steps):
                        reasons.add("max_nodes")
                    if len(term) > edge_limit - edge_count:
                        reasons.add("max_edges")
                    finished = False
                    break
                children = []
                for reference in term:
                    if len(steps) >= node_limit or edge_count >= edge_limit:
                        if len(steps) >= node_limit:
                            reasons.add("max_nodes")
                        if edge_count >= edge_limit:
                            reasons.add("max_edges")
                        finished = False
                        break
                    child = _node(node.flow, reference)
                    children.append(len(steps))
                    steps.append(TraceStep(
                        reference, child.operation, step.depth + 1,
                        child.term_count, child.divisor,
                    ))
                    edge_count += 1
                if children:
                    terms.append(tuple(children))
                if not finished:
                    break
            steps[position] = replace(step, terms=tuple(terms), complete=finished)
        position += 1
    return ProvenanceTrace(
        root, tuple(steps), tuple(RootContribution(ref, count) for ref, count in roots.items()),
        not reasons, tuple(sorted(reasons)), edge_count,
    )


class ValueBudgetExceeded(ValueError):
    """Signal that planning stopped before any backend values were read."""

    def __init__(self, reason, total_terms):
        self.reason = reason
        self.total_terms = total_terms
        super().__init__(f"value evaluation exceeds {reason}")


def evaluate_values(flow, references, *, max_terms=10_000, max_total_terms=100_000):
    """Plan a dependency DAG, then evaluate it in dependency order.

    The total budget counts terms and factor references, including input reads.
    This bounds work even for many-operand products and long identity chains.
    Shared elements cost once per call. A depth limit of 64 protects recursive
    recipes, while graph tracing has its own independent display limits.
    """
    per_limit = validate_max_terms(max_terms)
    total_limit = validate_max_terms(max_total_terms, "max_total_terms")
    planned = {}
    order = []
    total = 0

    def charge(amount):
        nonlocal total
        total += amount
        if total_limit is not None and total > total_limit:
            raise ValueBudgetExceeded("max_total_terms", total)

    def plan(reference, depth):
        if reference in planned:
            return
        if depth > 64:
            raise ValueBudgetExceeded("max_value_depth", total)
        node = _node(flow, reference)
        if node.operation == "input":
            if extract_shape(node.source) != node.shape:
                raise ValueError(f"input {node.name!r} changed shape after recording")
            charge(1)
            terms = ()
        else:
            if per_limit is not None and node.term_count > per_limit:
                raise ValueBudgetExceeded("max_terms", total)
            charge(node.term_count)
            gathered = []
            for term in node.term_factory(reference.coordinate):
                factors = tuple(term)
                charge(len(factors))
                for factor in factors:
                    plan(factor, depth + 1)
                gathered.append(factors)
            terms = tuple(gathered)
        planned[reference] = terms
        order.append(reference)

    for reference in references:
        plan(reference, 0)
    values = {}
    for reference in order:
        node = _node(flow, reference)
        if node.operation == "input":
            value = _source_value(node.source, node.shape)(reference.coordinate)
        elif node.operation not in {"sum", "mean", "matmul", "einsum"}:
            value = values[planned[reference][0][0]]
        elif node.divisor == 0:
            value = float("nan")
        else:
            products = []
            for term in planned[reference]:
                product = values[term[0]]
                for factor in term[1:]:
                    product = product * values[factor]
                products.append(product)
            value = sum(products)
            if node.operation == "mean":
                value = value / node.divisor
        values[reference] = value
    return values, {
        "status": "evaluated", "reason": None, "total_terms": total,
        "max_terms": per_limit, "max_total_terms": total_limit,
        "scope": "recursive_terms_factors_and_inputs",
    }
