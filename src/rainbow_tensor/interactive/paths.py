"""Preserve ordered occurrence paths without evaluating tensor values."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalculationPath:
    """One occurrence and the role of its edge from its parent expression.

    Occurrence and parent numbers index the current trace, never a global node
    registry. Equal source coordinates can have different paths. ``term`` and
    ``factor`` are zero-based positions where those roles apply. Candidate
    edges describe structural alternatives, not selected numerical sources.
    """

    occurrence: int
    parent: int | None
    role: str
    term: int | None = None
    factor: int | None = None


def calculation_paths(trace):
    """Label each reached edge while retaining repeated source occurrences."""
    if not trace.steps:
        return ()
    paths = {0: CalculationPath(0, None, "result")}
    for parent, step in enumerate(trace.steps):
        selection = getattr(step, "selection_operator", None)
        if selection:
            for position, child in enumerate(step.candidates):
                role = ("condition", "when_true", "when_false")[position] if (
                    selection == "where"
                ) else "candidate"
                paths[child] = CalculationPath(child, parent, role, position)
        elif step.binary_operator:
            roles = ("numerator", "denominator") if step.binary_operator == "divide" else (
                "left", "right"
            )
            for position, child in enumerate(step.operands):
                paths[child] = CalculationPath(child, parent, roles[position], 0, position)
        else:
            for term, group in enumerate(step.terms):
                for factor, child in enumerate(group):
                    if len(group) > 1:
                        role = "factor"
                    elif step.operation in {"sum", "mean", "matmul", "einsum"}:
                        role = "summand"
                    else:
                        role = "broadcast" if step.operation == "broadcast" else "source"
                    paths[child] = CalculationPath(child, parent, role, term, factor)
    return tuple(paths[index] for index in range(len(trace.steps)))
