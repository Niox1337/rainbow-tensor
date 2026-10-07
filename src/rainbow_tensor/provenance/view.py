"""Render a recorded operation chain without materializing intermediate arrays."""

from ..explanations import t
from ..layout import build_layout
from ..numerics import numeric_explanation
from ..ops.elementwise import BINARY_SYMBOLS
from ..ops.selection import EXTREMA
from ..renderers import resolve_renderer
from ..theme import resolve_theme
from ..tracing import BinaryExpression, OperandRef, OutputTrace, SelectionExpression
from ..visual import _preview_explanation, _shape_caption_parts, _visual
from .query import (
    ValueBudgetExceeded,
    _limit,
    _node,
    evaluate_values,
    serialize_selected_sources,
)


def _label(flow, reference):
    """Name a scalar or indexed element consistently in local equations."""
    coordinate = ", ".join(map(str, reference.coordinate)) or "()"
    return f"{_node(flow, reference).name}[{coordinate}]"


def _equations(flow, trace):
    """Show each reached element once while retaining repeated factors in terms."""
    lines = []
    seen = set()
    for step in trace.steps:
        if step.reference in seen or step.operation == "input":
            continue
        seen.add(step.reference)
        if step.selection_operator is not None:
            candidates = [_label(flow, trace.steps[child].reference) for child in step.candidates]
            if not step.complete:
                candidates.append("...")
            lines.append(t(
                "flow.equation", output=_label(flow, step.reference), operation=step.operation,
                expression=f"{step.selection_operator}({', '.join(candidates)})",
            ))
            lines.append(t("selection.candidates", count=(
                3 if step.selection_operator == "where" else step.term_count
            )))
            continue
        if step.binary_operator is not None:
            expression = "..."
            if step.operands:
                left, right = (trace.steps[child].reference for child in step.operands)
                expression = (
                    f"{_label(flow, left)} {BINARY_SYMBOLS[step.binary_operator]} "
                    f"{_label(flow, right)}"
                )
            lines.append(t(
                "flow.equation", output=_label(flow, step.reference),
                operation=step.operation, expression=expression,
            ))
            continue
        terms = [
            " * ".join(_label(flow, trace.steps[child].reference) for child in term)
            for term in step.terms
        ]
        expression = " + ".join(terms)
        if not step.complete:
            expression = (expression + " + ...") if expression else "..."
        elif not terms:
            expression = t("trace.empty_mean") if step.divisor == 0 else "0"
        if step.divisor not in (0, 1):
            expression = f"({expression}) / {step.divisor}"
        lines.append(t(
            "flow.equation", output=_label(flow, step.reference),
            operation=step.operation, expression=expression,
        ))
    return lines


def _immediate_trace(node, provenance):
    """Retain the final operation's local trace for the focus explorer."""
    if provenance.root is None:
        return None
    coordinate = provenance.root.coordinate
    if node.selection_operator is not None:
        step = provenance.steps[0]
        candidates = tuple(
            OperandRef(position if node.selection_operator == "where" else 0,
                       provenance.steps[child].reference.coordinate)
            for position, child in enumerate(step.candidates[:8])
        )
        count = 3 if node.selection_operator == "where" else node.term_count
        return OutputTrace(
            node.operation, coordinate, node.term_count, (),
            step.complete and len(candidates) == count,
            expression=SelectionExpression(node.selection_operator, candidates, count),
        )
    if node.binary_operator is not None:
        step = provenance.steps[0]
        expression = None
        if step.operands:
            expression = BinaryExpression(node.binary_operator, tuple(
                OperandRef(operand, provenance.steps[child].reference.coordinate)
                for operand, child in enumerate(step.operands)
            ))
        return OutputTrace(node.operation, coordinate, 1, (), step.complete,
                           expression=expression)
    terms = []
    for children in provenance.steps[0].terms[:8]:
        term = tuple(provenance.steps[child].reference for child in children)
        factors = []
        for position, reference in enumerate(term):
            if node.operation in {"matmul", "einsum"}:
                operand = position
            elif node.origin is not None:
                operand = node.origin(coordinate)[0]
            else:
                operand = next(
                    i for i, source in enumerate(node.inputs)
                    if (source.node_id, source.output_port)
                    == (reference.node_id, reference.output_port)
                )
            factors.append(OperandRef(operand, reference.coordinate))
        terms.append(tuple(factors))
    return OutputTrace(
        node.operation, coordinate, node.term_count, tuple(terms),
        len(terms) == node.term_count, node.divisor,
    )


def selected_source_explanation(flow, decision):
    """Explain an evaluated choice separately from the structural candidate list."""
    return t(
        "selection.chosen", output=_label(flow, decision.output),
        source=_label(flow, decision.source),
        reason=t(f"selection.reason.{decision.reason}"),
    )


def render_flow(
    node, *, focus=None, theme=None, precision=2, renderer=None, max_depth=6,
    max_nodes=80, max_edges=120, max_panels=8, max_terms=10_000,
    max_total_terms=100_000,
):
    """Render selected ancestry with a shared budget for every panel's values.

    Structural traces do not read arrays. Only coordinates admitted by the
    default bounded layouts are planned for numerical work. Custom renderers
    receive question marks for additional positions rather than growing work.
    """
    panel_limit = _limit(max_panels, "max_panels")
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    trace = node.trace(focus, max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges)
    selections = {}
    reached = {}
    for step in trace.steps:
        reference = step.reference
        key = (reference.node_id, reference.output_port)
        reached[key] = _node(node.flow, reference)
        selections.setdefault(key, set()).add(reference.coordinate)
    final_key = (node.node_id, node.output_port)
    reached[final_key] = node
    # Registration order is topological and includes each output port separately.
    ordered = sorted(
        reached.values(), key=lambda source: (int(source.node_id[1:]), source.output_port)
    )
    omitted_panels = max(0, len(ordered) - panel_limit)
    displayed = ordered[-panel_limit:]
    planned = []
    panels = []
    for source in displayed:
        key = (source.node_id, source.output_port)
        selected = sorted(selections.get(key, ()))
        layout = build_layout(source.shape, selected=selected, theme=theme)
        planned.extend(source.ref(cell.coord) for cell in layout.cells if not cell.ellipsis)
        panels.append({
            "shape": source.shape, "selected": selected,
            "caption_parts": _shape_caption_parts(source.name, source.shape, theme),
        })
    try:
        values, evaluation = evaluate_values(
            node.flow, planned, max_terms=max_terms, max_total_terms=max_total_terms
        )
    except ValueBudgetExceeded as exc:
        values = {}
        evaluation = {
            "status": "skipped", "reason": exc.reason, "total_terms": exc.total_terms,
            "max_terms": max_terms, "max_total_terms": max_total_terms,
            "scope": "recursive_terms_factors_and_inputs",
        }
    for source, panel in zip(displayed, panels):
        panel["value_fn"] = lambda coord, source=source: values.get(source.ref(coord), "?")
    explanation = [t("flow.heading", name=node.name)]
    if any(step.selection_operator == "where" for step in trace.steps):
        explanation.append(t("selection.where_eager"))
    if any(step.selection_operator in EXTREMA for step in trace.steps):
        explanation.append(t("selection.extrema_rule"))
    reached_references = {step.reference for step in trace.steps}
    explanation.extend(
        selected_source_explanation(node.flow, decision)
        for decision in evaluation.get("selected_sources", ())
        if decision.output in reached_references
    )
    if any(step.binary_operator == "divide" for step in trace.steps):
        explanation.append(t("elementwise.divide_zero"))
    if trace.root is None:
        explanation.append(t("trace.empty"))
    else:
        explanation.extend(_equations(node.flow, trace))
    if not trace.complete:
        explanation.append(t("flow.truncated", limits=", ".join(trace.truncated_reasons)))
    if trace.roots:
        explanation.append(t("flow.roots" if trace.complete else "flow.roots_partial"))
        explanation.extend(
            t("flow.root_count", source=_label(node.flow, root.reference), count=root.count)
            for root in trace.roots
        )
    if omitted_panels:
        explanation.append(t("flow.panels_omitted", count=omitted_panels))
    if evaluation["status"] == "skipped":
        explanation.append(t("flow.values_skipped", limit=evaluation["reason"]))
    explanation.extend(_preview_explanation([source.shape for source in displayed], theme))
    semantics = {
        "arithmetic": "python_scalar", "scope": "recorded_operations",
        "operand_sources": sorted(node.numeric_sources),
    }
    explanation.extend(numeric_explanation({**semantics, "operand_sources": [
        kind for kind in semantics["operand_sources"] if kind == "array_values"
    ]}))
    if "generated_values" in node.numeric_sources:
        explanation.append(t("flow.generated"))
    content = renderer.render_panels(
        panels=panels, connectors=[""] * (len(panels) - 1), explanation=explanation,
        theme=theme, precision=precision,
    )
    visual = _visual(
        content, node.shape, renderer, selected=panels[-1]["selected"],
        result=node.shape, explanation=explanation,
    )
    visual.trace = _immediate_trace(node, trace)
    visual.provenance = trace
    visual.metadata.update(
        output_panel_indices=(len(panels) - 1,),
        focused_output_panel=len(panels) - 1 if trace.root else None,
        trace_in_explanation=True,
        value_evaluation={
            key: value for key, value in evaluation.items() if key != "selected_sources"
        },
        evaluated_selections=serialize_selected_sources(evaluation),
        numeric_semantics=semantics,
        provenance={
            "complete": trace.complete, "truncated_reasons": trace.truncated_reasons,
            "panel_nodes": tuple((source.node_id, source.output_port) for source in displayed),
            "omitted_panels": omitted_panels,
        },
    )
    return visual
