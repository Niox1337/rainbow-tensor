"""Shared single-element focus for views with an exact output-to-input mapping."""

from ..tracing import OperandRef, OutputTrace, _normalize_focus, _trace_explanation
from ..visual import _visual

FIRST_OUTPUT = object()


def mapped_trace(operation, focus, result_shape, origin):
    """Describe one identity mapping without reading an array or enumerating a result.

    ``None`` preserves the static display. The private ``FIRST_OUTPUT`` marker
    lets an explorer select the first element while preparing and rendering the
    operation only once. Empty outputs have no first element and no trace.
    ``origin`` returns the original operand number and its source coordinate.
    """
    if focus is None:
        return None
    coordinate = _normalize_focus(None if focus is FIRST_OUTPUT else focus, result_shape)
    if coordinate is None:
        return None
    operand, source = origin(coordinate)
    reference = OperandRef(operand, tuple(source))
    return OutputTrace(operation, coordinate, 1, ((reference,),), True)


def render_mapped(
    panels,
    connectors,
    explanation,
    theme,
    precision,
    renderer,
    shape,
    result,
    trace,
    *,
    source_panels=None,
    output_panels=None,
    focus_operand=0,
):
    """Pin a mapped pair, render once, and retain output panel identities.

    Source panel positions are indexed by operand identity, which may differ
    from the position of the one factor stored in an identity trace. Broadcast
    has two output panels and uses ``focus_operand`` to select its output port.
    Existing themes and cell tints remain attached to their panels.
    """
    output_panels = tuple(output_panels or (len(panels) - 1,))
    source_panels = tuple(source_panels or range(len(panels) - 1))
    output_panel = output_panels[focus_operand if len(output_panels) > 1 else 0]
    selected = None
    if trace is not None:
        reference = trace.terms[0][0]
        selected = [reference.coordinate]
        panels[source_panels[reference.operand]]["selected"] = selected
        panels[output_panel]["selected"] = [trace.output_coord]
        explanation = [*explanation, *_trace_explanation(trace)]
    content = renderer.render_panels(
        panels=panels,
        connectors=connectors,
        explanation=explanation,
        theme=theme,
        precision=precision,
    )
    visual = _visual(
        content, shape, renderer, selected=selected, result=result, explanation=explanation
    )
    visual.trace = trace
    visual.metadata.update(
        output_panel_indices=output_panels,
        focused_output_panel=output_panel if trace is not None else None,
        trace_in_explanation=trace is not None,
    )
    if len(output_panels) > 1:
        visual.metadata["focus_operand"] = focus_operand
    return visual
