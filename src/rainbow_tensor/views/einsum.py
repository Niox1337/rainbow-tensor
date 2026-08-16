"""einsum view.

The einsum view colours every subscript label by its role (free, shared, or
contracted) and keeps that colour consistent across operands and the output.
It is self contained apart from the shared rendering substrate in
:mod:`rainbow_tensor.visual`, and its own caption builder, so it does not lean
on the shared caption helper the other families use.
"""

from ..evaluation import (
    DEFAULT_MAX_TERMS,
    DEFAULT_MAX_TOTAL_TERMS,
    budgeted_values,
    evaluation_explanation,
    evaluation_plan,
)
from ..explanations import t
from ..numerics import numeric_explanation, numeric_semantics
from ..ops import (
    einsum_contracted_labels,
    einsum_result_shape,
    einsum_selected_coords,
    parse_einsum_subscripts,
)
from ..ops.einsum import einsum_source_terms, einsum_term_count
from ..renderers import resolve_renderer
from ..shape import extract_shape, format_shape
from ..theme import resolve_theme
from ..tracing import _build_trace, _normalize_focus, _trace_explanation
from ..visual import _preview_explanation, _source_value, _value_fn_for, _visual

# Role colour families, chosen so free, shared, and contracted labels read as
# three distinct groups while every label keeps one colour everywhere it appears.
_EINSUM_FREE = ("#2563eb", "#0891b2", "#4f46e5", "#0d9488", "#0284c7")
_EINSUM_SHARED = ("#db2777", "#9333ea", "#c026d3", "#e11d48")
_EINSUM_CONTRACTED = ("#ea580c", "#ca8a04", "#dc2626", "#b45309")


def _einsum_roles(input_axes, output_axes):
    """Classify each einsum label as free, shared, or contracted.

    A label that is not in the output is contracted. Otherwise a label that
    appears in more than one operand is shared, and the rest are free. Each
    label gets exactly one role, so the three roles partition the labels.
    """
    output = set(output_axes)
    operand_count = {}
    for labels in input_axes:
        for label in set(labels):
            operand_count[label] = operand_count.get(label, 0) + 1
    roles = {}
    for labels in (*input_axes, output_axes):
        for label in labels:
            if label in roles:
                continue
            if label not in output:
                roles[label] = "contracted"
            elif operand_count.get(label, 0) >= 2:
                roles[label] = "shared"
            else:
                roles[label] = "free"
    return roles


def _einsum_label_colors(input_axes, output_axes):
    """Map every einsum label to one colour drawn from its role family.

    The colour is stable across every operand and the output, so a label keeps
    one colour in every caption and figure, and the three roles stay visually
    distinct.
    """
    roles = _einsum_roles(input_axes, output_axes)
    ramps = {"free": _EINSUM_FREE, "shared": _EINSUM_SHARED, "contracted": _EINSUM_CONTRACTED}
    counters = {"free": 0, "shared": 0, "contracted": 0}
    colors = {}
    for labels in (*input_axes, output_axes):
        for label in labels:
            if label in colors:
                continue
            role = roles[label]
            colors[label] = ramps[role][counters[role] % len(ramps[role])]
            counters[role] += 1
    return colors


def _einsum_theme(labels, theme, label_colors):
    """Return a theme whose axis ramp follows the subscript labels."""
    if not labels:
        return theme
    return theme.variant(axis_colors=tuple(label_colors[label] for label in labels))


def _einsum_leaf_tint(labels, theme, label_colors):
    """Border a panel's leaf cells with the leaf label colour.

    A non-leaf axis shows its label colour as a frame, but the leaf axis is
    drawn as plain cells, so without this the leaf label colour would appear in
    the caption yet never in the figure. Bordering each cell with the leaf label
    colour keeps the figure and the caption consistent for every label.
    """
    if not labels:
        return None
    border = label_colors[labels[-1]]
    fill = theme.surface
    return lambda coord: (fill, border)


def _einsum_caption_parts(name, labels, shape, theme, label_colors):
    """Build a coloured caption for an einsum panel."""
    neutral = theme.heading
    parts = [(f"{name} ", neutral)]
    if labels:
        for label in labels:
            parts.append((label, label_colors[label]))
    else:
        parts.append(("scalar", theme.text_muted))
    parts.append((" (", neutral))
    for axis, dim in enumerate(shape):
        color = label_colors[labels[axis]] if labels else theme.text_muted
        parts.append((str(dim), color))
        if axis < len(shape) - 1:
            parts.append((", ", neutral))
    parts.append((")", neutral))
    return parts


def _einsum_result_value_fn(input_axes, output_axes, shapes, source_values, evaluation):
    """Compute complete visible outputs once, or show unknown values when skipped."""
    @budgeted_values(evaluation)
    def result_value(coord):
        output_coord = () if not output_axes else coord
        total = 0
        for source_coords in einsum_source_terms(input_axes, output_axes, shapes, output_coord):
            term = 1
            for source_coord, value_fn in zip(source_coords, source_values):
                term *= value_fn(source_coord)
            total += term
        return total

    return result_value


def einsum(
    subscripts, *arrays, theme=None, precision=2, renderer=None, focus=None,
    max_terms=DEFAULT_MAX_TERMS,
    max_total_terms=DEFAULT_MAX_TOTAL_TERMS,
):
    """Visualise an einsum expression.

    Each operand is drawn with its subscript labels. Every label keeps one
    colour across all operand captions, operand figures, and the output, drawn
    from a colour family for its role, so free, shared, and contracted labels
    stay visually distinct. The output panel shows the derived free index shape.

    Numerical previews multiply and accumulate input values with Python scalar
    arithmetic, not the backend's einsum kernel. Accumulation dtype, rounding,
    and overflow may differ. Shape tuples supply generated row-major placeholder
    values. ``visual.metadata["numeric_semantics"]`` records this model and each
    operand's value source, even when evaluation is skipped.

    ``focus`` is an output coordinate tuple, with negative coordinates accepted
    and ``()`` required for a scalar output. Supplying it highlights that output
    and its input factors. ``visual.trace`` records the first eight ordered
    terms, including repeated factors introduced by singleton broadcasting.

    ``max_terms`` limits products summed for each output cell, defaulting to
    10,000. Outputs that exceed it display ``?`` without partial evaluation.
    ``max_total_terms`` additionally caps all visible output terms at 100,000.
    If either limit is exceeded, all output values are skipped. Set both limits
    to ``None`` to remove them. Source highlights are sampled
    when values are skipped, while ``visual.trace.term_count`` retains the full
    count. Details are stored in ``visual.metadata["value_evaluation"]``.
    """
    if not arrays:
        raise ValueError("einsum needs at least one operand")

    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    shapes = [extract_shape(array) for array in arrays]
    input_axes, output_axes = parse_einsum_subscripts(subscripts, len(arrays), shapes)
    result = einsum_result_shape(subscripts, shapes)
    focused = _normalize_focus(focus, result)
    term_count = einsum_term_count(input_axes, output_axes, shapes)
    semantics = numeric_semantics(arrays)
    display_result = result or (1,)
    label_colors = _einsum_label_colors(input_axes, output_axes)
    result_theme = _einsum_theme(output_axes, theme, label_colors)
    evaluation = evaluation_plan(
        term_count, max_terms, max_total_terms, display_result,
        [focused or (0,)] if focus is not None else None, result_theme,
    )
    source_values = [_source_value(array, shape) for array, shape in zip(arrays, shapes)]
    trace = _build_trace(
        "einsum", focused, term_count,
        einsum_source_terms(input_axes, output_axes, shapes, focused),
    )
    if evaluation["status"] == "skipped":
        selected = [
            sorted({term[operand].coordinate for term in trace.terms})
            for operand in range(len(arrays))
        ]
    else:
        selected = einsum_selected_coords(input_axes, output_axes, shapes, focused)
    result_value = _einsum_result_value_fn(
        input_axes, output_axes, shapes, source_values, evaluation
    )
    contracted = einsum_contracted_labels(input_axes, output_axes)

    panels = []
    for i, (array, labels, shape) in enumerate(zip(arrays, input_axes, shapes)):
        panels.append(
            {
                "shape": shape,
                "value_fn": _value_fn_for(array),
                "theme": _einsum_theme(labels, theme, label_colors),
                "cell_tint": _einsum_leaf_tint(labels, theme, label_colors),
                "caption_parts": _einsum_caption_parts(
                    f"operand {i}", labels, shape, theme, label_colors
                ),
            }
        )
        if focus is not None or evaluation["status"] == "skipped":
            panels[-1]["selected"] = selected[i]

    panels.append(
        {
            "shape": display_result,
            "value_fn": result_value,
            "theme": result_theme,
            "cell_tint": _einsum_leaf_tint(output_axes, theme, label_colors),
            "caption_parts": _einsum_caption_parts(
                "output", output_axes, display_result, theme, label_colors
            ),
        }
    )
    if focus is not None or evaluation["status"] == "skipped":
        panels[-1]["selected"] = [focused or (0,)]
    connectors = ["*"] * (len(arrays) - 1) + ["->"]
    input_text = ", ".join("".join(labels) or "scalar" for labels in input_axes)
    output_text = "".join(output_axes) or "scalar"
    if contracted:
        contracted_line = t("einsum.contracted", labels=", ".join(contracted))
    else:
        contracted_line = t("einsum.none_contracted")
    explanation = [
        t("einsum.operand_subscripts", subscripts=input_text),
        t("einsum.output_subscript", subscript=output_text),
        contracted_line,
        t("common.result_shape", shape=format_shape(result)),
    ] + _preview_explanation([*shapes, display_result], theme)
    explanation.extend(evaluation_explanation(evaluation, len(trace.terms)))
    explanation.extend(numeric_explanation(semantics))
    if focus is not None:
        explanation.extend(_trace_explanation(trace))
    content = renderer.render_panels(
        panels=panels,
        connectors=connectors,
        explanation=explanation,
        theme=theme,
        precision=precision,
    )
    visual = _visual(
        content, shapes[0], renderer, selected=selected, result=result, explanation=explanation
    )
    visual.trace = trace
    visual.metadata["value_evaluation"] = evaluation
    visual.metadata["numeric_semantics"] = semantics
    return visual
