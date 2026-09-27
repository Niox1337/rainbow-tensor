"""Notebook lessons that isolate one contribution within a recorded expression."""

from .explanations import t
from .interactive import FocusExplorer
from .layout import build_layout
from .numerics import numeric_explanation
from .ops.elementwise import BINARY_SYMBOLS
from .provenance.flow import Flow
from .provenance.lesson import _position, build_lesson
from .provenance.model import TrackedTensor
from .provenance.query import _limit, _node
from .provenance.view import _immediate_trace, _label
from .renderers import resolve_renderer
from .theme import resolve_theme
from .tracing import _normalize_focus
from .views import matmul, mean, sum
from .visual import _preview_explanation, _shape_caption_parts, _visual


def _lesson_text(node, snapshot):
    """Explain one occurrence without flattening local division or binary roles."""
    step = snapshot.step
    if step is None:
        return [t("lesson.empty")]
    lines = [t(
        "lesson.selected", number=snapshot.occurrence,
        label=_label(node.flow, step.reference), operation=step.operation,
    )]
    if snapshot.term is not None:
        lines.append(t("lesson.term", number=snapshot.term + 1, total=step.term_count))
        references = [snapshot.trace.steps[i].reference for i in snapshot.factor_occurrences]
        separator = f" {BINARY_SYMBOLS[snapshot.binary_operator]} " if (
            snapshot.binary_operator is not None
        ) else " * "
        expression = separator.join(_label(node.flow, reference) for reference in references)
        if snapshot.numeric_complete:
            expression += " = " + separator.join(str(value) for value in snapshot.factor_values)
        lines.append(t(
            "lesson.product", expression=expression,
            value=snapshot.term_value if snapshot.numeric_complete else "?",
        ))
    elif not step.term_count:
        lines.append(t("lesson.no_terms"))
    if snapshot.available_terms < step.term_count:
        lines.append(t("lesson.prefix", shown=snapshot.available_terms, total=step.term_count))
    if snapshot.subtotal is not None and step.operation in {"sum", "mean", "matmul", "einsum"}:
        lines.append(t("lesson.subtotal", value=snapshot.subtotal))
    if step.operation == "mean":
        lines.append(t("trace.empty_mean") if step.divisor == 0 else t(
            "lesson.divisor", divisor=step.divisor,
        ))
    lines.append(t(
        "lesson.result", label=_label(node.flow, step.reference),
        value=snapshot.output_value if snapshot.numeric_complete else "?",
    ))
    if not snapshot.trace.complete:
        lines.append(t("lesson.partial", limits=", ".join(snapshot.trace.truncated_reasons)))
    if not snapshot.numeric_complete:
        lines.append(t("lesson.values_skipped", limit=snapshot.evaluation["reason"]))
    return lines


def _render_lesson(node, focus, occurrence, term, *, theme=None, precision=2, renderer=None,
                   max_panels=8, **limits):
    """Render the chosen contribution from one shared snapshot evaluation."""
    theme = resolve_theme(theme)
    renderer = resolve_renderer(renderer)
    if getattr(renderer, "mime_type", "image/svg+xml") != "image/svg+xml":
        raise ValueError("walkthrough requires an SVG renderer")
    panel_limit = _limit(max_panels, "max_panels")
    trace = node.trace(focus, **{
        key: limits[key] for key in ("max_depth", "max_nodes", "max_edges")
    })
    selected = set()
    if trace.steps:
        index = _position(occurrence, len(trace.steps), "occurrence")
        step = trace.steps[index]
        groups = (step.operands,) if step.binary_operator and step.operands else step.terms
        chosen = 0 if term is None and groups else term
        if chosen is not None:
            chosen = _position(chosen, len(groups), "term")
            selected.update(trace.steps[child].reference for child in groups[chosen])
        selected.add(step.reference)
    reached = {
        (step.reference.node_id, step.reference.output_port): _node(node.flow, step.reference)
        for step in trace.steps
    }
    reached[(node.node_id, node.output_port)] = node
    ordered = sorted(reached.values(), key=lambda item: (int(item.node_id[1:]), item.output_port))
    displayed = ordered[-panel_limit:]
    planned = []
    panels = []
    for source in displayed:
        coordinates = sorted(
            reference.coordinate for reference in selected
            if (reference.node_id, reference.output_port) == (source.node_id, source.output_port)
        )
        if source is node and trace.root is not None and trace.root.coordinate not in coordinates:
            coordinates.append(trace.root.coordinate)
        layout = build_layout(source.shape, selected=coordinates, theme=theme)
        planned.extend(source.ref(cell.coord) for cell in layout.cells if not cell.ellipsis)
        panels.append({
            "shape": source.shape, "selected": coordinates,
            "caption_parts": _shape_caption_parts(source.name, source.shape, theme),
        })
    snapshot = build_lesson(
        node, focus, occurrence=occurrence, term=term, preview_references=planned, **limits,
    )
    for source, panel in zip(displayed, panels):
        panel["value_fn"] = lambda coord, source=source: snapshot.values.get(source.ref(coord), "?")
    explanation = _lesson_text(node, snapshot)
    omitted = len(ordered) - len(displayed)
    if omitted:
        explanation.append(t("flow.panels_omitted", count=omitted))
    explanation.extend(_preview_explanation([source.shape for source in displayed], theme))
    semantics = {
        "arithmetic": "python_scalar", "scope": "recorded_operations",
        "operand_sources": sorted(node.numeric_sources),
    }
    explanation.extend(numeric_explanation({
        **semantics, "operand_sources": [
            source for source in semantics["operand_sources"] if source == "array_values"
        ],
    }))
    if "generated_values" in node.numeric_sources:
        explanation.append(t("flow.generated"))
    if any(step.binary_operator == "divide" for step in snapshot.trace.steps):
        explanation.append(t("elementwise.divide_zero"))
    content = renderer.render_panels(
        panels=panels, connectors=[""] * (len(panels) - 1), explanation=explanation,
        theme=theme, precision=precision,
    )
    visual = _visual(content, node.shape, renderer, result=node.shape, explanation=explanation)
    visual.trace = _immediate_trace(node, trace)
    visual.provenance = snapshot.trace
    visual.metadata.update(
        lesson=snapshot, trace_in_explanation=True,
        interaction_description="\n".join(_lesson_text(node, snapshot)),
        focused_output_panel=len(panels) - 1 if trace.root else None,
        output_panel_indices=(len(panels) - 1,),
        value_evaluation=dict(snapshot.evaluation),
        numeric_semantics=semantics,
        panel_nodes=tuple((source.node_id, source.output_port) for source in displayed),
    )
    return visual


class _LessonFocusExplorer(FocusExplorer):
    """Synchronize contribution controls after both browser and Python focus changes."""

    def __init__(self, owner, widgets, figure_class, focus):
        self.owner = owner
        super().__init__(owner._render, (), {"focus": focus}, widgets, figure_class)

    def _show(self):
        super()._show()
        self.owner._sync(self.visual.metadata["lesson"])


def _owned(widgets):
    """Include layout and style models when collecting widget resources to close."""
    owned = list(widgets)
    for widget in tuple(owned):
        for name in ("layout", "style"):
            item = getattr(widget, name, None)
            if item is not None and item not in owned:
                owned.append(item)
    return owned


class Walkthrough:
    """Select an output, an occurrence, and one ordered contribution at a time.

    Create this controller with :func:`walkthrough`. ``snapshot`` exposes
    the bounded numerical lesson, while ``visual`` remains a static exportable
    figure. Occurrence IDs refer to the current trace only. Selecting another
    output resets the lesson to its first occurrence and first available term.
    Input values remain live. Close the controller after the lesson to release
    its notebook communications.
    """

    def __init__(self, node, focus=None, **options):
        import ipywidgets as widgets

        from ._widgets import ExplorerFigure

        self.node = node
        self._options = options
        self._selection = (0, None)
        self._syncing = False
        self._closed = False
        self._callbacks = False
        self.occurrences = widgets.Dropdown(description=t("lesson.occurrence"))
        self.previous_button = widgets.Button(description=t("lesson.previous"))
        self.next_button = widgets.Button(description=t("lesson.next"))
        self.term_label = widgets.Label()
        self._controls = widgets.HBox([
            self.occurrences, self.previous_button, self.next_button, self.term_label,
        ], layout=widgets.Layout(flex_flow="row wrap"))
        self._owned_widgets = _owned([
            self.occurrences, self.previous_button, self.next_button,
            self.term_label, self._controls,
        ])
        try:
            self.explorer = _LessonFocusExplorer(self, widgets, ExplorerFigure, focus)
            self.widget = widgets.VBox([self._controls, self.explorer.widget])
            self._owned_widgets.extend(_owned([self.widget]))
        except Exception:
            self.close()
            raise
        self.occurrences.observe(self._on_occurrence, names="value")
        self.previous_button.on_click(self._on_previous)
        self.next_button.on_click(self._on_next)
        self._callbacks = True

    @property
    def snapshot(self):
        """Return the last successful occurrence snapshot."""
        return self.visual.metadata["lesson"]

    @property
    def visual(self):
        """Return the current exportable figure and teaching explanation."""
        return self.explorer.visual

    @property
    def focus(self):
        """Return the selected final output coordinate, or None for an empty output."""
        return self.explorer.focus

    def _render(self, focus=None):
        normalized = _normalize_focus(focus, self.node.shape)
        occurrence, term = self._selection
        if hasattr(self, "explorer") and normalized != self.focus:
            occurrence, term = 0, None
        return _render_lesson(self.node, normalized, occurrence, term, **self._options)

    def _sync(self, snapshot):
        self._selection = (snapshot.occurrence or 0, snapshot.term)
        self._syncing = True
        try:
            self.occurrences.description = t("lesson.occurrence")
            self.occurrences.options = tuple(
                (f"{index}: {_label(self.node.flow, step.reference)} ({step.operation})", index)
                for index, step in enumerate(snapshot.trace.steps)
            )
            self.occurrences.value = snapshot.occurrence
            self.occurrences.disabled = snapshot.occurrence is None
            self.previous_button.description = t("lesson.previous")
            self.next_button.description = t("lesson.next")
            self.previous_button.disabled = snapshot.term is None or snapshot.term == 0
            self.next_button.disabled = (
                snapshot.term is None or snapshot.term + 1 >= snapshot.available_terms
            )
            if snapshot.term is not None:
                self.term_label.value = t(
                    "lesson.term", number=snapshot.term + 1, total=snapshot.step.term_count,
                )
            elif snapshot.step is not None and snapshot.step.term_count:
                self.term_label.value = t("lesson.prefix", shown=0, total=snapshot.step.term_count)
            else:
                self.term_label.value = t("lesson.no_terms")
        finally:
            self._syncing = False

    def _select(self, occurrence, term):
        if self._closed:
            raise RuntimeError("this walkthrough is closed")
        previous = self._selection
        self._selection = (occurrence, term)
        try:
            return self.explorer.set_focus(self.focus)
        except Exception:
            self._selection = previous
            raise

    def set_focus(self, coordinate):
        """Select another final output and reset its contribution walkthrough."""
        if self._closed:
            raise RuntimeError("this walkthrough is closed")
        return self.explorer.set_focus(coordinate)

    def select_occurrence(self, occurrence):
        """Choose one trace occurrence without collapsing repeated references."""
        return self._select(occurrence, None)

    def select_term(self, term):
        """Select a zero-based available term, preserving prior state on failure."""
        _position(term, self.snapshot.available_terms, "term")
        return self._select(self.snapshot.occurrence or 0, term)

    def next_term(self):
        """Advance one available term without increasing structural or value budgets."""
        if self.snapshot.term is None:
            raise IndexError("this occurrence has no contributing terms")
        return self.select_term(self.snapshot.term + 1)

    def previous_term(self):
        """Return to the previous term, rejecting movement before the first term."""
        if self.snapshot.term is None:
            raise IndexError("this occurrence has no contributing terms")
        return self.select_term(self.snapshot.term - 1)

    def _handle(self, action):
        try:
            action()
        except (TypeError, ValueError, IndexError, RuntimeError, ArithmeticError) as error:
            self._sync(self.snapshot)
            self.explorer.status.value = t("interactive.error", error=error)

    def _on_occurrence(self, change):
        if not self._syncing and not self._closed and change["new"] is not None:
            self._handle(lambda: self.select_occurrence(change["new"]))

    def _on_previous(self, button):
        if not self._closed and not button.disabled:
            self._handle(self.previous_term)

    def _on_next(self, button):
        if not self._closed and not button.disabled:
            self._handle(self.next_term)

    def _ipython_display_(self):
        from IPython.display import display

        display(self.widget)

    def close(self):
        """Release every owned widget and stop accepting lesson updates."""
        if self._closed:
            return
        self._closed = True
        if self._callbacks:
            self.occurrences.unobserve(self._on_occurrence, names="value")
            self.previous_button.on_click(self._on_previous, remove=True)
            self.next_button.on_click(self._on_next, remove=True)
        if hasattr(self, "explorer"):
            self.explorer.close()
        for widget in self._owned_widgets:
            widget.close()


def walkthrough(
    operation, *args, focus=None, theme=None, precision=2, renderer=None,
    max_depth=6, max_nodes=80, max_edges=120, max_panels=8,
    max_terms=10_000, max_total_terms=100_000, **operation_options,
):
    """Explain sum, mean, matmul, or a recorded Flow one contribution at a time.

    Pass a supported operation and its inputs, or a tracked result directly.
    For example, ``walkthrough(sum, x, axis=1)`` exposes the ordered terms in
    each row sum. ``walkthrough(result)`` also exposes intermediate occurrences
    in a Flow. Numerical work uses one shared recursive Flow budget per refresh.
    Live controls require a notebook host with widget support. ``visual.save``
    exports the selected static figure without live controls.
    """
    if isinstance(operation, TrackedTensor):
        if args or operation_options:
            raise TypeError("a recorded tensor already contains its operation inputs")
        node = operation
    else:
        flow = Flow()
        if operation in (sum, mean):
            if not 1 <= len(args) <= 2:
                raise TypeError("sum and mean walkthroughs require an array and optional axis")
            source = flow.input(args[0], name="X")
            method = flow.sum if operation is sum else flow.mean
            node = method(source, *args[1:], name="Y", **operation_options)
        elif operation is matmul:
            if len(args) != 2 or operation_options:
                raise TypeError("a matmul walkthrough requires exactly two operands")
            node = flow.matmul(
                flow.input(args[0], name="A"), flow.input(args[1], name="B"), name="Y",
            )
        else:
            raise ValueError("walkthrough supports sum, mean, matmul, and recorded flow tensors")
    return Walkthrough(
        node, focus, theme=theme, precision=precision, renderer=renderer,
        max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges, max_panels=max_panels,
        max_terms=max_terms, max_total_terms=max_total_terms,
    )
