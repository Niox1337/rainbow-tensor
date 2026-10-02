"""Notebook controls that trace a chosen output back to its sources."""

from functools import partial
from html import escape
from json import JSONDecodeError, loads

from .._svg.interaction import capture_cells, interactive_svg
from ..explanations import t
from ..tracing import _normalize_focus, _trace_explanation
from ..views import (
    add,
    broadcast,
    concatenate,
    divide,
    einsum,
    expand_dims,
    index,
    matmul,
    mean,
    moveaxis,
    multiply,
    repeat,
    reshape,
    squeeze,
    stack,
    subtract,
    sum,
    swapaxes,
    take,
    transpose,
)
from ..views.shapes import _index_visual


def _own_widget(owned, widget):
    """Track a new control and its auxiliary models before the next allocation."""
    for model in (widget, getattr(widget, "layout", None), getattr(widget, "style", None)):
        if model is not None and model not in owned:
            owned.append(model)
    return widget


class FocusExplorer:
    """Keep clickable output cells and editable output coordinates together.

    Create this object with :func:`explore`. ``visual`` is the latest successful
    static result and can be saved normally. ``set_focus`` also works from Python,
    including negative coordinates. Updates read the original arrays again, so
    changed input values are reflected without retaining stale arithmetic caches.
    The inputs must retain their shape while this explorer is in use.

    Clicking an output cell redraws immediately. Coordinate fields redraw only
    after the update button is pressed. Enter and Space activate a focused cell.
    ``close`` releases the controls when an explorer is no longer needed.
    """

    def __init__(self, operation, args, options, widgets, figure_class):
        self._operation = operation
        self._args = args
        self._options = dict(options)
        self._closed = False
        self._callbacks = False
        self._owned_widgets = []
        with capture_cells() as panels:
            self.visual = operation(*args, **options)
        self._panels = panels
        self._clickable = frozenset()
        if self.visual.mime_type != "image/svg+xml":
            raise ValueError("explore requires an SVG renderer")
        self.focus = self.visual.trace.output_coord if self.visual.trace is not None else None
        self.result_shape = self.visual.result_shape
        display = self._prepare_display(self.visual, panels, self.focus)
        own = partial(_own_widget, self._owned_widgets)
        try:
            self.coordinates = tuple(
                own(widgets.BoundedIntText(value=value, min=0, max=size - 1))
                for value, size in zip(self.focus or (), self.result_shape)
            )
            self.update_button = own(widgets.Button(icon="refresh", disabled=self.focus is None))
            self.status = own(widgets.Label())
            self.figure = own(figure_class())
            self.explanation = own(widgets.HTML())
            self._controls = own(widgets.HBox(
                [*self.coordinates, self.update_button],
                layout=own(widgets.Layout(flex_flow="row wrap")),
            ))
            self.widget = own(widgets.VBox([
                self._controls, self.status, self.figure, self.explanation,
            ]))
            self.update_button.on_click(self._on_update)
            self.figure.on_msg(self._on_cell)
            self._callbacks = True
            self._show(display)
        except Exception:
            self.close()
            raise

    def _prepare_display(self, visual, panels, focus):
        """Format and annotate a candidate before replacing the successful display."""
        output_panel = visual.metadata.get("focused_output_panel", len(panels) - 1)
        content, clickable = interactive_svg(visual.svg, panels, output_panel, focus)
        description = visual.metadata.get("interaction_description")
        if description is None:
            description = "\n".join(_trace_explanation(visual.trace))
        lines = list(visual.explanation)
        if visual.trace is not None and not visual.metadata.get("trace_in_explanation", False):
            lines.extend(_trace_explanation(visual.trace))
        return {
            "axes": tuple(t("interactive.axis", axis=axis) for axis in range(len(focus or ()))),
            "update": t("interactive.update"),
            "content": content,
            "clickable": clickable,
            "description": description,
            "label": t("interactive.result_grid"),
            "explanation": (
                '<pre style="white-space:pre-wrap">' + escape("\n".join(lines)) + "</pre>"
            ),
            "status": t("interactive.empty") if focus is None else t(
                "interactive.showing", coordinate=focus,
            ),
        }

    def _show(self, display):
        """Apply prepared widget values without evaluating inputs or translating again."""
        for control, label in zip(self.coordinates, display["axes"]):
            control.description = label
        self.update_button.description = display["update"]
        self._clickable = display["clickable"]
        with self.figure.hold_sync():
            self.figure.value = display["content"]
            self.figure.description = display["description"]
            self.figure.label = display["label"]
            self.figure.revision += 1
        self.explanation.value = display["explanation"]
        self.status.value = display["status"]

    def set_focus(self, coordinate):
        """Redraw one output and return its visual, preserving state on failure.

        ``coordinate`` follows the same tuple, rank, and bounds rules as the
        operation's ``focus`` argument. A scalar output uses ``()``. The original
        theme, renderer, precision, and any calculation budgets remain in effect.
        Empty results have no addressable output and reject focus updates.
        """
        if self._closed:
            raise RuntimeError("this explorer is closed")
        if self.focus is None:
            raise IndexError("cannot focus an empty result")
        normalized = _normalize_focus(coordinate, self.result_shape)
        options = {**self._options, "focus": normalized}
        with capture_cells() as panels:
            visual = self._operation(*self._args, **options)
        if visual.mime_type != "image/svg+xml":
            raise ValueError("explore requires an SVG renderer")
        if visual.result_shape != self.result_shape:
            raise ValueError("the output shape changed, create a new explorer")
        display = self._prepare_display(visual, panels, normalized)
        self.visual = visual
        self._panels = panels
        self.focus = normalized
        for control, value in zip(self.coordinates, normalized):
            control.value = value
        self._show(display)
        return visual

    def _on_cell(self, widget, content, buffers):
        """Accept only current, visible output coordinates from the browser.

        The kernel owns the allowlist. Hidden positions, source cells, malformed
        payloads, stale redraw events, and messages after close are ignored.
        A valid event uses the same update path as Python and keyboard controls.
        """
        if self._closed or not isinstance(content, dict) or content.get("type") != "focus":
            return
        revision = content.get("revision")
        if type(revision) is not int or revision != self.figure.revision:
            return
        coordinate = content.get("coordinate")
        if isinstance(coordinate, str):
            try:
                coordinate = loads(coordinate)
            except (JSONDecodeError, ValueError):
                return
        if not isinstance(coordinate, list) or any(type(value) is not int for value in coordinate):
            return
        coordinate = tuple(coordinate)
        if coordinate not in self._clickable:
            return
        try:
            self.set_focus(coordinate)
        except (TypeError, ValueError, IndexError, RuntimeError, ArithmeticError) as exc:
            self.status.value = t("interactive.error", error=exc)

    def _on_update(self, button):
        """Show an actionable error without discarding the previous figure."""
        if self.focus is None:
            return
        try:
            self.set_focus(tuple(control.value for control in self.coordinates))
        except (TypeError, ValueError, IndexError, RuntimeError, ArithmeticError) as exc:
            self.status.value = t("interactive.error", error=exc)

    def _ipython_display_(self):
        from IPython.display import display

        display(self.widget)

    def close(self):
        """Release acquired widgets, including an incompletely initialized explorer."""
        if self._closed:
            return
        self._closed = True
        if self._callbacks:
            self.update_button.on_click(self._on_update, remove=True)
            self.figure.on_msg(self._on_cell, remove=True)
        for widget in self._owned_widgets:
            widget.close()


def explore(operation, *args, **kwargs):
    """Explore one operation or a recorded tensor flow in a notebook.

    Pass the operation itself, followed by its usual arguments, for example
    ``explore(matmul, a, b, focus=(1, 2), max_terms=1000)``. Coordinates can be
    changed by clicking a visible result cell or using Enter or Space on it.
    Coordinate fields and the update button also reach hidden positions. Scalar
    results expose one clickable cell and no coordinate fields. The budget is never increased
    by interaction, and the latest static result stays available as ``visual``.
    Empty results have ``focus=None``, no coordinate fields, and a disabled
    update button. Their empty figure remains available for static export.

    ``explore(index, array, selection)`` starts at the first result position and
    highlights its corresponding source element. Each repeated gather position
    remains separately selectable, and reverse slices keep their result order.
    Both panels stay bounded by the theme's preview limits. The index mapping is
    rebuilt on updates so edits to index arrays are reflected as well as values.

    Shape transformations and combining operations expose the same controls.
    For broadcast, ``focus_operand=0`` or ``1`` selects the output to explore.
    ``explore(tracked_tensor)`` follows a recorded flow back through its sources.
    Custom SVG renderers retain coordinate controls even without cell metadata.

    ``ipywidgets`` and ``anywidget`` are installed with the package. They are
    imported only here to keep static rendering free of widget initialization.
    Live controls need a running notebook kernel and widget support in its host.
    """
    mapped = (
        reshape, transpose, swapaxes, moveaxis, squeeze, expand_dims,
        concatenate, stack, repeat, take, broadcast,
    )
    supported = (index, sum, mean, matmul, einsum, add, subtract, multiply, divide, *mapped)
    if operation not in supported:
        from ..provenance.model import TrackedTensor

        if not isinstance(operation, TrackedTensor):
            raise ValueError("explore supports tensor operations and recorded flow tensors")
        if args:
            raise TypeError("a recorded tensor already contains its operation inputs")
        operation = operation.visualize
    try:
        import ipywidgets as widgets

        from .._widgets import ExplorerFigure
    except ImportError as exc:
        raise ImportError(
            "Notebook controls require ipywidgets and anywidget. "
            "Repair the installation in this Python environment with: "
            "python -m pip install --upgrade rainbow-tensor"
        ) from exc
    if operation is index:
        operation = partial(_index_visual, focus_first=True)
    elif operation in mapped and kwargs.get("focus") is None:
        from ..views._focus import FIRST_OUTPUT

        kwargs["focus"] = FIRST_OUTPUT
    return FocusExplorer(operation, args, kwargs, widgets, ExplorerFigure)
