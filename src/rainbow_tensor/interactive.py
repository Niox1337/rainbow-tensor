"""Optional notebook controls that redraw an existing mathematical visual."""

from html import escape

from .tracing import _normalize_focus, _trace_explanation
from .views import einsum, matmul, mean, sum


class FocusExplorer:
    """Keep a mathematical visual and keyboard-editable output coordinates together.

    Create this object with :func:`explore`. ``visual`` is the latest successful
    static result and can be saved normally. ``set_focus`` also works from Python,
    including negative coordinates. Updates read the original arrays again, so
    changed input values are reflected without retaining stale arithmetic caches.
    The inputs must retain their shape while this explorer is in use.

    The notebook controls only redraw after the update button is pressed.
    ``close`` releases the controls when an explorer is no longer needed.
    """

    def __init__(self, operation, args, options, widgets):
        self._operation = operation
        self._args = args
        self._options = dict(options)
        self._closed = False
        self.visual = operation(*args, **options)
        if self.visual.mime_type != "image/svg+xml":
            raise ValueError("explore requires an SVG renderer")
        self.focus = self.visual.trace.output_coord
        self.result_shape = self.visual.result_shape
        self.coordinates = tuple(
            widgets.BoundedIntText(
                value=value, min=0, max=size - 1, description=f"Axis {axis}",
            )
            for axis, (value, size) in enumerate(zip(self.focus, self.result_shape))
        )
        self.update_button = widgets.Button(description="Update focus", icon="refresh")
        self.status = widgets.Label()
        self.figure = widgets.HTML()
        self.explanation = widgets.HTML()
        self._controls = widgets.HBox([*self.coordinates, self.update_button])
        self.widget = widgets.VBox([
            self._controls, self.status, self.figure, self.explanation,
        ])
        self._owned_widgets = [
            *self.coordinates, self.update_button, self.status, self.figure,
            self.explanation, self._controls, self.widget,
        ]
        for control in tuple(self._owned_widgets):
            for name in ("layout", "style"):
                model = getattr(control, name, None)
                if model is not None and model not in self._owned_widgets:
                    self._owned_widgets.append(model)
        self.update_button.on_click(self._on_update)
        self._show()

    def _show(self):
        """Refresh the widgets from the last successful static visual."""
        self.figure.value = self.visual.svg
        lines = list(self.visual.explanation)
        if not any(line.startswith("Focus:") for line in lines):
            lines.extend(_trace_explanation(self.visual.trace))
        self.explanation.value = (
            '<pre style="white-space:pre-wrap">' + escape("\n".join(lines)) + "</pre>"
        )
        self.status.value = f"Showing output {self.focus}."

    def set_focus(self, coordinate):
        """Redraw one output and return its visual, preserving state on failure.

        ``coordinate`` follows the same tuple, rank, and bounds rules as the
        operation's ``focus`` argument. A scalar output uses ``()``. The original
        theme, renderer, precision, and ``max_terms`` options remain in effect.
        """
        if self._closed:
            raise RuntimeError("this explorer is closed")
        normalized = _normalize_focus(coordinate, self.result_shape)
        options = {**self._options, "focus": normalized}
        visual = self._operation(*self._args, **options)
        if visual.mime_type != "image/svg+xml":
            raise ValueError("explore requires an SVG renderer")
        if visual.result_shape != self.result_shape:
            raise ValueError("the output shape changed, create a new explorer")
        self.visual = visual
        self.focus = normalized
        for control, value in zip(self.coordinates, normalized):
            control.value = value
        self._show()
        return visual

    def _on_update(self, button):
        """Show an actionable error without discarding the previous figure."""
        try:
            self.set_focus(tuple(control.value for control in self.coordinates))
        except (TypeError, ValueError, IndexError, RuntimeError) as exc:
            self.status.value = f"Could not update focus: {exc}"

    def _ipython_display_(self):
        from IPython.display import display

        display(self.widget)

    def close(self):
        """Release widget communications and disable further updates."""
        if self._closed:
            return
        self._closed = True
        self.update_button.on_click(self._on_update, remove=True)
        for widget in self._owned_widgets:
            widget.close()


def explore(operation, *args, **kwargs):
    """Explore outputs of ``sum``, ``mean``, ``matmul``, or ``einsum`` in a notebook.

    Pass the operation itself, followed by its usual arguments, for example
    ``explore(matmul, a, b, focus=(1, 2), max_terms=1000)``. Coordinates can be
    changed with the keyboard, then applied with the update button. Scalar
    results expose only that button. The calculation budget is never increased
    by interaction, and the latest static result stays available as ``visual``.

    Requires the optional ``rainbow-tensor[interactive]`` dependencies. They are
    imported only here, so static rendering does not require notebook widgets.
    Live controls need a running notebook kernel and widget support in its host.
    """
    if operation not in (sum, mean, matmul, einsum):
        raise ValueError("explore supports rt.sum, rt.mean, rt.matmul, and rt.einsum")
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError(
            'Notebook controls require: pip install "rainbow-tensor[interactive]"'
        ) from exc
    return FocusExplorer(operation, args, kwargs, widgets)
