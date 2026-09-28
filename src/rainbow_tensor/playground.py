"""Predict and inspect how reduction axes change an output's shape and sources."""

from ast import literal_eval
from html import escape

from .explanations import t
from .interactive import explore
from .ops import normalize_reduction_axes, reduce_result_shape, validate_keepdims
from .shape import extract_shape
from .tracing import _normalize_focus
from .views import mean, shape, sum
from .walkthrough import _owned

_UNCHANGED = object()


class ReductionPlayground:
    """Replace a reduction explorer safely when its parameters change.

    ``axis=None`` reduces all axes and ``axis=()`` reduces none. Tuple axes
    select several dimensions, and ``keepdims`` retains them at length one.
    The source figure stays visible while predict mode hides the result until
    ``reveal`` is called. A valid output focus survives a parameter change.
    Otherwise the replacement starts at its first output, or has no focus for
    an empty result. Invalid changes preserve the prior explorer and parameters.
    """

    def __init__(self, operation, array, *, axis=None, keepdims=False, predict=True,
                 **options):
        if operation not in (sum, mean):
            raise ValueError("a reduction playground supports sum and mean")
        if not isinstance(predict, bool):
            raise TypeError("predict must be a boolean")
        self.operation = operation
        self.array = array
        self.source_shape = extract_shape(array)
        self._options = options
        self._predict = predict
        self._closed = False
        self._callbacks = False
        self._owned_widgets = []
        self.axis, self.keepdims, _ = self._parameters(axis, keepdims)
        self.input_visual = self._input_visual()
        self.explorer = explore(operation, array, axis=self.axis, keepdims=self.keepdims, **options)
        try:
            self._build_widgets()
        except Exception:
            self.close()
            raise
        self.revealed = not predict
        self._sync_controls()
        self._show_result()

    def _parameters(self, axis, keepdims):
        """Normalize controls before opening or replacing a widget."""
        if extract_shape(self.array) != self.source_shape:
            raise ValueError("the source shape changed, create a new playground")
        axes = normalize_reduction_axes(
            axis, len(self.source_shape), allow_scalar_axis=self.operation is sum,
        )
        kept = validate_keepdims(keepdims)
        resolved = None if axis is None else axes
        if axis is not None and not isinstance(axis, tuple) and axes:
            resolved = axes[0]
        result_shape = reduce_result_shape(self.source_shape, axes, keepdims=kept)
        return resolved, kept, result_shape

    def _input_visual(self):
        visual = shape(self.array, **{
            name: self._options[name] for name in ("theme", "precision", "renderer")
            if name in self._options
        })
        if visual.mime_type != "image/svg+xml":
            raise ValueError("a reduction playground requires an SVG renderer")
        return visual

    def _build_widgets(self):
        import ipywidgets as widgets

        self.axis_mode = widgets.Dropdown(description=t("playground.axes"), options=[
            (t("playground.all_axes"), "all"), (t("playground.no_axes"), "none"),
            (t("playground.axis_tuple"), "selected"),
        ])
        self.axes = widgets.SelectMultiple(
            description=t("playground.axis_tuple"), options=[
                (t("interactive.axis", axis=axis), axis) for axis in range(len(self.source_shape))
            ],
        )
        self.keepdims_control = widgets.Checkbox(description=t("playground.keepdims"))
        self.apply_button = widgets.Button(description=t("playground.apply"))
        self.prediction = widgets.Text(description=t("playground.prediction"), placeholder="(2, 3)")
        self.reveal_button = widgets.Button(description=t("playground.reveal"))
        self.status = widgets.Label()
        self.expression_view = widgets.HTML()
        self.source_figure = widgets.HTML(value=self.input_visual.svg)
        self.result_box = widgets.VBox()
        self._controls = widgets.HBox([
            self.axis_mode, self.axes, self.keepdims_control, self.apply_button,
        ], layout=widgets.Layout(flex_flow="row wrap"))
        self._prediction_controls = widgets.HBox([self.prediction, self.reveal_button])
        self.widget = widgets.VBox([
            self.source_figure, self._controls, self.expression_view,
            self._prediction_controls, self.status, self.result_box,
        ])
        self._owned_widgets = _owned([
            self.axis_mode, self.axes, self.keepdims_control, self.apply_button,
            self.prediction, self.reveal_button, self.status, self.expression_view,
            self.source_figure, self.result_box, self._controls,
            self._prediction_controls, self.widget,
        ])
        self.axis_mode.observe(self._on_axis_mode, names="value")
        self.apply_button.on_click(self._on_apply)
        self.reveal_button.on_click(self._on_reveal)
        self._callbacks = True

    @property
    def expression(self):
        """Return the matching NumPy call for the currently applied parameters."""
        name = "sum" if self.operation is sum else "mean"
        return f"np.{name}(x, axis={self.axis!r}, keepdims={self.keepdims!r})"

    @property
    def visual(self):
        """Return the current static result, even while predict mode hides its widget."""
        return self.explorer.visual

    def _sync_controls(self):
        self.axis_mode.value = "all" if self.axis is None else (
            "none" if self.axis == () else "selected"
        )
        self.axes.value = normalize_reduction_axes(self.axis, len(self.source_shape))
        self.axes.disabled = self.axis_mode.value != "selected"
        self.keepdims_control.value = self.keepdims
        self.expression_view.value = "<code>" + escape(self.expression) + "</code>"

    def _show_result(self):
        self.result_box.children = (self.explorer.widget,) if self.revealed else ()
        self.reveal_button.description = t(
            "playground.hide" if self.revealed else "playground.reveal"
        )
        self.prediction.disabled = self.revealed
        self.status.value = t("playground.shape", shape=self.explorer.result_shape) if (
            self.revealed
        ) else t("playground.predict")

    def set_parameters(self, *, axis=_UNCHANGED, keepdims=_UNCHANGED):
        """Apply a new recipe atomically, closing the old explorer only after success."""
        if self._closed:
            raise RuntimeError("this reduction playground is closed")
        axis = self.axis if axis is _UNCHANGED else axis
        keepdims = self.keepdims if keepdims is _UNCHANGED else keepdims
        resolved, kept, result_shape = self._parameters(axis, keepdims)
        try:
            focus = _normalize_focus(self.explorer.focus, result_shape)
        except (ValueError, IndexError):
            focus = None
        options = {**self._options, "focus": focus}
        input_visual = self._input_visual()
        replacement = explore(self.operation, self.array, axis=resolved, keepdims=kept, **options)
        previous = self.explorer
        self.explorer = replacement
        self.axis, self.keepdims = resolved, kept
        self.input_visual = input_visual
        self.source_figure.value = input_visual.svg
        self.prediction.value = ""
        self.revealed = not self._predict
        self._sync_controls()
        self._show_result()
        previous.close()
        return self.visual

    def reveal(self):
        """Show the result and compare an optional tuple-shaped prediction."""
        if self._closed:
            raise RuntimeError("this reduction playground is closed")
        self.revealed = True
        self._show_result()
        if self.prediction.value.strip():
            try:
                predicted = literal_eval(self.prediction.value)
            except (ValueError, SyntaxError):
                predicted = None
            valid = isinstance(predicted, tuple) and all(
                isinstance(size, int) and not isinstance(size, bool) and size >= 0
                for size in predicted
            )
            key = "playground.matches" if valid and predicted == self.explorer.result_shape else (
                "playground.differs"
            )
            self.status.value += " " + t(key)
        return self.visual

    def hide(self):
        """Hide the current result so the learner can make another prediction."""
        if self._closed:
            raise RuntimeError("this reduction playground is closed")
        self.revealed = False
        self._show_result()

    def _on_axis_mode(self, change):
        self.axes.disabled = change["new"] != "selected"

    def _on_apply(self, button):
        if self._closed:
            return
        axis = None if self.axis_mode.value == "all" else (
            () if self.axis_mode.value == "none" else self.axes.value
        )
        try:
            self.set_parameters(axis=axis, keepdims=self.keepdims_control.value)
        except (TypeError, ValueError, IndexError, RuntimeError, ArithmeticError) as error:
            self._sync_controls()
            self.status.value = t("interactive.error", error=error)

    def _on_reveal(self, button):
        if not self._closed:
            self.hide() if self.revealed else self.reveal()

    def _ipython_display_(self):
        from IPython.display import display

        display(self.widget)

    def close(self):
        """Close the active explorer and all playground controls exactly once."""
        if self._closed:
            return
        self._closed = True
        if self._callbacks:
            self.axis_mode.unobserve(self._on_axis_mode, names="value")
            self.apply_button.on_click(self._on_apply, remove=True)
            self.reveal_button.on_click(self._on_reveal, remove=True)
        self.explorer.close()
        for widget in self._owned_widgets:
            widget.close()


def reduction_playground(operation, array, *, axis=None, keepdims=False, predict=True, **options):
    """Explore sum or mean axes with a prediction before revealing each result.

    For example, ``reduction_playground(sum, x, axis=1)`` displays the source,
    the NumPy expression, and axis controls. Change all axes, no axes, or a
    selected tuple, then reveal the shape and choose an output to see its sources.
    Set ``predict=False`` to show each result immediately. Remaining options
    follow ``explore``, including theme and numerical budgets. This playground
    changes explicit recipes without mutating recorded Flow nodes.
    """
    return ReductionPlayground(
        operation, array, axis=axis, keepdims=keepdims, predict=predict, **options,
    )
