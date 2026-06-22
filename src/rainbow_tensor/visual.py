"""Rendered result object and the shared rendering substrate.

``TensorVisual`` is what every public op returns: an SVG string plus a plain
text explanation that stay inspectable outside a notebook. The small helpers
here (value lookup, source values, large preview lines) are shared by the op
module and the einsum module, so they live below the result object rather than
beside any one caller.
"""

from math import prod

from .backends import value_at_coordinate
from .explanations import t
from .layout import axis_visible_limits
from .shape import flat_index, format_shape


class TensorVisual:
    """A rendered tensor visualisation.

    The ``svg`` attribute holds the SVG string, and ``text`` holds the plain
    explanation. In a notebook the image is displayed first and the explanation
    is printed as standard output underneath it. Outside a notebook both values
    stay available for inspection and testing.
    """

    def __init__(
        self,
        svg,
        shape,
        selected=None,
        result=None,
        explanation=None,
        renderer="svg",
        mime_type="image/svg+xml",
    ):
        self.svg = svg
        self.content = svg
        self.shape = shape
        self.selected = selected
        self.result_shape = result
        self.explanation = explanation or []
        self.text = "\n".join(self.explanation)
        self.renderer = renderer
        self.mime_type = mime_type

    def _repr_svg_(self):
        if self.mime_type == "image/svg+xml":
            return self.svg
        return None

    def __str__(self):
        return self.svg

    def __repr__(self):
        # Stable text repr so a re-run notebook's text/plain output never churns
        # on the object's memory address.
        return f"TensorVisual(shape={self.shape}, result_shape={self.result_shape})"

    def _ipython_display_(self):
        """Display the figure and print explanation text as standard output."""
        from IPython.display import SVG, display

        if self.mime_type == "image/svg+xml":
            display(SVG(self.svg))
        else:
            display({self.mime_type: self.content}, raw=True)
        if self.text:
            print(self.text)

    def save(self, path):
        """Write the SVG to ``path`` and return the path.

        The text is written as UTF-8 so the ellipsis and legend glyphs survive
        the round trip. ``path`` may be a string or a path-like object.
        """
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.svg)
        return path


def _visual(content, shape, renderer, selected=None, result=None, explanation=None):
    """Build the visual result object for a renderer output."""
    return TensorVisual(
        content,
        shape,
        selected=selected,
        result=result,
        explanation=explanation,
        renderer=getattr(renderer, "name", "custom"),
        mime_type=getattr(renderer, "mime_type", "image/svg+xml"),
    )


def _preview_explanation(shapes, theme):
    """Return standard output lines for large tensor previews."""
    lines = []
    for i, shape in enumerate(shapes):
        limits = axis_visible_limits(shape, theme.max_cells, theme.max_visible_cells)
        hidden = [axis for axis, (size, limit) in enumerate(zip(shape, limits)) if size > limit]
        if not hidden:
            continue
        if len(shapes) == 1:
            lines.append(
                t(
                    "preview.large_single",
                    shape=format_shape(shape),
                    max=theme.max_visible_cells,
                    total=prod(shape),
                )
            )
        else:
            lines.append(
                t(
                    "preview.large_multi",
                    i=i,
                    shape=format_shape(shape),
                    max=theme.max_visible_cells,
                    total=prod(shape),
                )
            )
        lines.append(t("preview.hidden", hidden=hidden))
    return lines


def _value_fn_for(obj):
    """Build a coordinate to value function for array-like input.

    A tuple carries no values, so generated values are used. Any object with
    a ``.shape`` attribute is read by coordinate, which covers NumPy arrays
    and any compatible array-like without importing a tensor library.
    """
    if isinstance(obj, tuple):
        return None
    if not hasattr(obj, "shape"):
        return None

    def value_fn(coord):
        return value_at_coordinate(obj, coord)

    return value_fn


def _source_value(array, shape):
    """Return a coordinate to value function for ``array``.

    An array-like is read by coordinate. A bare shape tuple carries no values,
    so the row major flat index stands in, matching :func:`shape`.
    """
    value_fn = _value_fn_for(array)
    if value_fn is None:
        return lambda coord: flat_index(coord, shape)
    return value_fn
