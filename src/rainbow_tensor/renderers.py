"""Renderer registry.

SVG remains the built in renderer. The notebook layer talks to the small
renderer interface here, which gives later HTML or canvas renderers a stable
entry point without moving shape maths into a display backend.
"""

from . import config
from .render_svg import render_panels, render_svg


class SvgRenderer:
    """Default renderer that returns SVG strings."""

    name = "svg"
    mime_type = "image/svg+xml"

    def render_tensor(self, **kwargs):
        """Render one tensor panel as SVG."""
        return render_svg(**kwargs)

    def render_panels(self, **kwargs):
        """Render several tensor panels as one SVG."""
        return render_panels(**kwargs)


SVG = SvgRenderer()
_RENDERERS = {SVG.name: SVG}
# The mutable default renderer lives in ``config``; ``None`` there means SVG.


def register_renderer(renderer):
    """Register a renderer by its ``name`` attribute."""
    name = getattr(renderer, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("renderer must have a non empty string name")
    _check_renderer(renderer)
    _RENDERERS[name] = renderer
    return renderer


def resolve_renderer(renderer=None):
    """Resolve a renderer object, a renderer name, or ``None``."""
    if renderer is None:
        return config.default_renderer if config.default_renderer is not None else SVG
    if isinstance(renderer, str):
        try:
            return _RENDERERS[renderer]
        except KeyError:
            known = ", ".join(sorted(_RENDERERS))
            raise ValueError(
                f"unknown renderer {renderer!r}, available renderers are {known}"
            ) from None
    _check_renderer(renderer)
    return renderer


def get_default_renderer():
    """Return the current default renderer."""
    return config.default_renderer if config.default_renderer is not None else SVG


def set_default_renderer(renderer):
    """Set the default renderer used when no call passes one."""
    config.default_renderer = resolve_renderer(renderer)
    return config.default_renderer


def _check_renderer(renderer):
    """Validate the renderer interface."""
    render_tensor = getattr(renderer, "render_tensor", None)
    render_panels_fn = getattr(renderer, "render_panels", None)
    if not callable(render_tensor) or not callable(render_panels_fn):
        raise TypeError("renderer must define render_tensor and render_panels")
