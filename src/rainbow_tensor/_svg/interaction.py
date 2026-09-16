"""Capture bounded SVG cells without changing static rendering or reading values."""

from contextlib import contextmanager
from contextvars import ContextVar
from html import escape
from json import dumps

from ..explanations import t

_panels = ContextVar("rendered_interaction_panels", default=None)


@contextmanager
def capture_cells():
    """Collect final panel fragments within one render, including empty panels.

    Context-local storage keeps concurrent and nested renders independent. Only
    already rendered cells are recorded, so capture adds no backend value reads.
    """
    panels = []
    token = _panels.set(panels)
    try:
        yield panels
    finally:
        _panels.reset(token)


def capturing_cells():
    """Return whether this render has an interested notebook explorer."""
    return _panels.get() is not None


def record_panel(body, cells):
    """Record one final body and its visible, non-ellipsis cell fragments."""
    panels = _panels.get()
    if panels is not None:
        panels.append((body, tuple(cells)))


def interactive_svg(svg, panels, output_panel, focus):
    """Annotate a copy of one output panel and return its allowed coordinates.

    Fragments are matched once in rendering order. Advancing past each complete
    panel prevents identical source and result bodies from becoming ambiguous.
    A custom renderer that changes those fragments retains keyboard controls.
    """
    if focus is None or output_panel is None:
        return svg, frozenset()
    cursor = 0
    for index, (body, cells) in enumerate(panels):
        start = svg.find(body, cursor)
        if start < 0 or not body:
            return svg, frozenset()
        cursor = start + len(body)
        if index != output_panel:
            continue
        pieces = []
        position = 0
        coordinates = set()
        for fragment, coordinate in cells:
            at = body.find(fragment, position)
            if at < 0:
                return svg, frozenset()
            label = escape(t("interactive.showing", coordinate=coordinate), quote=True)
            payload = escape(dumps(coordinate, separators=(",", ":")), quote=True)
            pressed = "true" if coordinate == focus else "false"
            pieces.extend((
                body[position:at],
                f'<g data-rt-coordinate="{payload}" role="button" tabindex="0" '
                f'aria-label="{label}" aria-pressed="{pressed}">{fragment}</g>',
            ))
            position = at + len(fragment)
            coordinates.add(coordinate)
        pieces.append(body[position:])
        return svg[:start] + "".join(pieces) + svg[cursor:], frozenset(coordinates)
    return svg, frozenset()
