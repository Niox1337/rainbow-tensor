"""Theme objects.

A :class:`Theme` bundles every visual choice the renderer needs into one
immutable value: colours, fonts, cell geometry, stroke width, corner radius,
and the truncation limit. ``AUTO`` follows the browser's colour preference,
while ``LIGHT`` and ``DARK`` select fixed palettes. A module level default is
used when a call does not pass its own theme, and each preset can be looked
up by name through :func:`resolve_theme`.

The axis colours form a rainbow ramp keyed by axis depth, which is where the
package gets its name. Axis 0 is red, axis 1 is orange, and deeper axes walk
through amber, green, blue, violet, and pink. Only the non leaf axes draw a
frame today, but the full ramp is ready for the higher dimensional work in a
later milestone.
"""

from dataclasses import dataclass, field, replace

from . import config

# Multi-word font names are wrapped in single quotes so the value stays valid
# inside a double-quoted SVG attribute. Double quotes here would break the XML.
SANS_FAMILY = (
    "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)
MONO_FAMILY = "ui-monospace, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

# A full rainbow ramp keyed by axis depth, fitting the project name. The hues
# are spread so adjacent axes never share a neighbouring hue: red and orange
# open the ramp, then it jumps straight to lime and teal before walking through
# blue, violet, and pink, so two stacked frames are always easy to tell apart.
LIGHT_AXIS_RAMP = (
    "#b91c1c",  # red
    "#b45309",  # orange
    "#4d7c0f",  # lime
    "#0f766e",  # teal
    "#1d4ed8",  # blue
    "#6d28d9",  # violet
    "#be185d",  # pink
)
DARK_AXIS_RAMP = (
    "#f87171",  # red
    "#fb923c",  # orange
    "#a3e635",  # lime
    "#2dd4bf",  # teal
    "#60a5fa",  # blue
    "#a78bfa",  # violet
    "#f472b6",  # pink
)


@dataclass(frozen=True)
class Theme:
    """An immutable bundle of colours, fonts, and geometry.

    Use :meth:`variant` to derive a tweaked copy, for example a theme with a
    larger cell or a different truncation limit, without mutating a preset.
    """

    name: str

    # Surfaces.
    background: str  # the canvas behind everything
    surface: str  # a resting cell fill
    surface_muted: str  # an unselected cell fill in an index view
    surface_selected: str  # a selected cell fill

    # Text.
    text: str  # a resting cell value
    text_muted: str  # secondary text, legend sizes, and unselected values
    text_selected: str  # a selected cell value
    heading: str  # the main label above the explanation

    # Structure.
    axis_colors: tuple = field(default=LIGHT_AXIS_RAMP)
    neutral: str = "#cbd5e1"  # an unselected frame stroke in an index view
    cell_border: str = "#e2e8f0"  # a resting cell outline
    selected_border: str = "#15803d"  # a selected cell outline
    card_border: str = "#e2e8f0"  # the outline of the canvas card

    # Typography.
    sans_family: str = SANS_FAMILY
    mono_family: str = MONO_FAMILY

    # Geometry.
    cell_w: float = 48.0
    cell_h: float = 38.0
    cell_gap: float = 8.0
    row_pad: float = 9.0
    row_gap: float = 12.0
    block_pad: float = 12.0
    block_gap: float = 28.0
    padding: float = 20.0
    frame_width: float = 2.5
    cell_radius: float = 7.0
    frame_radius: float = 10.0

    # Truncation: the most cells drawn along one axis before an ellipsis.
    max_cells: int = 12
    max_visible_cells: int = 240

    # Adaptive colours are resolved by SVG CSS, without another Python render.
    adaptive: bool = False

    def axis_color(self, axis):
        """Return the ramp colour for ``axis``, wrapping if the ramp runs out."""
        ramp = self.axis_colors
        return ramp[axis % len(ramp)]

    def variant(self, **changes):
        """Return a copy of this theme with the given fields replaced."""
        return replace(self, **changes)


LIGHT = Theme(
    name="light",
    background="#f8fafc",
    surface="#ffffff",
    surface_muted="#f1f5f9",
    surface_selected="#15803d",
    text="#0f172a",
    text_muted="#64748b",
    text_selected="#ffffff",
    heading="#0f172a",
    axis_colors=LIGHT_AXIS_RAMP,
    neutral="#cbd5e1",
    cell_border="#e2e8f0",
    selected_border="#166534",
    card_border="#e8edf3",
)

DARK = Theme(
    name="dark",
    background="#0f172a",
    surface="#1e293b",
    surface_muted="#172033",
    surface_selected="#22c55e",
    text="#f8fafc",
    text_muted="#94a3b8",
    text_selected="#0f172a",
    heading="#f8fafc",
    axis_colors=DARK_AXIS_RAMP,
    neutral="#64748b",
    cell_border="#334155",
    selected_border="#16a34a",
    card_border="#1e293b",
)

_LIGHT_EINSUM_RAMPS = {
    "free": ("#2563eb", "#0891b2", "#4f46e5", "#0d9488", "#0284c7"),
    "shared": ("#db2777", "#9333ea", "#c026d3", "#e11d48"),
    "contracted": ("#ea580c", "#ca8a04", "#dc2626", "#b45309"),
}
_DARK_EINSUM_RAMPS = {
    "free": ("#60a5fa", "#22d3ee", "#a5b4fc", "#2dd4bf", "#38bdf8"),
    "shared": ("#f472b6", "#c084fc", "#e879f9", "#fb7185"),
    "contracted": ("#fb923c", "#facc15", "#f87171", "#fbbf24"),
}

# Tokens distinguish semantic roles even when their light colours are equal.
_COLOR_FIELDS = (
    "background", "surface", "surface_muted", "surface_selected", "text",
    "text_muted", "text_selected", "heading", "neutral", "cell_border",
    "selected_border", "card_border",
)
_AUTO_PALETTE = {
    name.replace("_", "-"): (getattr(LIGHT, name), getattr(DARK, name))
    for name in _COLOR_FIELDS
}
_AUTO_PALETTE.update({
    f"axis-{i}": colors for i, colors in enumerate(zip(LIGHT_AXIS_RAMP, DARK_AXIS_RAMP))
})
for _role, _ramp in _LIGHT_EINSUM_RAMPS.items():
    for _i, _colors in enumerate(zip(_ramp, _DARK_EINSUM_RAMPS[_role])):
        _AUTO_PALETTE[f"einsum-{_role}-{_i}"] = _colors


def _auto_color(name):
    """Return a semantic SVG colour with a readable light fallback."""
    return f"var(--rt-{name}, {_AUTO_PALETTE[name][0]})"


_AUTO_FALLBACKS = {_auto_color(name): colors[0] for name, colors in _AUTO_PALETTE.items()}
_AUTO_EINSUM_RAMPS = {
    role: tuple(_auto_color(f"einsum-{role}-{i}") for i in range(len(ramp)))
    for role, ramp in _LIGHT_EINSUM_RAMPS.items()
}

AUTO = LIGHT.variant(
    name="auto",
    adaptive=True,
    axis_colors=tuple(_auto_color(f"axis-{i}") for i in range(len(LIGHT_AXIS_RAMP))),
    **{name: _auto_color(name.replace("_", "-")) for name in _COLOR_FIELDS},
)


def _auto_stylesheet():
    """Scope dark overrides to adaptive SVGs, leaving fixed figures unchanged."""
    declarations = ";".join(f"--rt-{name}:{dark}" for name, (_, dark) in _AUTO_PALETTE.items())
    return (
        '<style>@media (prefers-color-scheme: dark){svg[data-rt-theme="auto"]{'
        + declarations + "}"
        'svg[data-rt-theme="auto"] [data-rt-fill]{fill:var(--rt-fill-dark)}'
        'svg[data-rt-theme="auto"] [data-rt-stroke]{stroke:var(--rt-stroke-dark)}'
        "}</style>"
    )


_PRESETS = {theme.name: theme for theme in (AUTO, LIGHT, DARK)}
# The mutable default theme and axis ramp live in ``config``. ``None`` there
# means use ``AUTO`` and each theme's own ramp.


def register_theme(theme):
    """Register a theme so it can be looked up by name."""
    if not isinstance(theme, Theme):
        raise TypeError(f"expected a Theme, got {type(theme).__name__}")
    _PRESETS[theme.name] = theme
    return theme


def resolve_theme(theme):
    """Resolve ``theme`` into a :class:`Theme`.

    ``None`` selects the current module default. A string is looked up among
    the registered presets. A :class:`Theme` is returned unchanged.
    """
    if theme is None:
        base = config.default_theme if config.default_theme is not None else AUTO
        if config.default_axis_colors is not None:
            return base.variant(axis_colors=config.default_axis_colors)
        return base
    if isinstance(theme, Theme):
        return theme
    if isinstance(theme, str):
        try:
            return _PRESETS[theme]
        except KeyError:
            known = ", ".join(sorted(_PRESETS))
            raise ValueError(
                f"unknown theme {theme!r}, available themes are {known}"
            ) from None
    raise TypeError(
        f"theme must be a Theme, a theme name, or None, got {type(theme).__name__}"
    )


def get_default_theme():
    """Return the current module default theme."""
    return config.default_theme if config.default_theme is not None else AUTO


def set_default_theme(theme):
    """Set the module default theme used when a call passes no theme."""
    config.default_theme = resolve_theme(theme)
    return config.default_theme


def get_default_axis_colors():
    """Return the global axis colour ramp, or ``None`` when none is set."""
    return config.default_axis_colors


def set_default_axis_colors(colors):
    """Set a global axis colour ramp for later renders.

    ``colors`` is a sequence of CSS colour strings, one per axis depth, reused
    by every later call that does not pass its own ``theme``. A per call
    ``theme`` still wins, so the global ramp is a convenient default rather than
    a hard override. Pass ``None`` to clear it and fall back to each theme's
    built-in ramp.
    """
    if colors is None:
        config.default_axis_colors = None
        return None
    ramp = tuple(colors)
    if not ramp:
        raise ValueError("axis colours must be a non-empty sequence of colour strings")
    if not all(isinstance(c, str) for c in ramp):
        raise TypeError("axis colours must all be colour strings")
    config.default_axis_colors = ramp
    return config.default_axis_colors
