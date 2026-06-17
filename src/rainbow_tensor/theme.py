"""Theme objects.

A :class:`Theme` bundles every visual choice the renderer needs into one
immutable value: colours, fonts, cell geometry, stroke width, corner radius,
and the truncation limit. Two presets ship with the package, a clean
``LIGHT`` theme and a high contrast ``DARK`` theme. A module level default is
used when a call does not pass its own theme, and either preset can be looked
up by name through :func:`resolve_theme`.

The axis colours form a rainbow ramp keyed by axis depth, which is where the
package gets its name. Axis 0 is red, axis 1 is orange, and deeper axes walk
through amber, green, blue, violet, and pink. Only the non leaf axes draw a
frame today, but the full ramp is ready for the higher dimensional work in a
later milestone.
"""

from dataclasses import dataclass, field, replace

# Multi-word font names are wrapped in single quotes so the value stays valid
# inside a double-quoted SVG attribute. Double quotes here would break the XML.
SANS_FAMILY = (
    "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)
MONO_FAMILY = "ui-monospace, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

# A full rainbow ramp keyed by axis depth, fitting the project name.
LIGHT_AXIS_RAMP = (
    "#dc2626",  # red
    "#ea580c",  # orange
    "#d97706",  # amber
    "#16a34a",  # green
    "#2563eb",  # blue
    "#7c3aed",  # violet
    "#db2777",  # pink
)
DARK_AXIS_RAMP = (
    "#f87171",
    "#fb923c",
    "#fbbf24",
    "#4ade80",
    "#60a5fa",
    "#a78bfa",
    "#f472b6",
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
    surface_selected="#16a34a",
    text="#0f172a",
    text_muted="#64748b",
    text_selected="#ffffff",
    heading="#0f172a",
    axis_colors=LIGHT_AXIS_RAMP,
    neutral="#cbd5e1",
    cell_border="#e2e8f0",
    selected_border="#15803d",
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
    neutral="#475569",
    cell_border="#334155",
    selected_border="#16a34a",
    card_border="#1e293b",
)

_PRESETS = {LIGHT.name: LIGHT, DARK.name: DARK}
_default_theme = LIGHT


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
        return _default_theme
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
    return _default_theme


def set_default_theme(theme):
    """Set the module default theme used when a call passes no theme."""
    global _default_theme
    _default_theme = resolve_theme(theme)
    return _default_theme
