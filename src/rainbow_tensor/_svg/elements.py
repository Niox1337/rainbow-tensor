"""SVG primitives for paint, text measurement, formatting, and document wrapping."""

import unicodedata

from ..explanations import get_resolved_language
from ..explanations import t as translate
from ..theme import _AUTO_FALLBACKS, LIGHT, _auto_stylesheet

LABEL_HEIGHT = 34
LINE_HEIGHT = 20
LEGEND_HEIGHT = 30
LEGEND_GAP = 18
SWATCH = 13
TEXT_MARGIN = 20
LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 13
EXPLANATION_FONT_SIZE = 13
VALUE_FONT_SIZE = 15
VALUE_PAD = 10
# Approximate glyph width relative to the font size, per family.
MONO_CHAR_RATIO = 0.6
SANS_CHAR_RATIO = 0.55

def escape(text):
    """Escape a value so it is safe inside SVG text and attributes."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

def _paint_attributes(**colors):
    """Keep light presentation attributes when adaptive CSS is unavailable.

    Each adaptive paint also receives an inline CSS declaration. A browser
    resolves its semantic token when the preferred scheme changes, while SVG
    readers without CSS custom properties retain the literal light colour.
    Custom colour strings pass through without palette substitution.
    """
    attributes = []
    styles = []
    for name, color in colors.items():
        fallback = _AUTO_FALLBACKS.get(color, color)
        attributes.append(f'{name}="{escape(fallback)}"')
        if color in _AUTO_FALLBACKS:
            styles.append(f"{name}:{color}")
    if styles:
        attributes.append(f'style="{escape(";".join(styles))}"')
    return " ".join(attributes)

def svg_document(content, width, height, theme=None, aria_label=None):
    """Wrap rendered elements in a complete SVG document.

    A background rectangle in the theme colour fills the whole canvas so the
    dark preset reads as dark even on a transparent host page.
    Adaptive palettes include scoped CSS in the exported SVG itself.
    """
    t = theme or LIGHT
    accessible_name = aria_label or translate("svg.visualisation")
    language = get_resolved_language()
    language_attribute = f'lang="{escape(language)}" ' if language != "en" else ""
    background = (
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" '
        f'rx="14" {_paint_attributes(fill=t.background, stroke=t.card_border)}/>'
    )
    adaptive_attribute = 'data-rt-theme="auto" ' if t.adaptive else ""
    stylesheet = _auto_stylesheet() if t.adaptive else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'{adaptive_attribute}'
        f'{language_attribute}'
        f'role="img" aria-label="{escape(accessible_name)}" '
        f'width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" '
        f'style="max-width:100%;height:auto" '
        f'font-family="{t.sans_family}">'
        f"{stylesheet}{background}{content}"
        f"</svg>"
    )

def format_value(value, precision):
    """Format a cell value for display.

    Floats use a fixed number of decimals so a column lines up. Integers and
    other values are shown as their plain string.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{precision}f}"
    return str(value)

def _is_numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def _text_width(char_count, font_size, ratio):
    """Estimate width using full glyph cells for CJK and zero for combining marks.

    Numeric character counts remain supported for existing renderer callers.
    """
    if isinstance(char_count, str):
        char_count = sum(
            0 if unicodedata.combining(char) else
            2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
            for char in char_count
        )
    return char_count * font_size * ratio

def _label_element(x, y, parts):
    """Build a label text element from coloured parts."""
    spans = "".join(
        f'<tspan {_paint_attributes(fill=color)}>{escape(text)}</tspan>' for text, color in parts
    )
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" font-size="{LABEL_FONT_SIZE}" '
        f'font-weight="700">{spans}</text>'
    )

def _centered_parts(cx, y, parts, font_size, weight="700"):
    """Render coloured ``parts`` centred on ``cx`` at baseline ``y``."""
    spans = "".join(
        f'<tspan {_paint_attributes(fill=color)}>{escape(text)}</tspan>' for text, color in parts
    )
    return (
        f'<text x="{cx:.0f}" y="{y:.0f}" text-anchor="middle" '
        f'font-size="{font_size}" font-weight="{weight}">{spans}</text>'
    )
