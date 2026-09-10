"""Lazy group colours with stable identities and paired light and dark paints.

Group IDs sample hue and lightness at different rates in OkLCh space. The
sequence does not cycle through a short list, and never allocates a palette
for the full logical tensor. Coordinates and focus remain the precise way to
identify contributions when a figure contains too many similar colours.
"""

import math
from functools import lru_cache


class AdaptivePaint(str):
    """A CSS colour carrying both schemes without a mutable global registry.

    SVG rendering writes the light colour as a presentation attribute and
    stores both colours on the element for scoped media rules. Other renderers
    can use the valid CSS string or inspect ``light`` and ``dark`` explicitly.
    """

    __slots__ = ()

    def __new__(cls, light, dark):
        return super().__new__(cls, f"light-dark({light}, {dark})")

    def __getnewargs__(self):
        """Preserve paired paints when a renderer copies or pickles its options."""
        return self.light, self.dark

    @property
    def light(self):
        """Return the literal light colour used by non-CSS SVG viewers."""
        return self.partition("(")[2].partition(", ")[0]

    @property
    def dark(self):
        """Return the paired colour used when the viewer prefers dark mode."""
        return self.partition(", ")[2][:-1]


def _radical_inverse(value, base):
    """Reverse integer digits before division to avoid rounding the ID itself.

    Integer arithmetic consumes the ID before any float conversion. Different
    bases prevent adjacent groups from varying hue and lightness in lockstep.
    """
    numerator, denominator = 0, 1
    while value:
        value, digit = divmod(value, base)
        numerator = numerator * base + digit
        denominator *= base
    return numerator / denominator


def _linear_rgb(lightness, chroma, hue):
    """Convert OkLCh to linear sRGB using the published Oklab inverse matrix.

    Matrix reference: https://bottosson.github.io/posts/oklab/
    """
    angle = math.tau * hue
    a, b = chroma * math.cos(angle), chroma * math.sin(angle)
    cone_l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    cone_m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    cone_s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        4.0767416621 * cone_l - 3.3077115913 * cone_m + 0.2309699292 * cone_s,
        -1.2684380046 * cone_l + 2.6097574011 * cone_m - 0.3413193965 * cone_s,
        -0.0041960863 * cone_l - 0.7034186147 * cone_m + 1.7076147010 * cone_s,
    )


def _oklch_hex(lightness, chroma, hue):
    """Fit a colour into sRGB by reducing chroma while retaining hue and tone."""
    rgb = _linear_rgb(lightness, chroma, hue)
    if not all(0 <= channel <= 1 for channel in rgb):
        low, high = 0.0, chroma
        for _ in range(14):
            middle = (low + high) / 2
            candidate = _linear_rgb(lightness, middle, hue)
            if all(0 <= channel <= 1 for channel in candidate):
                low = middle
            else:
                high = middle
        rgb = _linear_rgb(lightness, low, hue)
    encoded = (
        12.92 * channel if channel <= 0.0031308 else 1.055 * channel ** (1 / 2.4) - 0.055
        for channel in rgb
    )
    return "#" + "".join(f"{round(max(0, min(1, channel)) * 255):02x}" for channel in encoded)


@lru_cache(maxsize=1024)
def _group_palette(index):
    """Compute one group's paired paints with a fixed upper bound on caching."""
    if index < 0:
        raise ValueError("group index must be nonnegative")
    hue = (_radical_inverse(index, 2) + 2 / 3) % 1
    tone = _radical_inverse(index, 3)
    chroma = 0.08 - 0.02 * _radical_inverse(index, 5)
    light = (
        _oklch_hex(0.88 + 0.07 * tone, chroma, hue),
        _oklch_hex(0.55 + 0.12 * tone, 0.15, hue),
    )
    dark = (
        _oklch_hex(0.30 + 0.10 * tone, chroma, hue),
        _oklch_hex(0.65 + 0.14 * tone, 0.13, hue),
    )
    return light, dark


def _hex_luminance(color):
    """Read common hex theme colours without trying to interpret arbitrary CSS."""
    if not isinstance(color, str) or not color.startswith("#"):
        return None
    value = color[1:]
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    if len(value) != 6:
        return None
    try:
        channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    except ValueError:
        return None
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), linear))


def group_tint(theme, index):
    """Return a stable fill and border for a logical group or operand ID.

    Focus and preview membership do not affect the colour. Fixed themes choose
    the fill with better contrast against their actual text colour. This keeps
    cell values readable when only the canvas background changes. Unrecognised
    CSS text colours fall back to the background brightness, then the preset name.
    """
    light, dark = _group_palette(index)
    if theme.adaptive:
        return tuple(AdaptivePaint(a, b) for a, b in zip(light, dark))
    text = _hex_luminance(theme.text)
    if text is not None:
        def contrast(tint):
            fill = _hex_luminance(tint[0])
            return (max(fill, text) + 0.05) / (min(fill, text) + 0.05)

        return max((light, dark), key=contrast)
    background = _hex_luminance(theme.background)
    is_dark = background < 0.18 if background is not None else theme.name == "dark"
    return dark if is_dark else light
