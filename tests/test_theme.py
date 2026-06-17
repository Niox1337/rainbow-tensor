"""Tests for the theme system."""

import pytest

import rainbow_tensor as rt
from rainbow_tensor.theme import (
    DARK,
    LIGHT,
    get_default_theme,
    register_theme,
    resolve_theme,
    set_default_theme,
)


def test_presets_have_distinct_backgrounds():
    assert LIGHT.background != DARK.background
    assert LIGHT.name == "light"
    assert DARK.name == "dark"


def test_resolve_accepts_theme_name_and_object():
    assert resolve_theme("dark") is DARK
    assert resolve_theme(LIGHT) is LIGHT


def test_resolve_none_returns_default():
    assert resolve_theme(None) is get_default_theme()


def test_unknown_theme_name_raises():
    with pytest.raises(ValueError):
        resolve_theme("solarized-banana")


def test_set_default_theme_round_trips():
    original = get_default_theme()
    try:
        set_default_theme("dark")
        assert get_default_theme() is DARK
        # a call with no theme now uses the dark default
        assert DARK.background in rt.shape((2, 2)).svg
    finally:
        set_default_theme(original)
    assert get_default_theme() is original


def test_axis_color_wraps_around_ramp():
    ramp_len = len(LIGHT.axis_colors)
    assert LIGHT.axis_color(0) == LIGHT.axis_color(ramp_len)


def test_variant_overrides_only_named_fields():
    big = LIGHT.variant(cell_w=80, max_cells=4)
    assert big.cell_w == 80
    assert big.max_cells == 4
    assert big.background == LIGHT.background
    # the original preset is untouched
    assert LIGHT.cell_w == 48


def test_register_then_resolve_custom_theme():
    custom = LIGHT.variant(name="ocean", background="#001f3f")
    register_theme(custom)
    try:
        assert resolve_theme("ocean") is custom
    finally:
        # leave the registry clean for other tests
        from rainbow_tensor import theme as theme_module

        theme_module._PRESETS.pop("ocean", None)


def test_dark_theme_keeps_text_readable():
    # primary text on the dark surface must stay clearly lighter than the
    # surface, a coarse stand-in for an adequate contrast ratio
    def luminance(hex_color):
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    assert luminance(DARK.text) - luminance(DARK.surface) > 120
    assert luminance(LIGHT.surface) - luminance(LIGHT.text) > 120
