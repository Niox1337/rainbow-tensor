"""Colour conversion and bounded palette behaviour independent of SVG markup."""

import copy
import pickle

import pytest

from rainbow_tensor import AUTO, DARK, LIGHT
from rainbow_tensor.colors import _group_palette, _hex_luminance, _oklch_hex, group_tint


@pytest.mark.parametrize(("lightness", "expected"), [(0, "#000000"), (1, "#ffffff")])
def test_neutral_oklch_endpoints(lightness, expected):
    assert _oklch_hex(lightness, 0, 0) == expected


def test_reference_srgb_red_in_oklch():
    assert _oklch_hex(0.627955, 0.257683, 29.233885 / 360) == "#ff0000"


@pytest.mark.parametrize("theme", [LIGHT, DARK])
def test_full_default_preview_palette_has_distinct_readable_fills(theme):
    fills = [group_tint(theme, index)[0] for index in range(240)]
    assert len(set(fills)) == 240
    text = _hex_luminance(theme.text)
    for fill in fills:
        background = _hex_luminance(fill)
        contrast = (max(background, text) + 0.05) / (min(background, text) + 0.05)
        assert contrast >= 4.5
        assert fill != theme.surface_selected


def test_palette_cache_cannot_grow_with_the_logical_tensor():
    _group_palette.cache_clear()
    for index in range(1100):
        group_tint(LIGHT, index)
    assert _group_palette.cache_info().currsize <= 1024


def test_adaptive_paints_survive_renderer_option_copies():
    paint = group_tint(AUTO, 12)[0]
    for restored in (copy.deepcopy(paint), pickle.loads(pickle.dumps(paint))):
        assert restored == paint
        assert restored.light == group_tint(LIGHT, 12)[0]
        assert restored.dark == group_tint(DARK, 12)[0]


def test_custom_css_background_uses_known_text_brightness():
    theme = DARK.variant(name="midnight", background="navy")
    assert group_tint(theme, 9) == group_tint(DARK, 9)


@pytest.mark.parametrize(("theme", "expected"), [
    (LIGHT.variant(name="ocean", background="#001f3f"), LIGHT),
    (DARK.variant(background="#ffffff"), DARK),
])
@pytest.mark.parametrize("index", [0, 9, 239])
def test_canvas_background_changes_preserve_contrast_with_actual_cell_text(theme, expected, index):
    tint = group_tint(theme, index)
    assert tint == group_tint(expected, index)
    fill = _hex_luminance(tint[0])
    text = _hex_luminance(theme.text)
    assert (max(fill, text) + 0.05) / (min(fill, text) + 0.05) >= 4.5


@pytest.mark.parametrize(("theme", "expected"), [
    (LIGHT.variant(background="#001f3f", text="var(--cell-text)"), DARK),
    (DARK.variant(background="navy", text="var(--cell-text)"), DARK),
    (LIGHT.variant(background="var(--canvas)", text="var(--cell-text)"), LIGHT),
])
def test_unparseable_text_uses_background_then_preset_name(theme, expected):
    assert group_tint(theme, 9) == group_tint(expected, 9)
