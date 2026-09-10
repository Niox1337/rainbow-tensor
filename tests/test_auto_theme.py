"""Adaptive SVGs keep readable fallbacks without evaluating a tensor twice."""

import xml.etree.ElementTree as ET

import pytest

import rainbow_tensor as rt
from rainbow_tensor import config
from rainbow_tensor.render_svg import render_panels, render_svg
from rainbow_tensor.theme import AUTO, DARK, LIGHT, get_default_theme, resolve_theme

SVG = "{http://www.w3.org/2000/svg}"


def _root(svg):
    return ET.fromstring(svg)


def _painted(root, token):
    """Find rendered elements whose inline paint refers to a semantic token."""
    return [element for element in root.iter() if token in element.get("style", "")]


class CountingArray:
    """Record backend reads without requiring a second tensor representation."""

    def __init__(self, shape):
        self.shape = shape
        self.reads = []

    def __getitem__(self, coord):
        assert len(coord) == len(self.shape)
        assert all(0 <= index < size for index, size in zip(coord, self.shape))
        self.reads.append(coord)
        return 7


def test_auto_is_the_unconfigured_default(monkeypatch):
    monkeypatch.setattr(config, "default_theme", None)
    monkeypatch.setattr(config, "default_axis_colors", None)
    assert resolve_theme("auto") is AUTO
    assert resolve_theme(None) is AUTO
    assert get_default_theme() is AUTO
    assert _root(rt.shape((2, 3)).svg).get("data-rt-theme") == "auto"


def test_auto_variant_keeps_adaptation_when_renamed():
    theme = AUTO.variant(name="roomy", cell_w=80)
    root = _root(rt.shape((2,), theme=theme).svg)
    assert theme.adaptive
    assert root.get("data-rt-theme") == "auto"
    assert root.find(f"{SVG}style") is not None
    assert any(rect.get("width") == "80" for rect in root.iter(f"{SVG}rect"))


def test_auto_styles_are_scoped_and_distinguish_equal_light_colours():
    root = _root(render_svg((2,), selected=[(0,)], theme=AUTO))
    style = root.find(f"{SVG}style").text
    assert style.startswith('@media (prefers-color-scheme: dark){svg[data-rt-theme="auto"]{')
    assert ":root" not in style
    assert f"--rt-background:{DARK.background}" in style
    assert f"--rt-surface:{DARK.surface}" in style
    assert f"--rt-text-selected:{DARK.text_selected}" in style
    selected_text = _painted(root, "var(--rt-text-selected,")
    assert selected_text
    assert all(element.get("fill") == LIGHT.text_selected for element in selected_text)
    resting_root = _root(render_svg((2,), theme=AUTO))
    resting_cells = _painted(resting_root, "var(--rt-surface,")
    assert resting_cells
    assert all(element.get("fill") == LIGHT.surface for element in resting_cells)
    assert LIGHT.surface == LIGHT.text_selected
    assert DARK.surface != DARK.text_selected


@pytest.mark.parametrize("theme", [LIGHT, DARK, LIGHT.variant(name="ocean", surface="navy")])
def test_fixed_themes_remain_literal_and_have_no_media_rules(theme):
    root = _root(rt.shape((2, 3), theme=theme).svg)
    assert root.get("data-rt-theme") is None
    assert root.find(f"{SVG}style") is None
    assert root.find(f"{SVG}rect").get("fill") == theme.background
    assert any(element.get("fill") == theme.surface for element in root.iter(f"{SVG}rect"))
    assert not any("--rt-" in element.get("style", "") for element in root.iter())


def test_auto_custom_axis_ramp_is_used_without_substitution(monkeypatch):
    ramp = ("rebeccapurple", "rgb(12, 34, 56)", "var(--custom-axis, #123456)")
    monkeypatch.setattr(config, "default_theme", None)
    monkeypatch.setattr(config, "default_axis_colors", ramp)
    theme = resolve_theme(None)
    assert theme.adaptive
    assert theme.axis_colors == ramp
    root = _root(rt.shape((2, 2, 2, 2)).svg)
    strokes = {element.get("stroke") for element in root.iter(f"{SVG}rect")}
    assert set(ramp) <= strokes
    assert not any("var(--rt-axis-" in element.get("style", "") for element in root.iter())
    fixed_root = _root(rt.shape((2, 2), theme="light").svg)
    assert any(element.get("stroke") == LIGHT.axis_color(0) for element in fixed_root.iter())


def test_auto_custom_surface_stays_literal():
    theme = AUTO.variant(surface="papayawhip")
    root = _root(rt.shape((2,), theme=theme).svg)
    cells = [element for element in root.iter(f"{SVG}rect") if element.get("fill") == "papayawhip"]
    assert len(cells) == 2
    assert all("fill:" not in element.get("style", "") for element in cells)
    assert root.get("data-rt-theme") == "auto"


def test_auto_export_contains_self_contained_css_and_literal_fallbacks(tmp_path):
    visual = rt.index((2, 3), (1, slice(None)), theme="auto")
    path = tmp_path / "adaptive.svg"
    visual.save(path)
    assert path.read_text(encoding="utf-8") == visual.svg
    root = _root(path.read_text(encoding="utf-8"))
    assert root.find(f"{SVG}style") is not None
    paints = [value for element in root.iter() for name, value in element.attrib.items()
              if name in ("fill", "stroke")]
    assert paints
    assert all("var(" not in paint for paint in paints)
    assert root.find(f"{SVG}rect").get("fill") == LIGHT.background
    assert not list(root.iter(f"{SVG}script"))


def test_auto_does_not_add_backend_reads():
    fixed = CountingArray((2, 3))
    adaptive = CountingArray((2, 3))
    rt.shape(fixed, theme="light")
    visual = rt.shape(adaptive, theme="auto")
    assert adaptive.reads == fixed.reads
    assert len(adaptive.reads) == 6
    assert "prefers-color-scheme" in visual.svg


@pytest.mark.parametrize("shape, count", [((), 1), ((0, 10**30), 0), ((10**30, 0), 0)])
def test_auto_preserves_scalar_and_empty_read_contracts(shape, count):
    array = CountingArray(shape)
    root = _root(rt.shape(array, theme="auto").svg)
    assert len(array.reads) == count
    assert root.get("data-rt-theme") == "auto"
    if count == 0:
        assert any(element.get("role") == "note" for element in root.iter())


def test_operand_tints_adapt_with_literal_fallbacks():
    visual = rt.concatenate([(2,), (2,)], 0, theme="auto")
    root = _root(visual.svg)
    light = _root(rt.concatenate([(2,), (2,)], 0, theme="light").svg)
    dark = _root(rt.concatenate([(2,), (2,)], 0, theme="dark").svg)

    def cells(document):
        return [
            group.find(f"{SVG}rect") for group in document.iter(f"{SVG}g")
            if group.find(f"{SVG}title") is not None
        ]

    for automatic, fixed_light, fixed_dark in zip(cells(root), cells(light), cells(dark)):
        assert automatic.get("fill") == fixed_light.get("fill")
        assert f'--rt-fill-dark:{fixed_dark.get("fill")}' in automatic.get("style")
        assert automatic.get("data-rt-fill") == ""
    assert '[data-rt-fill]{fill:var(--rt-fill-dark)}' in root.find(f"{SVG}style").text
    assert all(element.get("data-rt-fill") is None for element in dark.iter())


def test_einsum_role_colours_adapt_in_cells_frames_and_captions():
    root = _root(rt.einsum("bij,bjk->bik", (2, 2, 3), (2, 3, 2), theme="auto").svg)
    for role in ("free", "shared", "contracted"):
        elements = _painted(root, f"var(--rt-einsum-{role}-")
        assert any(element.tag == f"{SVG}tspan" for element in elements)
        assert any(element.tag == f"{SVG}rect" for element in elements)
    fixed = rt.einsum("ij,jk->ik", (2, 3), (3, 2), theme="light")
    assert "--rt-einsum-" not in fixed.svg


def test_auto_panel_in_fixed_document_still_gets_css():
    root = _root(render_panels([
        {"shape": (1,), "theme": AUTO},
        {"shape": (1,), "theme": LIGHT},
    ], theme=LIGHT))
    assert root.get("data-rt-theme") == "auto"
    assert root.find(f"{SVG}style") is not None
    background = root.find(f"{SVG}rect")
    assert background.get("fill") == LIGHT.background
    assert background.get("style") is None
    cells = [element for element in root.iter(f"{SVG}rect") if element.get("width") == "48"]
    assert len(cells) == 2
    assert "var(--rt-surface," in cells[0].get("style", "")
    assert cells[1].get("style") is None


@pytest.mark.parametrize("document, overrides", [
    (LIGHT, (AUTO, DARK)),
    (DARK, (AUTO, LIGHT)),
    (AUTO, (LIGHT, DARK)),
])
def test_mixed_panel_backgrounds_cover_captions_without_more_reads(document, overrides):
    reads = []

    def value(coord):
        reads.append(coord)
        return 7

    root = _root(render_panels([
        {"shape": (1,), "theme": theme, "value_fn": value,
         "caption_parts": [(f"Panel {i}", theme.heading)]}
        for i, theme in enumerate(overrides)
    ], theme=document))
    backgrounds = root.findall(f"{SVG}rect")
    assert len(backgrounds) == 3
    for background, theme in zip(backgrounds[1:], overrides):
        expected = LIGHT.background if theme.adaptive else theme.background
        assert background.get("fill") == expected
        assert background.get("height") == root.get("height")
        if theme.adaptive:
            assert "var(--rt-background," in background.get("style", "")
        else:
            assert background.get("style") is None
    captions = [element for element in root.findall(f"{SVG}text")
                if element.get("font-size") == "14"]
    assert len(captions) == 2
    assert all(float(caption.get("y")) < float(root.get("height")) for caption in captions)
    assert reads == [(0,), (0,)]


def test_matching_panel_backgrounds_do_not_add_rectangles():
    for theme in (AUTO, LIGHT, DARK):
        root = _root(rt.concatenate([(2,), (2,)], 0, theme=theme).svg)
        assert len(root.findall(f"{SVG}rect")) == 1
