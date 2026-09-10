"""Reduction groups retain distinct, readable colours without eager work."""

import math
import tracemalloc

import pytest

import rainbow_tensor as rt
from rainbow_tensor.layout import build_layout


class RecordingRenderer:
    """Capture callbacks without reading any source or result values."""

    name = "group-colour-recording"
    mime_type = "text/plain"

    def render_tensor(self, **kwargs):
        raise AssertionError("A reduction must render both panels")

    def render_panels(self, *, panels, theme, **kwargs):
        self.panels = panels
        self.theme = theme
        return "recorded groups"


class PreviewRenderer(RecordingRenderer):
    """Read and colour only the cells admitted by each panel's layout."""

    def render_panels(self, *, panels, theme, **kwargs):
        content = super().render_panels(panels=panels, theme=theme, **kwargs)
        self.cells = []
        for panel in panels:
            layout = build_layout(
                panel["shape"], selected=panel.get("selected"),
                value_fn=panel.get("value_fn"), theme=panel.get("theme", theme),
            )
            visible = [cell for cell in layout.cells if not cell.ellipsis]
            for cell in visible:
                panel["cell_tint"](cell.coord)
            self.cells.append(visible)
        return content


class CountingArray:
    """Represent a large shape while recording scalar reads only."""

    def __init__(self, shape):
        self.shape = shape
        self.reads = 0

    def __getitem__(self, coordinate):
        assert len(coordinate) == len(self.shape)
        assert all(0 <= index < size for index, size in zip(coordinate, self.shape))
        self.reads += 1
        return 1


def _luminance(colour):
    """Compute relative sRGB luminance from a literal hexadecimal colour."""
    assert colour.startswith("#") and len(colour) == 7
    channels = [int(colour[offset:offset + 2], 16) / 255 for offset in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return sum(weight * value for weight, value in zip((0.2126, 0.7152, 0.0722), linear))


def _contrast(first, second):
    values = sorted((_luminance(first), _luminance(second)))
    return (values[1] + 0.05) / (values[0] + 0.05)


@pytest.mark.parametrize("theme", [rt.LIGHT, rt.DARK, rt.AUTO])
@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_sixteen_groups_have_distinct_actual_colours_and_matching_sources(theme, operation):
    renderer = RecordingRenderer()
    operation((2, 17), axis=0, focus=(16,), theme=theme, renderer=renderer)
    source, result = renderer.panels
    tints = [result["cell_tint"]((group,)) for group in range(16)]

    assert len({fill for fill, _ in tints}) == 16
    assert len({border for _, border in tints}) == 16
    for group, tint in enumerate(tints):
        assert source["cell_tint"]((0, group)) == tint
        assert source["cell_tint"]((1, group)) == tint
    assert source["cell_tint"]((0, 16)) is None
    assert result["cell_tint"]((16,)) is None

    if theme.adaptive:
        for mode in ("light", "dark"):
            assert len({getattr(fill, mode) for fill, _ in tints}) == 16
            assert len({getattr(border, mode) for _, border in tints}) == 16
    else:
        assert all(fill != theme.surface_selected for fill, _ in tints)


@pytest.mark.parametrize("theme", [rt.LIGHT, rt.DARK, rt.AUTO])
def test_focus_changes_do_not_recolour_other_groups_in_truncated_previews(theme):
    theme = theme.variant(max_cells=8, max_visible_cells=20)
    previous = {}
    for focus in (0, 50, 99):
        renderer = RecordingRenderer()
        rt.sum((2, 100), axis=0, focus=(focus,), theme=theme, renderer=renderer)
        source, result = renderer.panels
        layout = build_layout(result["shape"], selected=result["selected"], theme=theme)
        visible = {cell.coord[0] for cell in layout.cells if not cell.ellipsis}
        assert focus in visible
        for group in sorted(visible | {0, 1, 49, 97, 99}, reverse=True):
            if group == focus:
                continue
            tint = result["cell_tint"]((group,))
            assert source["cell_tint"]((1, group)) == tint
            if group in previous:
                assert tint == previous[group]
            previous[group] = tint


@pytest.mark.parametrize("theme", [rt.LIGHT, rt.DARK])
@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
@pytest.mark.parametrize("keepdims", [False, True])
def test_multi_axis_group_colours_follow_result_coordinates(theme, operation, keepdims):
    renderer = RecordingRenderer()
    focus = (0, 5, 0, 2) if keepdims else (5, 2)
    operation(
        (2, 17, 3, 4), axis=(0, 2), keepdims=keepdims, focus=focus,
        theme=theme, renderer=renderer,
    )
    source, result = renderer.panels
    for row, column in ((0, 0), (1, 3), (15, 2), (16, 3)):
        output = (0, row, 0, column) if keepdims else (row, column)
        tint = result["cell_tint"](output)
        assert source["cell_tint"]((0, row, 2, column)) == tint
        assert source["cell_tint"]((1, row, 0, column)) == tint
    assert result["cell_tint"](focus) is None
    assert source["cell_tint"]((1, 5, 2, 2)) is None


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_no_axis_reduction_matches_each_source_cell_to_its_own_group(operation):
    renderer = RecordingRenderer()
    operation((3, 6), axis=(), focus=(2, 5), theme=rt.LIGHT, renderer=renderer)
    source, result = renderer.panels
    tints = []
    for row in range(3):
        for column in range(6):
            coordinate = (row, column)
            assert source["cell_tint"](coordinate) == result["cell_tint"](coordinate)
            if coordinate != (2, 5):
                tints.append(result["cell_tint"](coordinate))
    assert len({fill for fill, _ in tints}) == 17


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_empty_groups_have_distinct_output_colours_without_source_reads(operation):
    array = CountingArray((0, 17))
    renderer = RecordingRenderer()
    visual = operation(
        array, axis=0, focus=(16,), theme=rt.LIGHT.variant(max_cells=20), renderer=renderer,
    )
    result = renderer.panels[1]
    assert len({result["cell_tint"]((group,))[0] for group in range(16)}) == 16
    for group in range(17):
        value = result["value_fn"]((group,))
        assert math.isnan(value) if operation is rt.mean else value == 0
    assert visual.trace.term_count == 0
    assert array.reads == 0


@pytest.mark.parametrize("shape, expected_trace", [((), True), ((2, 0), False)])
def test_scalar_and_empty_results_need_no_ordinary_group_palette(shape, expected_trace):
    renderer = PreviewRenderer()
    array = CountingArray(shape)
    visual = rt.sum(array, axis=() if not shape else 0, theme=rt.DARK, renderer=renderer)
    assert (visual.trace is not None) is expected_trace
    if not shape:
        assert renderer.panels[1]["cell_tint"](()) is None
        assert len(renderer.cells[1]) == 1
    else:
        assert renderer.cells == [[], []]
        assert array.reads == 0


@pytest.mark.parametrize("theme", [
    rt.LIGHT, rt.DARK, rt.DARK.variant(name="custom-dark"),
])
def test_group_fills_keep_cell_text_readable_in_fixed_and_renamed_themes(theme):
    renderer = RecordingRenderer()
    rt.sum((2, 17), axis=0, focus=(16,), theme=theme, renderer=renderer)
    for group in range(16):
        fill, _ = renderer.panels[1]["cell_tint"]((group,))
        assert _contrast(fill, theme.text) >= 4.5


@pytest.mark.parametrize("size", [2**54 + 37, 10**30 + 37])
@pytest.mark.parametrize("theme", [rt.LIGHT, rt.DARK])
def test_huge_group_ids_keep_distinct_colours_with_bounded_memory_and_no_reads(size, theme):
    array = CountingArray((2, size))
    renderer = RecordingRenderer()
    tracemalloc.start()
    try:
        rt.sum(array, axis=0, theme=theme, renderer=renderer)
        source, result = renderer.panels
        groups = [1, 2, 3, 4, size - 4, size - 3, size - 2, size - 1]
        tints = [result["cell_tint"]((group,)) for group in groups]
        assert len({fill for fill, _ in tints}) == len(groups)
        for group, tint in zip(groups, tints):
            assert source["cell_tint"]((1, group)) == tint
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000
    assert array.reads == 0


@pytest.mark.parametrize("theme", [rt.LIGHT, rt.DARK, rt.AUTO])
def test_preview_colouring_adds_no_backend_reads_when_result_values_are_skipped(theme):
    theme = theme.variant(max_cells=8, max_visible_cells=20)
    array = CountingArray((2, 10**30))
    renderer = PreviewRenderer()
    visual = rt.sum(array, axis=0, theme=theme, max_terms=1, renderer=renderer)
    assert visual.metadata["value_evaluation"]["status"] == "skipped"
    assert array.reads == len(renderer.cells[0])
    assert 0 < array.reads <= theme.max_visible_cells
    assert all(cell.value == "?" for cell in renderer.cells[1])
