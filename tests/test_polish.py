"""Tests for the 0.3.0 polish features.

Covers truncation of a long axis, the axis legend, hover titles, float
precision, right aligned numbers, and the save helper.
"""

import re

import pytest

import rainbow_tensor as rt
from rainbow_tensor.layout import COL_ELLIPSIS, ELLIPSIS, build_layout, visible_positions


def _width(svg):
    return int(re.search(r'width="(\d+)"', svg).group(1))


def test_visible_positions_keeps_short_axis_whole():
    assert visible_positions(5, 12) == [0, 1, 2, 3, 4]


def test_visible_positions_truncates_long_axis():
    positions = visible_positions(20, 12)
    assert positions[0] == 0
    assert positions[-1] == 19
    assert ELLIPSIS in positions
    # the drawn length never exceeds the limit
    assert len(positions) <= 12


def test_long_axis_draws_ellipsis_cell():
    visual = rt.shape((30,))
    assert COL_ELLIPSIS in visual.svg


def test_long_axis_stays_bounded_in_width():
    narrow = _width(rt.shape((6,)).svg)
    wide = _width(rt.shape((300,)).svg)
    # a far larger axis must not grow the canvas without bound
    assert wide < narrow * 3


def test_long_axis_cells_do_not_overlap():
    layout = build_layout((40,))
    real = [c for c in layout.cells if not c.ellipsis]
    boxes = [(c.x, c.y, c.width, c.height) for c in real]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax, ay, aw, ah = boxes[i]
            bx, by, bw, bh = boxes[j]
            assert ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay


def test_two_dimensional_long_axis_truncates_rows():
    layout = build_layout((30, 3))
    # one frame per visible row, capped by the truncation limit
    assert len(layout.frames) <= 12


def test_legend_names_each_axis():
    svg = rt.shape((2, 3, 4)).svg
    assert "axis 0" in svg
    assert "axis 1" in svg
    assert "axis 2" in svg


def test_legend_uses_axis_colours():
    svg = rt.shape((2, 3)).svg
    # the axis 0 swatch carries the red ramp colour
    assert 'rx="3" fill="#dc2626"' in svg


def test_leaf_legend_swatch_matches_label_colour():
    # in an index view the leaf axis is green on the figure and in the tuple,
    # so its legend swatch must be green too, not a stray grey
    index_svg = rt.index((2, 3), (slice(None), 1)).svg
    assert f'rx="3" fill="{rt.LIGHT.surface_selected}"' in index_svg
    assert f'rx="3" fill="{rt.LIGHT.text_muted}"' not in index_svg
    # in a shape view there is no selection, so the leaf swatch stays muted
    shape_svg = rt.shape((2, 3)).svg
    assert f'rx="3" fill="{rt.LIGHT.text_muted}"' in shape_svg


def test_hover_titles_carry_coordinate_and_flat_index():
    svg = rt.shape((2, 2)).svg
    assert "<title>" in svg
    assert "flat" in svg


def test_float_values_use_precision():
    np = pytest.importorskip("numpy")
    x = np.array([0.123456, 1.5, 2.0])
    two = rt.shape(x, precision=2).svg
    four = rt.shape(x, precision=4).svg
    assert ">0.12<" in two
    assert ">0.1235<" in four


def test_numeric_cells_are_right_aligned():
    svg = rt.shape((4,)).svg
    assert 'text-anchor="end"' in svg


def test_save_writes_svg_to_file(tmp_path):
    visual = rt.shape((2, 2, 2), theme="dark")
    path = tmp_path / "tensor.svg"
    returned = visual.save(path)
    assert returned == path
    assert path.read_text(encoding="utf-8") == visual.svg


def _first_cell_width(svg):
    return int(re.search(r'width="(\d+)" height="\d+" rx="7"', svg).group(1))


def _cell_values(svg):
    return re.findall(r'monospace"[^>]*fill="#0f172a">([^<]+)</text>', svg)


def test_high_precision_value_fits_inside_cell():
    np = pytest.importorskip("numpy")
    x = np.array([0.123456789, 1.0, 2.0])
    svg = rt.shape(x, precision=8).svg
    longest = max(len(v) for v in _cell_values(svg))
    cell_w = _first_cell_width(svg)
    # the widest value plus its padding must fit within the cell it sits in
    assert longest * 15 * 0.6 + 2 * 8 <= cell_w


def test_higher_precision_grows_the_cell():
    np = pytest.importorskip("numpy")
    x = np.array([0.123456789, 1.0, 2.0])
    low = _first_cell_width(rt.shape(x, precision=2).svg)
    high = _first_cell_width(rt.shape(x, precision=8).svg)
    assert high > low


def test_short_values_keep_default_cell_width():
    # integers shorter than the default never shrink the cell below the theme
    svg = rt.shape((2, 3)).svg
    assert _first_cell_width(svg) == rt.LIGHT.cell_w


def test_precision_argument_reaches_index():
    np = pytest.importorskip("numpy")
    x = np.linspace(0.0, 1.0, 6).reshape(2, 3)
    svg = rt.index(x, (0, slice(None)), precision=1).svg
    assert ">0.0<" in svg
