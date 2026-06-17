"""Tests for SVG rendering."""

import xml.etree.ElementTree as ET

import pytest

import rainbow_tensor as rt
from rainbow_tensor.render_svg import (
    AXIS_FRAME_COLORS,
    SELECT_VALUE_COLOR,
    escape,
    render_svg,
    svg_document,
)


@pytest.mark.parametrize(
    "make",
    [
        lambda: render_svg((3,)),
        lambda: render_svg((2, 3)),
        lambda: render_svg((2, 2, 2)),
        lambda: render_svg((2, 2, 2), theme="dark"),
        lambda: render_svg((30,)),
        lambda: rt.show_index((2, 2, 2), (0, slice(None), 1)).svg,
        lambda: rt.show_index((3, 4), (1, slice(None)), theme="dark").svg,
    ],
)
def test_rendered_svg_is_well_formed_xml(make):
    # Parse the SVG as strict XML. A malformed attribute, such as an unescaped
    # quote inside font-family, would raise here even though browsers recover.
    svg = make()
    ET.fromstring(svg)


def test_escape_replaces_markup_characters():
    assert escape("<a & b>") == "&lt;a &amp; b&gt;"


def test_svg_document_is_well_formed():
    doc = svg_document("<rect/>", 100, 50)
    assert doc.startswith("<svg")
    assert doc.endswith("</svg>")
    assert 'width="100"' in doc


def test_render_svg_starts_with_svg_tag():
    svg = render_svg((3,))
    assert svg.startswith("<svg")


def test_render_svg_includes_values():
    svg = render_svg((2, 2, 2))
    assert ">7<" in svg


def test_shape_view_colours_frames_by_axis():
    svg = render_svg((2, 2, 2))
    assert AXIS_FRAME_COLORS[0] in svg
    assert AXIS_FRAME_COLORS[1] in svg


def test_three_dimensional_shape_draws_six_frames():
    # two axis-0 block frames plus four axis-1 row frames; frames are the only
    # rects drawn with no fill, so counting them ignores cells and the card
    svg = render_svg((2, 2, 2))
    assert svg.count('fill="none"') == 6


def test_index_view_highlights_selected_values_green():
    svg = render_svg((2, 2, 2), selected=[(0, 0, 1), (0, 1, 1)])
    assert SELECT_VALUE_COLOR in svg


def test_label_parts_render_coloured_tspans():
    svg = render_svg((3,), label_parts=[("Shape (", "#000000"), ("3", "#dc2626")])
    assert "<tspan" in svg
    assert 'fill="#dc2626"' in svg


def test_width_grows_to_fit_long_explanation():
    import re

    long_line = "Axis 0 is removed because integer index 0 is used and it is long"
    svg = render_svg((2,), label="Index (0,)", explanation=[long_line])
    width = int(re.search(r'width="(\d+)"', svg).group(1))
    # the explanation is far wider than the tiny 1D tensor, so the canvas must
    # have grown well beyond the default tensor width to avoid clipping
    assert width > len(long_line) * 7


def test_render_svg_includes_explanation_lines():
    svg = render_svg(
        (2, 2, 2),
        selected=[(0, 0, 1), (0, 1, 1)],
        label="Index (0, :, 1)",
        explanation=["Result shape: (2,)"],
    )
    assert "Result shape: (2,)" in svg
