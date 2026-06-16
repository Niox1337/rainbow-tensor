"""Tests for SVG rendering."""

from rainbow_tensor.render_svg import (
    AXIS_FRAME_COLORS,
    SELECT_VALUE_COLOR,
    escape,
    render_svg,
    svg_document,
)


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
    # two axis-0 block frames plus four axis-1 row frames
    svg = render_svg((2, 2, 2))
    assert svg.count("<rect") == 6


def test_index_view_highlights_selected_values_green():
    svg = render_svg((2, 2, 2), selected=[(0, 0, 1), (0, 1, 1)])
    assert SELECT_VALUE_COLOR in svg


def test_label_parts_render_coloured_tspans():
    svg = render_svg((3,), label_parts=[("Shape (", "#000000"), ("3", "#dc2626")])
    assert "<tspan" in svg
    assert 'fill="#dc2626"' in svg


def test_render_svg_includes_explanation_lines():
    svg = render_svg(
        (2, 2, 2),
        selected=[(0, 0, 1), (0, 1, 1)],
        label="Index (0, :, 1)",
        explanation=["Result shape: (2,)"],
    )
    assert "Result shape: (2,)" in svg
