"""Tests for SVG rendering."""

from rainbow_tensor.render_svg import escape, render_svg, svg_document


def test_escape_replaces_markup_characters():
    assert escape("<a & b>") == "&lt;a &amp; b&gt;"


def test_svg_document_is_well_formed():
    doc = svg_document("<rect/>", 100, 50)
    assert doc.startswith("<svg")
    assert doc.endswith("</svg>")
    assert 'width="100"' in doc


def test_render_svg_starts_with_svg_tag():
    svg = render_svg((3,), label="Shape (3)")
    assert svg.startswith("<svg")


def test_render_svg_includes_label_and_values():
    svg = render_svg((2, 2, 2), label="Shape (2, 2, 2)")
    assert "Shape (2, 2, 2)" in svg
    assert ">7<" in svg


def test_render_svg_draws_one_rect_per_cell_plus_blocks():
    svg = render_svg((2, 2, 2))
    assert svg.count("<rect") == 8 + 2


def test_render_svg_highlights_selected_cells():
    selected = [(0, 0, 1), (0, 1, 1)]
    svg = render_svg((2, 2, 2), selected=selected)
    assert "#fde68a" in svg


def test_render_svg_dims_unselected_cells_when_selection_exists():
    svg = render_svg((2, 2, 2), selected=[(0, 0, 1)])
    assert 'opacity="0.30"' in svg


def test_render_svg_includes_explanation_lines():
    svg = render_svg(
        (2, 2, 2),
        selected=[(0, 0, 1), (0, 1, 1)],
        label="Index [0, :, 1]",
        explanation=["Result shape: (2,)"],
    )
    assert "Result shape: (2,)" in svg
    assert "Index [0, :, 1]" in svg
