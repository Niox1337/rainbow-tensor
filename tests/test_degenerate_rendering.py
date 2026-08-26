"""Scalars are real elements, while empty tensors never fabricate or read values."""

import itertools
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.layout import Layout, axis_visible_limits, build_layout
from rainbow_tensor.render_svg import render_panels, render_svg
from rainbow_tensor.shape import coordinates, generate_values, validate_shape
from rainbow_tensor.theme import DARK, LIGHT

SVG = "{http://www.w3.org/2000/svg}"


class NoElements:
    """An empty array whose backend must never be asked to read a value."""

    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, coordinate):
        raise AssertionError(f"empty array was read at {coordinate}")


class UnusedSelection:
    """Detect selection work before an empty layout has short-circuited."""

    def __iter__(self):
        raise AssertionError("empty tensor selection was iterated")

    def __bool__(self):
        raise AssertionError("empty tensor selection was queried")


def unreadable(coordinate):
    """Fail if a renderer invents a coordinate for an empty tensor."""
    raise AssertionError(f"empty tensor callback at {coordinate}")


def test_scalar_layout_uses_one_real_coordinate_and_preserves_selection():
    reads = []

    def value(coordinate):
        reads.append(coordinate)
        return 42

    layout = build_layout((), selected=[()], value_fn=value)
    assert reads == [()]
    assert layout.shape == ()
    assert not layout.empty
    assert len(layout.cells) == 1
    cell = layout.cells[0]
    assert (cell.coord, cell.flat, cell.value, cell.selected) == ((), 0, 42, True)
    assert not cell.ellipsis
    assert all(frame.axis is None for frame in layout.frames)
    assert cell.x >= 0 and cell.x + cell.width <= layout.width
    assert cell.y >= 0 and cell.y + cell.height <= layout.height


def test_layout_existing_positional_constructor_remains_valid():
    layout = Layout([], [], 200, 100)
    assert (layout.width, layout.height) == (200, 100)


@pytest.mark.parametrize("shape", [(0,), (0, 10**12), (10**12, 0), (10**12, 0, 10**12)])
def test_empty_layout_never_expands_axes_reads_values_or_queries_selection(shape):
    layout = build_layout(
        shape, selected=UnusedSelection(), value_fn=unreadable,
        theme=LIGHT.variant(max_cells=None, max_visible_cells=None),
    )
    assert layout.shape == shape
    assert layout.empty
    assert layout.cells == []
    assert layout.frames == []
    assert 0 < layout.width < 1000
    assert 0 < layout.height < 1000
    assert axis_visible_limits(shape, None, 1) == shape


@pytest.mark.parametrize("theme", [LIGHT, DARK])
def test_empty_renderers_show_a_marker_and_logical_shape_without_value_cells(theme):
    shape = (10**12, 0, 10**12)
    selection = UnusedSelection()
    for svg in (
        render_svg(shape, selected=selection, value_fn=unreadable, theme=theme, legend=False),
        render_panels(
            [{"shape": shape, "selected": selection, "value_fn": unreadable,
              "cell_tint": unreadable}], theme=theme,
        ),
    ):
        document = ET.fromstring(svg)
        notes = document.findall(f".//{SVG}g[@role='note']")
        assert len(notes) == 1
        assert str(shape) in notes[0].attrib["aria-label"]
        assert "No elements" in "".join(document.itertext())
        assert "Shape (1000000000000, 0, 1000000000000)" in "".join(document.itertext())
        assert document.findall(f".//{SVG}title") == []


@pytest.mark.parametrize("theme", [LIGHT, DARK])
def test_scalar_shape_and_renderer_keep_the_value_and_logical_shape(theme):
    visual = rt.shape(np.array(17), theme=theme)
    assert visual.shape == ()
    document = ET.fromstring(visual.svg)
    assert "Shape ()" in "".join(document.itertext())
    titles = document.findall(f".//{SVG}title")
    assert len(titles) == 1
    assert "value 17" in titles[0].text
    assert "flat 0" in titles[0].text
    assert rt.shape(()).shape == ()
    assert ">0</text>" in rt.shape(()).svg


@pytest.mark.parametrize("shape", [(0,), (0, 2), (2, 0, 3), (10**12, 0, 10**12)])
def test_empty_shape_and_index_views_never_read_the_source(shape):
    array = NoElements(shape)
    visual = rt.shape(array)
    assert visual.shape == shape
    assert "No elements" in visual.svg
    assert "hidden" not in visual.text.lower()
    for show_result in (False, True):
        indexed = rt.index(array, (Ellipsis,), show_result=show_result)
        assert indexed.result_shape == shape
        assert indexed.index_mapping.result_count == 0
        assert "No elements" in indexed.svg


@pytest.mark.parametrize("index", [(), (Ellipsis,), (None,), (None, Ellipsis, None)])
def test_scalar_index_matches_numpy_and_exposes_logical_panel_coordinates(index):
    array = np.array(29)

    class InspectResult:
        name = "inspect-result"
        mime_type = "text/plain"

        def render_tensor(self, **kwargs):
            raise AssertionError("expected source and result panels")

        def render_panels(self, panels, **kwargs):
            self.result = panels[-1]
            self.coordinates = list(np.ndindex(self.result["shape"]))
            self.values = [self.result["value_fn"](coord) for coord in self.coordinates]
            return "inspected"

    renderer = InspectResult()
    visual = rt.index(array, index, show_result=True, renderer=renderer)
    expected = array[index]
    assert visual.result_shape == expected.shape
    assert renderer.result["shape"] == expected.shape
    assert renderer.coordinates == list(np.ndindex(expected.shape))
    assert renderer.values == expected.reshape(-1).tolist()
    assert list(visual.selected) == [()]


def test_scalar_coordinates_and_generated_values():
    assert list(coordinates(())) == [()]
    assert generate_values(()) == {(): 0}


def test_empty_coordinates_do_not_pool_other_axes(monkeypatch):
    def reject_product(*args):
        raise AssertionError("empty tensor must not pool Cartesian product inputs")

    monkeypatch.setattr(itertools, "product", reject_product)
    assert list(coordinates((10**12, 0, 10**12))) == []
    assert generate_values((10**12, 0)) == {}


@pytest.mark.parametrize("dimension", [False, True, 0.0, 2.0, np.bool_(False), np.float64(0)])
def test_zero_support_does_not_accept_boolean_or_floating_dimensions(dimension):
    with pytest.raises(TypeError):
        validate_shape((dimension,))


def test_zero_integer_protocol_dimension_is_normalized():
    assert validate_shape((np.int64(0), np.int64(3))) == (0, 3)
