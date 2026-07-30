"""Source-to-result indexing views retain output order, repeats, and empty shapes."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import rainbow_tensor as rt


class ComparisonRenderer:
    """Read every cell in small result panels independently of the SVG renderer."""

    name = "index-comparison-test"
    mime_type = "text/plain"

    def render_tensor(self, **kwargs):
        raise AssertionError("comparison mode must show two panels")

    def render_panels(self, **kwargs):
        self.panels = kwargs["panels"]
        self.connectors = kwargs["connectors"]
        result = self.panels[-1]
        self.values = [result["value_fn"](coord) for coord in np.ndindex(result["shape"])]
        return "comparison"


@pytest.mark.parametrize(
    "index",
    [
        (slice(None), slice(1, 4, 2)),
        (slice(None, None, -1), slice(None, None, -2)),
        ([2, 0, 2], [3, 1, 3]),
        ([[2], [0]], [3, 1]),
        (np.array([True, False, True]),),
        (None, [2, 0], slice(None)),
        (2, 1),
        (slice(0, 0), slice(None)),
        (np.empty((0, 4), dtype=bool), Ellipsis),
    ],
)
def test_comparison_values_and_shape_match_numpy(index):
    array = np.arange(12).reshape(3, 4)
    expected = array[index]
    renderer = ComparisonRenderer()
    visual = rt.index(array, index, show_result=True, renderer=renderer)

    assert visual.result_shape == expected.shape
    np.testing.assert_array_equal(renderer.values, expected.ravel())
    assert not renderer.panels[0].get("selected")
    assert renderer.connectors == ["->"]
    assert visual.index_mapping.result_count == expected.size
    for coord in np.ndindex(expected.shape):
        assert array[visual.index_mapping.source_coord(coord)] == expected[coord]


def test_duplicate_results_do_not_duplicate_source_highlights():
    array = np.array([10, 20, 30])
    normal = rt.index(array, ([2, 0, 2],))
    compared = rt.index(array, ([2, 0, 2],), show_result=True)
    assert set(compared.selected) == set(normal.selected) == {(0,), (2,)}
    assert [compared.index_mapping.source_coord((i,)) for i in range(3)] == [
        (2,), (0,), (2,),
    ]


def test_shape_tuple_result_values_use_original_flat_positions():
    renderer = ComparisonRenderer()
    rt.index((3, 4), ([2, 0], [3, 1]), show_result=True, renderer=renderer)
    assert renderer.values == [11, 1]


def test_empty_result_svg_has_no_fabricated_value_cells():
    visual = rt.index((3, 4), (slice(0, 0),), show_result=True)
    root = ET.fromstring(visual.svg)
    titles = root.findall(".//{http://www.w3.org/2000/svg}title")
    assert len(titles) == 12  # Only the source has values.
    assert "No elements" in visual.svg
    assert "The result is empty" in visual.text


def test_large_advanced_result_reads_only_visible_source_and_result_cells():
    class Source:
        shape = (1_000_000, 1_000_000)

        def __init__(self):
            self.reads = 0

        def __getitem__(self, coordinate):
            self.reads += 1
            return coordinate[0] * self.shape[1] + coordinate[1]

    array = Source()
    visual = rt.index(array, ([999_999], slice(None)), show_result=True)
    assert visual.index_mapping.result_count == 1_000_000
    assert visual.index_mapping.source_coord((0, 999_999)) == (999_999, 999_999)
    assert array.reads <= 4 * rt.get_default_theme().max_visible_cells


def test_show_result_requires_an_explicit_boolean():
    with pytest.raises(TypeError, match="show_result"):
        rt.index((2, 3), (0,), show_result="yes")
