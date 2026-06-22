"""Reductions and math.

Every result shape and coordinate mapping is cross-checked against NumPy, so
the figures the package draws describe the same operation NumPy performs.
"""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.ops import (
    matmul_result_shape,
    matmul_source_terms,
    reduce_result_shape,
    reduce_source_coords,
)

REDUCE_CASES = [
    ((4,), 0),
    ((3, 4), 0),
    ((3, 4), 1),
    ((2, 3, 4), 1),
    ((2, 3, 4), -1),
]


@pytest.mark.parametrize("shape, axis", REDUCE_CASES)
def test_reduce_matches_numpy(shape, axis):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    res = reduce_result_shape(shape, axis)

    assert res == np.sum(x, axis).shape
    for rc in np.ndindex(res):
        vals = [x[sc] for sc in reduce_source_coords(rc, shape, axis)]
        assert sum(vals) == np.sum(x, axis)[rc]
        assert np.isclose(sum(vals) / len(vals), np.mean(x, axis)[rc])


def test_reduce_rejects_bad_axis():
    with pytest.raises(ValueError):
        reduce_result_shape((2, 3), 2)


def test_reduction_colours_groups_with_matching_backgrounds():
    import re
    from collections import Counter

    from rainbow_tensor.visual import _LIGHT_TINTS

    tint_fills = {fill for fill, _ in _LIGHT_TINTS}
    # Reducing axis 0 of (2, 3) leaves result (3,), so three column groups, each
    # with two source values plus the one result element they fold into.
    for op in (rt.sum, rt.mean):
        visual = op((2, 3), 0)
        assert visual.result_shape == (3,)
        cell_fills = Counter(re.findall(r'<rect[^>]*fill="(#[0-9a-fA-F]+)"', visual.svg))
        # The first group is shown through the selected highlight, and the other
        # groups each take their own tint background.
        assert rt.LIGHT.surface_selected in cell_fills
        assert len(set(cell_fills) & tint_fills) >= 2
        # Every group colours its two source cells and its one result cell the
        # same, so a result element shares the background of its source group.
        for fill, count in cell_fills.items():
            if fill in tint_fills or fill == rt.LIGHT.surface_selected:
                assert count == 3


def test_reduce_to_scalar_renders():
    x = np.arange(4)
    visual = rt.sum(x, 0)
    assert visual.result_shape == ()
    assert visual.svg.startswith("<svg")


def test_reduce_result_keeps_surviving_axis_colours():
    # Reducing a non-last axis must not recolour the surviving axes by their new
    # position. Reducing axis 0 of (2, 3, 4) leaves (3, 4); the surviving "3"
    # axis keeps the source axis 1 colour rather than taking the axis 0 colour.
    from rainbow_tensor.theme import resolve_theme

    theme = resolve_theme(None)
    svg = rt.mean(np.arange(24).reshape(2, 3, 4), 0).svg
    assert theme.axis_color(1) in svg  # surviving axis keeps its source colour
    # and the result panel never paints that frame in the reduced axis 0 colour
    assert f'mean (</tspan><tspan fill="{theme.axis_color(0)}">3' not in svg


MATMUL_CASES = [
    ((3,), (3,)),
    ((2, 3), (3, 4)),
    ((3,), (3, 4)),
    ((2, 3), (3,)),
    ((5, 2, 3), (5, 3, 4)),
    ((5, 2, 3), (3, 4)),
    ((2, 3), (5, 3, 4)),
    ((1, 2, 3), (5, 3, 4)),
    ((6, 5, 2, 3), (5, 3, 4)),
    ((5, 2, 3), (3,)),
    ((3,), (5, 3, 4)),
]


@pytest.mark.parametrize("a, b", MATMUL_CASES)
def test_matmul_matches_numpy(a, b):
    x = np.arange(int(np.prod(a))).reshape(a)
    y = np.arange(int(np.prod(b))).reshape(b)
    expected = np.matmul(x, y)
    result = matmul_result_shape(a, b)

    assert result == expected.shape
    for oc in np.ndindex(result):
        total = sum(x[ac] * y[bc] for ac, bc in matmul_source_terms(oc, a, b))
        assert total == expected[oc]
    # The scalar 1-D by 1-D case reduces to an empty output coordinate.
    if result == ():
        total = sum(x[ac] * y[bc] for ac, bc in matmul_source_terms((), a, b))
        assert total == expected[()]


def test_matmul_rejects_inner_mismatch():
    with pytest.raises(ValueError):
        matmul_result_shape((2, 3), (4, 5))
    with pytest.raises(ValueError):
        matmul_result_shape((), (2,))


def test_public_reductions_render_and_report_shape():
    x = np.arange(12).reshape(3, 4)
    assert rt.sum(x, 0).result_shape == (4,)
    assert rt.mean(x, 1).result_shape == (3,)
    assert rt.matmul(x, np.ones((4, 2))).result_shape == (3, 2)
    assert rt.matmul(np.arange(4), np.arange(4)).result_shape == ()
    assert rt.matmul(np.ones((5, 2, 3)), np.ones((3, 4))).result_shape == (5, 2, 4)
    for visual in (
        rt.sum(x, 0),
        rt.mean(x, 1),
        rt.matmul(x, np.ones((4, 2))),
        rt.matmul(np.arange(4), np.arange(4)),
    ):
        assert visual.svg.startswith("<svg")
