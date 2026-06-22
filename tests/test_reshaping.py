"""Reshaping and moving axes.

Every result shape and coordinate mapping is cross-checked against NumPy, so
the figures the package draws describe the same operation NumPy performs.
"""

import re

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.ops import (
    expand_dims_axes,
    expand_dims_result_shape,
    expand_dims_source_coord,
    moveaxis_axes,
    reshape_result_shape,
    reshape_source_coord,
    squeeze_axes,
    squeeze_result_shape,
    squeeze_source_coord,
    swapaxes_axes,
    transpose_axes,
    transpose_result_shape,
    transpose_source_coord,
)
from rainbow_tensor.theme import LIGHT

RESHAPE_CASES = [
    ((6,), (2, 3)),
    ((2, 3), (3, 2)),
    ((2, 3), (6,)),
    ((2, 3, 4), (4, 6)),
    ((2, 3, 4), (-1, 4)),
    ((12,), (2, -1, 3)),
]


@pytest.mark.parametrize("old, new", RESHAPE_CASES)
def test_reshape_matches_numpy(old, new):
    x = np.arange(int(np.prod(old))).reshape(old)
    expected = x.reshape(new)
    resolved = reshape_result_shape(old, new)

    assert resolved == expected.shape
    for rc in np.ndindex(resolved):
        assert expected[rc] == x[reshape_source_coord(rc, old, resolved)]


def test_reshape_rejects_bad_size():
    with pytest.raises(ValueError):
        reshape_result_shape((2, 3), (4, 2))
    with pytest.raises(ValueError):
        reshape_result_shape((2, 3), (-1, -1))


TRANSPOSE_CASES = [
    ((2, 3), None),
    ((2, 3), (1, 0)),
    ((2, 3, 4), None),
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4, 5), (0, 2, 1, 3)),
]


@pytest.mark.parametrize("shape, axes", TRANSPOSE_CASES)
def test_transpose_matches_numpy(shape, axes):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    expected = x.transpose() if axes is None else x.transpose(axes)
    perm = transpose_axes(len(shape), axes)

    assert transpose_result_shape(shape, perm) == expected.shape
    for rc in np.ndindex(expected.shape):
        assert expected[rc] == x[transpose_source_coord(rc, perm)]


def test_transpose_rejects_non_permutation():
    with pytest.raises(ValueError):
        transpose_axes(3, (0, 1))
    with pytest.raises(ValueError):
        transpose_axes(2, (0, 0))


SWAPAXES_CASES = [
    ((2, 3), 0, 1),
    ((2, 3, 4), 0, 2),
    ((2, 3, 4), -1, 0),
    ((2, 3, 4, 5), 1, 3),
]


@pytest.mark.parametrize("shape, axis1, axis2", SWAPAXES_CASES)
def test_swapaxes_matches_numpy(shape, axis1, axis2):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    expected = np.swapaxes(x, axis1, axis2)
    perm = swapaxes_axes(len(shape), axis1, axis2)

    assert transpose_result_shape(shape, perm) == expected.shape
    for rc in np.ndindex(expected.shape):
        assert expected[rc] == x[transpose_source_coord(rc, perm)]


def test_swapaxes_rejects_bad_axis():
    with pytest.raises(ValueError):
        swapaxes_axes(3, 0, 3)


SQUEEZE_CASES = [
    ((1, 3), None),
    ((3, 1), None),
    ((1, 3, 1), None),
    ((1, 3, 1), 0),
    ((1, 3, 1), 2),
    ((1, 3, 1), -1),
    ((1, 3, 1), (0, 2)),
    ((1, 3, 1), (0, -1)),
    ((2, 3), None),
    ((1, 1, 1), None),
]


@pytest.mark.parametrize("shape, axis", SQUEEZE_CASES)
def test_squeeze_matches_numpy(shape, axis):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    expected = np.squeeze(x, axis)
    result = squeeze_result_shape(shape, axis)

    assert result == expected.shape
    removed = squeeze_axes(shape, axis)
    for rc in np.ndindex(result):
        assert expected[rc] == x[squeeze_source_coord(rc, removed)]


def test_squeeze_rejects_non_unit_axis():
    with pytest.raises(ValueError):
        squeeze_result_shape((2, 3), 0)
    with pytest.raises(ValueError):
        squeeze_result_shape((1, 3), 5)


EXPAND_CASES = [
    ((3,), 0),
    ((3,), 1),
    ((3,), -1),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), 2),
    ((2, 3), -1),
    ((2, 3), (0, 1)),
    ((2, 3), (0, 3)),
    ((2, 3), (-1, -2)),
]


@pytest.mark.parametrize("shape, axis", EXPAND_CASES)
def test_expand_dims_matches_numpy(shape, axis):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    expected = np.expand_dims(x, axis)
    result = expand_dims_result_shape(shape, axis)

    assert result == expected.shape
    inserted = expand_dims_axes(len(shape), axis)
    for rc in np.ndindex(result):
        assert expected[rc] == x[expand_dims_source_coord(rc, inserted)]


def test_expand_dims_rejects_out_of_range():
    with pytest.raises(ValueError):
        expand_dims_result_shape((2, 3), 4)
    with pytest.raises(ValueError):
        expand_dims_result_shape((2, 3), (0, 0))


MOVEAXIS_CASES = [
    ((2, 3, 4), 0, 2),
    ((2, 3, 4), 2, 0),
    ((2, 3, 4), 0, -1),
    ((2, 3, 4), -1, 0),
    ((2, 3, 4, 5), [0, 1], [2, 3]),
    ((2, 3, 4, 5), [0, 1, 2], [2, 3, 1]),
    ((2, 3, 4, 5), [0, -1], [-1, 0]),
    ((2, 3), 0, 1),
]


@pytest.mark.parametrize("shape, source, destination", MOVEAXIS_CASES)
def test_moveaxis_matches_numpy(shape, source, destination):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    expected = np.moveaxis(x, source, destination)
    order = moveaxis_axes(len(shape), source, destination)
    result = transpose_result_shape(shape, order)

    assert result == expected.shape
    for rc in np.ndindex(result):
        assert expected[rc] == x[transpose_source_coord(rc, order)]


def test_moveaxis_rejects_bad_input():
    with pytest.raises(ValueError):
        moveaxis_axes(3, [0, 1], [2])
    with pytest.raises(ValueError):
        moveaxis_axes(3, 5, 0)


def test_swapaxes_renders_result_with_moved_axis_colours():
    x = np.arange(24).reshape(2, 3, 4)
    visual = rt.swapaxes(x, 0, 2)

    assert visual.result_shape == np.swapaxes(x, 0, 2).shape
    assert "swapaxes" in visual.svg
    assert "Swapping axes 0 and 2." in visual.text
    # axis 0 (red) and axis 2 keep their ramp colours through the swap
    assert LIGHT.axis_color(0) in visual.svg
    assert LIGHT.axis_color(2) in visual.svg


def test_squeeze_source_frames_mark_removed_axes():
    # The removed axis frames must use the accent colour, matching the green
    # caption numbers, instead of falling back to the default axis ramp.
    from rainbow_tensor.theme import resolve_theme

    theme = resolve_theme(None)
    svg = rt.squeeze(np.ones((1, 3, 1))).svg
    source = svg.split("-&gt;")[0]  # frames before the connector belong to the source
    strokes = re.findall(r'stroke="(#[0-9a-f]+)" stroke-width="2.5"', source)
    assert theme.surface_selected in strokes  # removed axis 0 frame is the accent
    assert theme.axis_color(0) not in strokes  # and not the default axis 0 colour


def test_public_reshaping_render_and_report_shape():
    x = np.arange(12).reshape(3, 4)
    assert rt.reshape(x, (2, 6)).result_shape == (2, 6)
    assert rt.transpose(x).result_shape == (4, 3)
    assert rt.swapaxes(np.arange(24).reshape(2, 3, 4), 0, 2).result_shape == (4, 3, 2)
    assert rt.squeeze(np.ones((1, 3, 1))).result_shape == (3,)
    assert rt.squeeze(np.ones((1, 3, 1)), 0).result_shape == (3, 1)
    assert rt.expand_dims(x, 1).result_shape == (3, 1, 4)
    assert rt.expand_dims(x, -1).result_shape == (3, 4, 1)
    assert rt.moveaxis(np.zeros((2, 3, 4)), 0, 2).result_shape == (3, 4, 2)
    assert rt.moveaxis(np.zeros((2, 3, 4, 5)), [0, 1], [2, 3]).result_shape == (4, 5, 2, 3)
    for visual in (
        rt.reshape(x, (2, 6)),
        rt.transpose(x),
        rt.swapaxes(np.arange(24).reshape(2, 3, 4), 0, 2),
        rt.squeeze(np.ones((1, 3, 1))),
        rt.expand_dims(x, (0, 2)),
        rt.moveaxis(np.zeros((2, 3, 4)), 0, 2),
        rt.moveaxis(np.zeros((2, 3, 4, 5)), [0, 1], [2, 3]),
    ):
        assert visual.svg.startswith("<svg")
