"""Reshape, transpose, and axis reductions.

Every result shape and coordinate mapping is cross-checked against NumPy, so
the figures the package draws describe the same operation NumPy performs.
"""

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.ops import (
    einsum_contracted_labels,
    einsum_result_shape,
    expand_dims_axes,
    expand_dims_result_shape,
    expand_dims_source_coord,
    matmul_result_shape,
    matmul_source_terms,
    parse_einsum_subscripts,
    reduce_result_shape,
    reduce_source_coords,
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

    from rainbow_tensor.notebook import _LIGHT_TINTS

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


EINSUM_CASES = [
    ("ij,jk->ik", [(2, 3), (3, 4)]),
    ("ij,jk", [(2, 3), (3, 4)]),
    ("ij,jk->ijk", [(2, 3), (3, 4)]),
    ("abc,cde,ef->abdf", [(2, 3, 4), (4, 5, 6), (6, 7)]),
    ("ij->", [(2, 3)]),
    ("ii->i", [(3, 3)]),
    # Ellipsis notation, cross-checked against NumPy below.
    ("...ij,...jk->...ik", [(2, 2, 3), (2, 3, 4)]),
    ("...ij,...jk", [(2, 2, 3), (2, 3, 4)]),
    ("...ii->...i", [(2, 3, 3)]),
    ("...i->...", [(2, 4)]),
    ("...,...->...", [(2, 3), (2, 3)]),
    ("...ij,jk->...ik", [(4, 2, 3), (3, 5)]),
    ("a...b,b...c->a...c", [(2, 3, 4), (4, 3, 5)]),
]


@pytest.mark.parametrize("subscripts, shapes", EINSUM_CASES)
def test_einsum_result_shape_matches_numpy(subscripts, shapes):
    arrays = [np.ones(shape) for shape in shapes]
    expected = np.einsum(subscripts, *arrays)

    assert einsum_result_shape(subscripts, shapes) == expected.shape


def test_parse_einsum_subscripts_returns_inputs_and_output():
    inputs, output = parse_einsum_subscripts("ij,jk->ik", 2)

    assert inputs == (("i", "j"), ("j", "k"))
    assert output == ("i", "k")
    assert einsum_contracted_labels(inputs, output) == ("j",)


def test_einsum_rejects_invalid_subscripts():
    with pytest.raises(ValueError):
        parse_einsum_subscripts("ij,jk->ik->i", 2)
    with pytest.raises(ValueError):
        # An ellipsis needs the operand shapes to know how many axes it covers.
        parse_einsum_subscripts("i...j->ij", 1)
    with pytest.raises(ValueError):
        parse_einsum_subscripts("ij,jk->iz", 2)
    with pytest.raises(ValueError):
        einsum_result_shape("ij,jk->ik", [(2, 3), (2, 4)])


def test_einsum_rejects_invalid_ellipsis():
    # Two ellipses in one operand.
    with pytest.raises(ValueError):
        parse_einsum_subscripts("...i...->...i", 1, [(2, 3, 4)])
    # A stray dot that is not part of a full ellipsis.
    with pytest.raises(ValueError):
        parse_einsum_subscripts("..i->i", 1, [(2, 3)])
    # More named labels than the operand rank leaves for the ellipsis.
    with pytest.raises(ValueError):
        parse_einsum_subscripts("...ijk->...", 1, [(2,)])
    # Ellipsis dimensions that do not match in size cannot share a label.
    with pytest.raises(ValueError):
        einsum_result_shape("...i,...i->...i", [(2, 3), (4, 3)])


def test_public_functions_render_and_report_shape():
    x = np.arange(12).reshape(3, 4)
    assert rt.reshape(x, (2, 6)).result_shape == (2, 6)
    assert rt.transpose(x).result_shape == (4, 3)
    assert rt.swapaxes(np.arange(24).reshape(2, 3, 4), 0, 2).result_shape == (4, 3, 2)
    assert rt.sum(x, 0).result_shape == (4,)
    assert rt.mean(x, 1).result_shape == (3,)
    assert rt.einsum("ij,jk->ik", x, np.ones((4, 2))).result_shape == (3, 2)
    assert rt.squeeze(np.ones((1, 3, 1))).result_shape == (3,)
    assert rt.squeeze(np.ones((1, 3, 1)), 0).result_shape == (3, 1)
    assert rt.expand_dims(x, 1).result_shape == (3, 1, 4)
    assert rt.expand_dims(x, -1).result_shape == (3, 4, 1)
    assert rt.matmul(x, np.ones((4, 2))).result_shape == (3, 2)
    assert rt.matmul(np.arange(4), np.arange(4)).result_shape == ()
    assert rt.matmul(np.ones((5, 2, 3)), np.ones((3, 4))).result_shape == (5, 2, 4)
    for visual in (
        rt.reshape(x, (2, 6)),
        rt.transpose(x),
        rt.swapaxes(np.arange(24).reshape(2, 3, 4), 0, 2),
        rt.sum(x, 0),
        rt.mean(x, 1),
        rt.einsum("ij,jk->ik", x, np.ones((4, 2))),
        rt.squeeze(np.ones((1, 3, 1))),
        rt.expand_dims(x, (0, 2)),
        rt.matmul(x, np.ones((4, 2))),
        rt.matmul(np.arange(4), np.arange(4)),
    ):
        assert visual.svg.startswith("<svg")


def test_reduce_to_scalar_renders():
    x = np.arange(4)
    visual = rt.sum(x, 0)
    assert visual.result_shape == ()
    assert visual.svg.startswith("<svg")


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
