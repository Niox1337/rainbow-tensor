"""einsum maths and view.

The result shape is cross-checked against NumPy, and the view keeps every
subscript label coloured consistently across operands and the output.
"""

import numpy as np
import pytest

from rainbow_tensor import einsum
from rainbow_tensor.ops import (
    einsum_contracted_labels,
    einsum_result_shape,
    parse_einsum_subscripts,
)

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


def test_einsum_renders_subscripts_and_result_values():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(12).reshape(3, 4)
    visual = einsum("ij,jk->ik", a, b)
    expected = np.einsum("ij,jk->ik", a, b)

    assert visual.result_shape == expected.shape
    assert ">20<" in visual.svg
    assert "operand 0" in visual.svg
    assert "output" in visual.svg
    assert "Contracted labels" in visual.text
    assert visual.selected == [[(0, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0)]]


def test_einsum_label_colours_match_between_figure_and_caption():
    import re

    from rainbow_tensor.views.einsum import _einsum_label_colors

    # A leaf axis label used to be coloured only in the caption, never in the
    # figure. Every label colour must now appear as a stroke in the figure too.
    for sub, shapes in [
        ("ij,jk->ik", [(2, 3), (3, 4)]),
        ("bij,bjk->bik", [(2, 2, 3), (2, 3, 4)]),
        ("abc,cde,ef->abdf", [(2, 3, 4), (4, 5, 6), (6, 7)]),
    ]:
        inputs, output = parse_einsum_subscripts(sub, len(shapes))
        label_colors = _einsum_label_colors(inputs, output)
        svg = einsum(sub, *shapes).svg
        strokes = set(re.findall(r'stroke="(#[0-9a-fA-F]+)"', svg))
        for label, color in label_colors.items():
            assert color in strokes, f"{sub}: label {label} colour {color} missing from figure"


def test_einsum_roles_are_visually_distinct():
    from rainbow_tensor.views.einsum import (
        _EINSUM_CONTRACTED,
        _EINSUM_FREE,
        _EINSUM_SHARED,
        _einsum_label_colors,
        _einsum_roles,
    )

    families = {
        "free": set(_EINSUM_FREE),
        "shared": set(_EINSUM_SHARED),
        "contracted": set(_EINSUM_CONTRACTED),
    }
    # The three families share no colours, so a label's role is readable.
    assert not (families["free"] & families["shared"])
    assert not (families["free"] & families["contracted"])
    assert not (families["shared"] & families["contracted"])

    inputs, output = parse_einsum_subscripts("bij,bjk->bik", 2)
    roles = _einsum_roles(inputs, output)
    colors = _einsum_label_colors(inputs, output)
    assert roles == {"b": "shared", "i": "free", "j": "contracted", "k": "free"}
    for label, role in roles.items():
        assert colors[label] in families[role]


def test_einsum_ellipsis_renders_and_matches_numpy_shape():
    a = np.arange(12).reshape(2, 2, 3)
    b = np.arange(24).reshape(2, 3, 4)
    visual = einsum("...ij,...jk->...ik", a, b)

    assert visual.result_shape == np.einsum("...ij,...jk->...ik", a, b).shape
    assert visual.svg.startswith("<svg")
    assert "operand 0" in visual.svg


def test_einsum_general_operands_match_numpy_shape():
    a = np.ones((2, 3, 4))
    b = np.ones((4, 5, 6))
    c = np.ones((6, 7))
    visual = einsum("abc,cde,ef->abdf", a, b, c)

    assert visual.result_shape == np.einsum("abc,cde,ef->abdf", a, b, c).shape
    assert visual.svg.startswith("<svg")
