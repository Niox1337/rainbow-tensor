"""Broadcasting tests, cross checked against NumPy.

The result shape, the source coordinate mapping, and the stretched axes are all
checked against ``numpy.broadcast_shapes`` and ``numpy.broadcast_to`` so the
pure Python maths stays in step with the reference behaviour.
"""

import itertools

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.ops import (
    broadcast_result_shape,
    broadcast_source_coord,
    broadcast_stretched_axes,
)

BROADCAST_CASES = [
    [(3, 1), (1, 4)],
    [(1,), (5,)],
    [(2, 3, 4), (4,)],
    [(2, 1, 4), (3, 4)],
    [(5,), (1,)],
    [(3, 1, 5), (1, 4, 5)],
    [(1, 1), (2, 3)],
    [(4,), (4,)],
]


@pytest.mark.parametrize("shapes", BROADCAST_CASES)
def test_result_shape_matches_numpy(shapes):
    assert broadcast_result_shape(shapes) == np.broadcast_shapes(*shapes)


@pytest.mark.parametrize("shapes", BROADCAST_CASES)
def test_source_coord_and_stretched_match_numpy(shapes):
    result = broadcast_result_shape(shapes)
    for s in shapes:
        arr = np.arange(int(np.prod(s))).reshape(s)
        broadcast = np.broadcast_to(arr, result)
        for coord in itertools.product(*[range(n) for n in result]):
            assert arr[broadcast_source_coord(coord, s)] == broadcast[coord]

        extra = len(result) - len(s)
        expected = list(range(extra)) + [
            extra + i for i, size in enumerate(s) if size == 1 and result[extra + i] != 1
        ]
        assert broadcast_stretched_axes(s, result) == expected


def test_incompatible_shapes_raise():
    with pytest.raises(ValueError):
        broadcast_result_shape([(3,), (4,)])


def test_public_broadcast_renders_and_reports_shape():
    visual = rt.broadcast((3, 1), (1, 4))
    assert visual.result_shape == (3, 4)
    assert visual.svg.startswith("<svg")


def test_public_broadcast_with_arrays():
    a = np.arange(3).reshape(3, 1)
    b = np.arange(4).reshape(1, 4)
    visual = rt.broadcast(a, b)
    assert visual.result_shape == (3, 4)
