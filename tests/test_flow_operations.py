"""Tracked recipes retain exact origins and agree with small NumPy operations."""

from dataclasses import replace
from itertools import islice

import numpy as np
import pytest

from rainbow_tensor.provenance.flow import Flow


def assert_values(node, expected):
    """Compare every small output, including a scalar or empty result."""
    expected = np.asarray(expected)
    assert node.shape == expected.shape
    for coordinate in np.ndindex(expected.shape):
        assert node.value(coordinate) == pytest.approx(expected[coordinate], nan_ok=True)


MAPPINGS = [
    ((2, 3, 4), "reshape", ((4, -1),)),
    ((2, 3), "reshape", ((6,),)),
    ((), "reshape", ((1, 1),)),
    ((0, 3), "reshape", ((3, 0),)),
    ((2, 3, 4), "transpose", ()),
    ((2, 3, 4), "transpose", ((1, 2, 0),)),
    ((2, 3, 4), "swapaxes", (0, -1)),
    ((2, 3, 4), "moveaxis", ((0, 2), (2, 0))),
    ((1, 2, 1), "squeeze", ()),
    ((1, 2, 1), "squeeze", ((-1,),)),
    ((), "squeeze", (0,)),
    ((2, 3), "expand_dims", ((0, -1),)),
    ((), "expand_dims", (0,)),
    ((2, 3), "repeat", ([0, 2, 1], 1)),
    ((2, 3), "repeat", (2, 0)),
    ((), "repeat", (3, 0)),
    ((2, 3), "take", ([2, -1, 0], 1)),
    ((2, 3), "take", (-1, 1)),
    ((), "take", ([0, 0], 0)),
    ((2, 3), "index", (([1, 0, 1], slice(None, None, -1)),)),
    ((2, 3), "index", ((1, 2),)),
    ((2, 3), "index", ((slice(0, 0),),)),
]


@pytest.mark.parametrize("shape, operation, args", MAPPINGS)
def test_mapping_recipes_match_values_and_original_coordinates(shape, operation, args):
    flow = Flow()
    array = np.arange(int(np.prod(shape))).reshape(shape)
    source = flow.input(array, name="X")
    result = getattr(flow, operation)(source, *args)
    expected = array[args[0]] if operation == "index" else getattr(np, operation)(array, *args)
    assert_values(result, expected)
    for coordinate in np.ndindex(result.shape):
        terms = tuple(result.term_factory(coordinate))
        assert result.term_count == 1
        assert len(terms) == len(terms[0]) == 1
        ref = terms[0][0]
        assert (ref.node_id, ref.output_port) == (source.node_id, 0)
        assert array[ref.coordinate] == expected[coordinate]


@pytest.mark.parametrize("operation", ["sum", "mean"])
@pytest.mark.parametrize("shape, axis, keepdims", [
    ((2, 3, 4), None, False),
    ((2, 3, 4), (0, 2), True),
    ((2, 3), (), False),
    ((2, 3), -1, False),
    ((), None, False),
    ((2, 0, 3), 1, True),
    ((0, 2), 1, False),
])
def test_reduction_recipes_match_numpy_and_keep_group_boundaries(operation, shape, axis, keepdims):
    flow = Flow()
    array = np.arange(int(np.prod(shape))).reshape(shape)
    source = flow.input(array)
    result = getattr(flow, operation)(source, axis, keepdims=keepdims)
    if operation == "mean" and 0 in shape and result.term_count == 0:
        with pytest.warns(RuntimeWarning):
            expected = np.mean(array, axis, keepdims=keepdims)
    else:
        expected = getattr(np, operation)(array, axis, keepdims=keepdims)
    assert_values(result, expected)
    for coordinate in np.ndindex(result.shape):
        terms = tuple(result.term_factory(coordinate))
        assert len(terms) == result.term_count
        assert all(len(term) == 1 and term[0].node_id == source.node_id for term in terms)
    assert result.divisor == (result.term_count if operation == "mean" else 1)


@pytest.mark.parametrize("a_shape, b_shape", [
    ((3,), (3,)), ((2, 3), (3, 4)), ((3,), (2, 3, 4)),
    ((2, 3, 4), (4,)), ((2, 1, 3, 4), (5, 4, 2)), ((2, 0), (0, 3)),
])
def test_matmul_recipes_preserve_each_product(a_shape, b_shape):
    flow = Flow()
    a = np.arange(int(np.prod(a_shape))).reshape(a_shape)
    b = np.arange(int(np.prod(b_shape))).reshape(b_shape)
    left, right = flow.input(a), flow.input(b)
    result = flow.matmul(left, right)
    assert_values(result, np.matmul(a, b))
    for coordinate in np.ndindex(result.shape):
        terms = tuple(result.term_factory(coordinate))
        assert len(terms) == a_shape[-1]
        for left_ref, right_ref in terms:
            assert left_ref.node_id == left.node_id
            assert right_ref.node_id == right.node_id


@pytest.mark.parametrize("subscripts, shapes", [
    ("ij,jk->ik", ((2, 3), (3, 4))),
    ("ii->i", ((3, 3),)),
    ("ii->", ((3, 3),)),
    ("...ij,...jk->...ik", ((2, 1, 3), (1, 3, 4))),
    ("i,i->", ((1,), (4,))),
    (",->", ((), ())),
    ("ij,jk->ik", ((2, 0), (0, 3))),
])
def test_einsum_recipes_keep_operand_order_and_diagonals(subscripts, shapes):
    flow = Flow()
    arrays = tuple(np.arange(int(np.prod(shape))).reshape(shape) for shape in shapes)
    inputs = tuple(flow.input(array) for array in arrays)
    result = flow.einsum(subscripts, *inputs)
    assert_values(result, np.einsum(subscripts, *arrays))
    for coordinate in np.ndindex(result.shape):
        terms = tuple(result.term_factory(coordinate))
        assert len(terms) == result.term_count
        assert all(tuple(ref.node_id for ref in term) == tuple(x.node_id for x in inputs)
                   for term in terms)


@pytest.mark.parametrize("operation, axis", [("concatenate", 0), ("concatenate", -1),
                                               ("stack", 0), ("stack", -1)])
def test_combining_keeps_input_identity(operation, axis):
    flow = Flow()
    arrays = (np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3) + 20)
    inputs = tuple(flow.input(array) for array in arrays)
    result = getattr(flow, operation)(inputs, axis)
    expected = getattr(np, operation)(arrays, axis)
    assert_values(result, expected)
    by_id = {node.node_id: array for node, array in zip(inputs, arrays)}
    for coordinate in np.ndindex(result.shape):
        ref = next(result.term_factory(coordinate))[0]
        assert by_id[ref.node_id][ref.coordinate] == expected[coordinate]


def test_broadcast_keeps_distinct_outputs_of_one_operation():
    flow = Flow()
    arrays = (np.array([[3], [7]]), np.array([1, 2, 4]))
    inputs = tuple(flow.input(array) for array in arrays)
    outputs = flow.broadcast(*inputs, name="B")
    assert outputs[0].node_id == outputs[1].node_id
    assert [node.output_port for node in outputs] == [0, 1]
    assert [node.name for node in outputs] == ["B[0]", "B[1]"]
    for source, output, expected in zip(inputs, outputs, np.broadcast_arrays(*arrays)):
        assert_values(output, expected)
        assert next(output.term_factory((1, 2)))[0].node_id == source.node_id
    assert_values(flow.stack(outputs), np.stack(np.broadcast_arrays(*arrays)))


def test_repeated_slice_transpose_reduction_chain_retains_duplicate_contributions():
    flow = Flow()
    source = flow.input(np.arange(1, 7).reshape(2, 3), name="X")
    selected = flow.index(source, ([1, 0, 1], slice(None, None, -1)), name="S")
    transposed = flow.transpose(selected, name="T")
    output = flow.sum(transposed, axis=1, name="Y")
    assert output.value((0,)) == 15
    contributors = []
    for term in output.term_factory((0,)):
        transpose_term = next(transposed.term_factory(term[0].coordinate))
        index_term = next(selected.term_factory(transpose_term[0].coordinate))
        contributors.append(index_term[0].coordinate)
    assert contributors == [(1, 2), (0, 2), (1, 2)]


def test_factories_capture_mutable_parameters_and_consume_generators_once():
    flow = Flow()
    array = np.arange(6).reshape(2, 3)
    source = flow.input(array)
    indices = np.array([1, 0, 1])
    selected = flow.index(source, (indices, slice(None)))
    axes = [1, 0]
    transposed = flow.transpose(source, axes)
    counts = [0, 2]
    repeated = flow.repeat(source, iter(counts), axis=0)
    reshaped = flow.reshape(source, iter([3, 2]))
    indices[:] = 0
    axes[:] = [0, 1]
    counts[:] = [1, 1]
    assert_values(selected, array[[1, 0, 1]])
    assert_values(transposed, array.T)
    assert_values(repeated, np.repeat(array, [0, 2], axis=0))
    assert_values(reshaped, array.reshape(3, 2))


def test_factories_and_huge_recipes_never_read_values():
    class Unreadable:
        shape = (3, 2)

        def __getitem__(self, coordinate):
            raise AssertionError("recipe construction must not read array values")

    flow = Flow()
    source = flow.input(Unreadable())
    huge = flow.repeat(source, 10**30)
    output = flow.sum(huge, axis=0)
    assert output.term_count == 3 * 10**30
    assert len(tuple(islice(output.term_factory((1,)), 3))) == 3
    assert next(huge.term_factory((3 * 10**30 - 1, 1)))[0].coordinate == (2, 1)


def test_named_inputs_and_operations_have_unique_graph_identities():
    flow = Flow()
    source = flow.input((2,), name="X")
    other = flow.input((2,), name="input_1")
    automatic = flow.input((2,))
    assert len({source.node_id, other.node_id, automatic.node_id}) == 3
    with pytest.raises(ValueError, match="already used"):
        flow.reshape(source, (2,), name="X")
    with pytest.raises(ValueError, match="nonempty"):
        flow.input((2,), name=" ")
    assert_values(flow.reshape(source, (2,), name="Y"), [0, 1])


def test_operations_reject_foreign_untracked_and_forged_inputs():
    flow = Flow()
    source = flow.input((2,))
    with pytest.raises(TypeError, match="tracked tensors"):
        flow.reshape(np.arange(2), (2,))
    with pytest.raises(ValueError, match="same flow"):
        flow.stack((source, Flow().input((2,))))
    with pytest.raises(ValueError, match="not registered"):
        flow.sum(replace(source, node_id="missing"))
    with pytest.raises(ValueError, match="at least one"):
        flow.concatenate(())


def test_scalar_sum_and_mean_keep_their_distinct_axis_rules():
    flow = Flow()
    source = flow.input(np.array(5))
    assert flow.sum(source, axis=0).value(()) == 5
    assert flow.sum(source, axis=-1).value(()) == 5
    with pytest.raises(ValueError, match="out of range"):
        flow.mean(source, axis=0)
    assert flow.mean(source, axis=()).value(()) == 5


def test_python_integer_arithmetic_does_not_overflow_like_native_int8():
    flow = Flow()
    source = flow.input(np.array([120, 120], dtype=np.int8))
    assert flow.sum(source).value(()) == 240
    assert flow.matmul(source, source).value(()) == 28800
    assert flow.mean(source).value(()) == 120


def test_single_term_mean_still_divides_integer_values():
    flow = Flow()
    scalar = flow.input(np.array(5))
    vector = flow.input(np.array([5, 7]))
    assert type(flow.mean(scalar).value(())) is float
    assert type(flow.mean(vector, axis=()).value((0,))) is float
