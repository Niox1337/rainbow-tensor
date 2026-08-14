"""Integer parameters follow the index protocol without lossy coercion."""

import numpy as np
import pytest

import rainbow_tensor as rt


class IndexValue:
    """An integer parameter that implements only Python's index protocol."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class ResultRenderer:
    name = "integer-parameter-results"
    mime_type = "text/plain"

    def render_tensor(self, **kwargs):
        return "tensor"

    def render_panels(self, *, panels, **kwargs):
        panel = panels[-1]
        self.values = {
            coord: panel["value_fn"](coord) for coord in np.ndindex(panel["shape"])
        }
        return "panels"


class UnreadArray:
    shape = (2, 1, 3)

    def __init__(self):
        self.reads = []

    def __getitem__(self, coordinate):
        self.reads.append(coordinate)
        raise AssertionError("Invalid parameters must fail before backend reads")


OPERATIONS = (
    "transpose", "swapaxes", "moveaxis", "squeeze", "expand_dims", "take",
    "repeat", "concatenate", "stack", "sum", "mean", "reshape",
)


def operation_arguments(name, integer):
    """Provide equivalent valid arguments using each integer representation."""
    return {
        "transpose": {"axes": tuple(integer(i) for i in (2, 1, 0))},
        "swapaxes": {"axis1": integer(0), "axis2": integer(2)},
        "moveaxis": {"source": integer(0), "destination": integer(2)},
        "squeeze": {"axis": integer(1)},
        "expand_dims": {"axis": integer(1)},
        "take": {"indices": [integer(i) for i in (2, 0, 2)], "axis": integer(2)},
        "repeat": {"repeats": integer(2), "axis": integer(0)},
        "concatenate": {"axis": integer(2)},
        "stack": {"axis": integer(1)},
        "sum": {"axis": integer(2)},
        "mean": {"axis": integer(2)},
        "reshape": {"new_shape": (integer(3), integer(2))},
    }[name]


@pytest.mark.parametrize("integer", [np.int64, np.uint64, IndexValue])
@pytest.mark.parametrize("name", OPERATIONS)
def test_integer_protocol_matches_numpy_values_and_shapes(name, integer):
    array = np.arange(6).reshape(2, 1, 3)
    operand = [array, array + 100] if name in ("concatenate", "stack") else array
    arguments = operation_arguments(name, int)
    if name == "reshape":
        expected = np.reshape(operand, arguments["new_shape"])
    else:
        expected = getattr(np, name)(operand, **arguments)

    renderer = ResultRenderer()
    visual = getattr(rt, name)(
        operand, **operation_arguments(name, integer), renderer=renderer
    )

    assert visual.result_shape == expected.shape
    assert renderer.values == {coord: expected[coord] for coord in np.ndindex(expected.shape)}
    assert all(type(size) is int for size in visual.result_shape)


INVALID = [True, np.bool_(False), 1.0, np.float64(1.0)]


@pytest.mark.parametrize("invalid", INVALID)
@pytest.mark.parametrize("name", OPERATIONS)
def test_invalid_integer_parameters_fail_before_reading_operands(name, invalid):
    array = UnreadArray()
    operand = [array, array] if name in ("concatenate", "stack") else array
    arguments = operation_arguments(name, int)
    key = {
        "transpose": "axes", "swapaxes": "axis1", "moveaxis": "source",
        "reshape": "new_shape",
    }.get(name, "axis")
    arguments[key] = (0, invalid, 2) if key in ("axes", "new_shape") else invalid

    with pytest.raises(TypeError, match="integer"):
        getattr(rt, name)(operand, **arguments)

    assert array.reads == []


@pytest.mark.parametrize("invalid", INVALID)
@pytest.mark.parametrize("name", ["repeat", "take"])
def test_counts_and_gather_indices_reject_lossy_coercion(name, invalid):
    array = UnreadArray()
    key = "repeats" if name == "repeat" else "indices"
    with pytest.raises(TypeError, match="integer"):
        getattr(rt, name)(array, **{key: [1, invalid]}, axis=0)
    assert array.reads == []


@pytest.mark.parametrize("invalid", INVALID)
def test_scalar_repeat_count_rejects_bool_and_float(invalid):
    array = UnreadArray()
    with pytest.raises(TypeError, match="integer"):
        rt.repeat(array, invalid)
    assert array.reads == []


@pytest.mark.parametrize("invalid", INVALID)
@pytest.mark.parametrize("name", ["moveaxis", "squeeze", "expand_dims"])
def test_axis_iterables_reject_boolean_and_float_entries(name, invalid):
    array = UnreadArray()
    arguments = (
        {"source": (invalid,), "destination": (0,)}
        if name == "moveaxis" else {"axis": (invalid,)}
    )
    with pytest.raises(TypeError, match="integer"):
        getattr(rt, name)(array, **arguments)
    assert array.reads == []


def test_reshape_infers_integer_protocol_unknown_dimension():
    array = np.arange(6).reshape(2, 1, 3)
    renderer = ResultRenderer()
    visual = rt.reshape(array, (IndexValue(-1), np.int64(2)), renderer=renderer)
    expected = array.reshape(-1, 2)

    assert visual.result_shape == expected.shape
    assert renderer.values == {coord: expected[coord] for coord in np.ndindex(expected.shape)}


@pytest.mark.parametrize("name, kwargs, numpy_kwargs", [
    ("moveaxis", {"source": (IndexValue(-1),), "destination": (IndexValue(0),)},
     {"source": (-1,), "destination": (0,)}),
    ("transpose", {"axes": (IndexValue(-1), IndexValue(-2), IndexValue(-3))},
     {"axes": (-1, -2, -3)}),
    ("squeeze", {"axis": iter([IndexValue(-2)])}, {"axis": (-2,)}),
    ("expand_dims", {"axis": iter([IndexValue(0), IndexValue(-1)])}, {"axis": (0, -1)}),
    ("repeat", {"repeats": iter([IndexValue(2), IndexValue(1)]), "axis": 0},
     {"repeats": [2, 1], "axis": 0}),
    ("take", {"indices": iter([IndexValue(-1), IndexValue(0)]), "axis": 0},
     {"indices": [-1, 0], "axis": 0}),
])
def test_integer_iterables_are_resolved_once(name, kwargs, numpy_kwargs):
    array = np.arange(6).reshape(2, 1, 3)
    expected = getattr(np, name)(array, **numpy_kwargs)
    renderer = ResultRenderer()

    visual = getattr(rt, name)(array, **kwargs, renderer=renderer)

    assert visual.result_shape == expected.shape
    assert renderer.values == {coord: expected[coord] for coord in np.ndindex(expected.shape)}


@pytest.mark.parametrize("name, kwargs", [
    ("moveaxis", {"source": (0, -3), "destination": (0, 1)}),
    ("expand_dims", {"axis": (0, -5)}),
    ("squeeze", {"axis": (1, -2)}),
    ("transpose", {"axes": (0, 0, 2)}),
    ("take", {"indices": [0], "axis": 3}),
    ("repeat", {"repeats": -1}),
])
def test_existing_axis_and_count_constraints_remain(name, kwargs):
    array = UnreadArray()
    with pytest.raises(ValueError):
        getattr(rt, name)(array, **kwargs)
    assert array.reads == []
