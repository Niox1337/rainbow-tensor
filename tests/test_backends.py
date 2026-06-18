"""Backend duck typing tests."""

import pytest

import rainbow_tensor as rt
from rainbow_tensor.backends import scalar_value, value_at_coordinate


class TupleIndexedArray:
    shape = (2, 2)

    def __getitem__(self, coord):
        return coord[0] * 10 + coord[1]


class ChainedIndexedArray:
    shape = (2, 2)

    def __init__(self):
        self._values = [[10, 11], [20, 21]]

    def __getitem__(self, index):
        if isinstance(index, tuple):
            raise TypeError("tuple indexing is not available")
        return self._values[index]


class NumpyScalar:
    def __init__(self, value):
        self._value = value

    def numpy(self):
        return ItemScalar(self._value)


class ItemScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class ListScalar:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


def test_backend_value_access_uses_tuple_indexing():
    assert value_at_coordinate(TupleIndexedArray(), (1, 0)) == 10


def test_backend_value_access_falls_back_to_chained_indexing():
    assert value_at_coordinate(ChainedIndexedArray(), (1, 1)) == 21


def test_scalar_value_accepts_numpy_and_tolist_paths():
    assert scalar_value(NumpyScalar(7)) == 7
    assert scalar_value(ListScalar(8)) == 8


def test_public_shape_renders_chained_backend_values():
    visual = rt.shape(ChainedIndexedArray())

    assert visual.shape == (2, 2)
    assert ">21<" in visual.svg


def test_public_operations_render_chained_backend_values():
    array = ChainedIndexedArray()

    assert ">10<" in rt.index(array, (1, slice(None))).svg
    assert ">21<" in rt.reshape(array, (4,)).svg
    assert ">20<" in rt.transpose(array).svg


def test_torch_tensor_renders_when_installed():
    torch = pytest.importorskip("torch")
    tensor = torch.arange(6).reshape(2, 3)
    visual = rt.shape(tensor)

    assert visual.shape == (2, 3)
    assert ">5<" in visual.svg


def test_jax_array_renders_when_installed():
    jnp = pytest.importorskip("jax.numpy")
    array = jnp.arange(6).reshape(2, 3)
    visual = rt.shape(array)

    assert visual.shape == (2, 3)
    assert ">5<" in visual.svg
