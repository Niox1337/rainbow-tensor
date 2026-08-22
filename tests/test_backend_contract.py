"""CPU backend contracts compare rendered cell values with independent NumPy results.

Set RAINBOW_TEST_BACKEND to torch, jax, or tensorflow to require that backend.
Without a selection, missing optional backends are skipped. These tests exercise
array reads and visualization semantics, not framework kernels or GPU support.
"""

import importlib
import os

import numpy as np
import pytest

import rainbow_tensor as rt

_BACKENDS = ("torch", "jax", "tensorflow")
_REQUIRED_BACKEND = os.environ.get("RAINBOW_TEST_BACKEND")
if _REQUIRED_BACKEND is not None and _REQUIRED_BACKEND not in _BACKENDS:
    raise pytest.UsageError(
        f"RAINBOW_TEST_BACKEND must be one of: torch, jax, tensorflow. Got {_REQUIRED_BACKEND!r}."
    )


class CpuBackend:
    """Construct real CPU arrays without changing process-wide backend settings."""

    def __init__(self, name, module):
        self.name = name
        self.module = module

    def array(self, values):
        if self.name == "torch":
            return self.module.tensor(values, device="cpu")
        if self.name == "jax":
            return self.module.device_put(values, self.module.devices("cpu")[0])
        with self.module.device("/CPU:0"):
            return self.module.convert_to_tensor(values)

    def transpose(self, array):
        if self.name == "tensorflow":
            with self.module.device("/CPU:0"):
                return self.module.transpose(array)
        return array.T


@pytest.fixture(scope="module", params=(_REQUIRED_BACKEND,) if _REQUIRED_BACKEND else _BACKENDS)
def backend(request):
    name = request.param
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        if _REQUIRED_BACKEND is not None:
            pytest.fail(f"Required backend {name!r} could not be imported: {exc}", pytrace=False)
        if isinstance(exc, ModuleNotFoundError) and exc.name == name:
            pytest.skip(f"Optional backend {name!r} is not installed")
        raise
    return CpuBackend(name, module)


class RecordingRenderer:
    """Evaluate the cell callbacks a renderer receives, including every result cell."""

    name = "backend-contract"
    mime_type = "text/plain"

    def __init__(self):
        self.panels = []
        self.selected = ()

    @staticmethod
    def _values(panel):
        shape = panel["shape"]
        values = [panel["value_fn"](coord) for coord in np.ndindex(shape)]
        return np.asarray(values).reshape(shape)

    def render_tensor(self, **kwargs):
        self.panels = [self._values(kwargs)]
        self.selected = tuple(kwargs.get("selected", ()))
        return "recorded tensor cells"

    def render_panels(self, **kwargs):
        self.panels = [self._values(panel) for panel in kwargs["panels"]]
        return "recorded panel cells"


def _source(backend, dtype, layout):
    source_shape = (3, 2) if layout == "transposed" else (2, 3)
    reference = np.arange(6, dtype=dtype).reshape(source_shape) - 2
    if dtype == "float32":
        reference = reference / 4
    array = backend.array(reference)
    if layout == "transposed":
        reference = reference.T
        array = backend.transpose(array)
        if backend.name == "torch":
            assert not array.is_contiguous()
    return array, reference


@pytest.mark.parametrize("dtype", ["int32", "float32"])
@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
@pytest.mark.parametrize(
    "operation", ["index", "reshape", "transpose", "sum", "mean", "matmul", "einsum"]
)
def test_cpu_backend_operation_matches_numpy_cells(backend, dtype, layout, operation):
    array, reference = _source(backend, dtype, layout)
    renderer = RecordingRenderer()
    if operation == "index":
        selection = (slice(None), slice(1, None))
        expected = reference[selection]
        visual = rt.index(array, selection, renderer=renderer)
        assert renderer.selected == ((0, 1), (0, 2), (1, 1), (1, 2))
        actual = np.asarray([renderer.panels[0][coord] for coord in renderer.selected])
        actual = actual.reshape(expected.shape)
    elif operation == "reshape":
        expected = np.reshape(reference, (3, 2))
        visual = rt.reshape(array, (3, 2), renderer=renderer)
        actual = renderer.panels[-1]
    elif operation == "transpose":
        expected = np.transpose(reference)
        visual = rt.transpose(array, renderer=renderer)
        actual = renderer.panels[-1]
    elif operation == "sum":
        expected = np.sum(reference, axis=0)
        visual = rt.sum(array, 0, renderer=renderer)
        actual = renderer.panels[-1]
    elif operation == "mean":
        expected = np.mean(reference, axis=-1)
        visual = rt.mean(array, -1, renderer=renderer)
        actual = renderer.panels[-1]
    else:
        right_reference = np.arange(6, dtype=dtype).reshape(3, 2) + 1
        right = backend.array(right_reference)
        if operation == "matmul":
            expected = np.matmul(reference, right_reference)
            visual = rt.matmul(array, right, renderer=renderer)
        else:
            expected = np.einsum("ij,jk->ik", reference, right_reference)
            visual = rt.einsum("ij,jk->ik", array, right, renderer=renderer)
        np.testing.assert_array_equal(renderer.panels[1], right_reference)
        actual = renderer.panels[-1]

    assert visual.shape == reference.shape
    assert visual.result_shape == expected.shape
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(renderer.panels[0], reference)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("operation", ["sum", "mean"])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("axis", [None, (), (0,), (0, 2), (-1, 0)])
def test_cpu_backend_multi_axis_reduction_cells(backend, operation, keepdims, axis):
    reference = (np.arange(24, dtype="float32").reshape(2, 3, 4) - 8) / 4
    array = backend.array(reference)
    expected = getattr(np, operation)(reference, axis=axis, keepdims=keepdims)
    focus = tuple(size - 1 for size in expected.shape)
    renderer = RecordingRenderer()
    visual = getattr(rt, operation)(
        array, axis=axis, keepdims=keepdims, focus=focus, renderer=renderer,
    )

    assert visual.shape == reference.shape
    assert visual.result_shape == expected.shape
    assert visual.trace.output_coord == focus
    assert renderer.panels[-1].shape == (expected.shape or (1,))
    np.testing.assert_array_equal(renderer.panels[0], reference)
    actual = renderer.panels[-1].reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("operation", ["sum", "mean"])
def test_cpu_backend_transposed_tuple_reduction_cells(backend, operation):
    array, reference = _source(backend, "float32", "transposed")
    expected = getattr(np, operation)(reference, axis=(1, 0), keepdims=True)
    renderer = RecordingRenderer()
    visual = getattr(rt, operation)(array, axis=(1, 0), keepdims=True, renderer=renderer)

    assert visual.shape == reference.shape
    assert visual.result_shape == expected.shape == (1, 1)
    assert renderer.panels[-1].shape == expected.shape
    np.testing.assert_array_equal(renderer.panels[0], reference)
    np.testing.assert_allclose(renderer.panels[-1], expected, rtol=1e-6, atol=1e-6)
