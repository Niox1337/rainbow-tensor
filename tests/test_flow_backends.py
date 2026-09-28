"""Recorded flows read CPU backend scalars and preserve coordinate ancestry."""

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
    """Construct CPU tensors with the same protocol for each optional framework."""

    def __init__(self, name, module):
        self.name = name
        self.module = module

    def array(self, values):
        """Place a small reference array on a CPU device without changing defaults."""
        if self.name == "torch":
            return self.module.tensor(values, device="cpu")
        if self.name == "jax":
            return self.module.device_put(values, self.module.devices("cpu")[0])
        with self.module.device("/CPU:0"):
            return self.module.convert_to_tensor(values)

    def transpose(self, array):
        """Use the framework transpose while retaining its real scalar-read protocol."""
        if self.name == "tensorflow":
            with self.module.device("/CPU:0"):
                return self.module.transpose(array)
        return array.T


@pytest.fixture(scope="module", params=(_REQUIRED_BACKEND,) if _REQUIRED_BACKEND else _BACKENDS)
def backend(request):
    """Skip missing optional frameworks unless the environment explicitly requires one."""
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
    """Read every coordinate of these small flow panels for independent comparison."""

    name = "flow-backend-contract"
    mime_type = "text/plain"

    @staticmethod
    def _values(panel):
        shape = panel["shape"]
        values = [panel["value_fn"](coordinate) for coordinate in np.ndindex(shape)]
        return np.asarray(values).reshape(shape)

    def render_tensor(self, **kwargs):
        self.panels = [self._values(kwargs)]
        return "recorded tensor cells"

    def render_panels(self, **kwargs):
        self.panels = [self._values(panel) for panel in kwargs["panels"]]
        return "recorded flow cells"


def _check_repeated_transposed_chain(adapter, monkeypatch):
    """Test a real transposed input and count calls at the scalar-read boundary."""
    import rainbow_tensor.visual as visual_module

    reference = np.arange(1, 7, dtype="int32").reshape(3, 2).T
    array = adapter.transpose(adapter.array(reference.T))
    original = visual_module.value_at_coordinate
    reads = []

    def scalar_read(source, coordinate):
        reads.append(tuple(coordinate))
        return original(source, coordinate)

    monkeypatch.setattr(visual_module, "value_at_coordinate", scalar_read)
    flow = rt.Flow()
    source = flow.input(array)
    picked = flow.index(source, ([1, 0, 1], slice(None, None, -1)))
    result = flow.sum(flow.transpose(picked), axis=1)
    trace = result.trace((0,))
    assert reads == []
    assert trace.complete
    assert {item.reference.coordinate: item.count for item in trace.roots} == {
        (1, 2): 2, (0, 2): 1,
    }
    expected = reference[[1, 0, 1], ::-1].T.sum(axis=1)
    assert result.value((0,)) == expected[0]
    assert reads == [(1, 2), (0, 2)]
    reads.clear()
    renderer = RecordingRenderer()
    visual = result.visualize((0,), renderer=renderer)
    np.testing.assert_array_equal(renderer.panels[0], reference)
    np.testing.assert_array_equal(renderer.panels[-1], expected)
    assert len(reads) == len(set(reads)) == reference.size
    assert visual.metadata["numeric_semantics"]["operand_sources"] == ["array_values"]


def _check_grouped_arithmetic(adapter):
    """Compare a multiplication followed by mean without flattening its operation groups."""
    left_reference = np.array([[0.25, 0.5, 0.75], [1.0, 1.25, 1.5]], dtype="float32")
    right_reference = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype="float32")
    flow = rt.Flow()
    left = flow.input(adapter.array(left_reference))
    right = flow.input(adapter.array(right_reference))
    result = flow.mean(flow.matmul(left, right), axis=1)
    expected = np.mean(left_reference @ right_reference, axis=1)
    renderer = RecordingRenderer()
    visual = result.visualize((1,), renderer=renderer)
    np.testing.assert_allclose(renderer.panels[-1], expected, rtol=1e-6, atol=1e-6)
    assert result.value((1,)) == pytest.approx(float(expected[1]))
    assert visual.provenance.steps[0].operation == "mean"
    assert visual.provenance.steps[0].divisor == 2
    assert [visual.provenance.steps[i].operation for i in (1, 2)] == ["matmul", "matmul"]


def _check_scalar_and_empty_sources(adapter):
    """Scalar reads remain zero-dimensional and empty ancestry never invents elements."""
    flow = rt.Flow()
    scalar = flow.input(adapter.array(np.array(7, dtype="int32")))
    repeated = flow.repeat(scalar, 3)
    mean = flow.mean(repeated)
    assert mean.value() == 7.0
    trace = mean.trace()
    assert len(trace.roots) == 1
    assert trace.roots[0].reference.coordinate == ()
    assert trace.roots[0].count == 3

    empty = flow.input(adapter.array(np.empty((0, 3), dtype="float32")))
    result = flow.sum(empty, axis=1)
    renderer = RecordingRenderer()
    visual = result.visualize(renderer=renderer)
    assert visual.result_shape == (0,)
    assert visual.provenance.root is None
    assert renderer.panels[-1].shape == (0,)
    assert renderer.panels[-1].size == 0


def test_cpu_backend_repeated_flow_reads_only_scalars(backend, monkeypatch):
    _check_repeated_transposed_chain(backend, monkeypatch)


def test_cpu_backend_flow_preserves_grouped_arithmetic(backend):
    _check_grouped_arithmetic(backend)


def test_cpu_backend_flow_handles_scalar_and_empty_inputs(backend):
    _check_scalar_and_empty_sources(backend)


def test_numpy_flow_reads_only_scalars(monkeypatch):
    _check_repeated_transposed_chain(np, monkeypatch)


def test_numpy_flow_preserves_grouped_arithmetic():
    _check_grouped_arithmetic(np)


def test_numpy_flow_handles_scalar_and_empty_inputs():
    _check_scalar_and_empty_sources(np)



def _check_binary_broadcast(adapter, operation):
    """Preserve two operand roles and compare broadcast values from real CPU scalars."""
    left_reference = np.array([[2], [4]], dtype="float32")
    right_reference = np.array([1, 2, 8], dtype="float32")
    flow = rt.Flow()
    left = flow.input(adapter.array(left_reference), name="left")
    right = flow.input(adapter.array(right_reference), name="right")
    result = getattr(flow, operation)(left, right)
    expected = getattr(np, operation)(left_reference, right_reference)
    renderer = RecordingRenderer()
    visual = result.visualize((1, 2), renderer=renderer)
    np.testing.assert_allclose(renderer.panels[-1], expected, rtol=1e-6, atol=1e-6)
    assert result.value((1, 2)) == pytest.approx(float(expected[1, 2]))
    assert visual.trace.expression.operator == operation
    assert [(ref.operand, ref.coordinate) for ref in visual.trace.expression.operands] == [
        (0, (1, 0)), (1, (2,)),
    ]
    assert {item.reference: item.count for item in visual.provenance.roots} == {
        left.ref((1, 0)): 1, right.ref((2,)): 1,
    }


def _check_binary_scalar_and_empty(adapter, operation):
    """Use backend scalar inputs with explicit constants and preserve empty ancestry."""
    flow = rt.Flow()
    scalar = flow.input(adapter.array(np.array(7, dtype="float32")))
    constant = flow.input(2)
    result = getattr(flow, operation)(scalar, constant)
    assert result.shape == ()
    assert result.value(()) == pytest.approx(getattr(np, operation)(7, 2))
    assert {item.reference.coordinate for item in result.trace().roots} == {()}

    empty = flow.input(adapter.array(np.empty((0, 3), dtype="float32")))
    result = getattr(flow, operation)(empty, constant)
    renderer = RecordingRenderer()
    visual = result.visualize(renderer=renderer)
    assert visual.result_shape == (0, 3)
    assert visual.trace is None
    assert visual.provenance.root is None
    assert renderer.panels[-1].shape == (0, 3)
    assert renderer.panels[-1].size == 0


def _check_normalized_repeated_flow(adapter, monkeypatch):
    """Keep numerator, denominator and repeated-gather paths while reading unique inputs once."""
    import rainbow_tensor.visual as visual_module

    reference = np.arange(1, 7, dtype="float32").reshape(3, 2).T
    array = adapter.transpose(adapter.array(reference.T))
    original = visual_module.value_at_coordinate
    reads = []

    def scalar_read(source, coordinate):
        reads.append(tuple(coordinate))
        return original(source, coordinate)

    monkeypatch.setattr(visual_module, "value_at_coordinate", scalar_read)
    flow = rt.Flow()
    source = flow.input(array, name="X")
    picked = flow.index(source, ([1, 0, 1], slice(None, None, -1)), name="S")
    row_total = flow.sum(picked, axis=1, keepdims=True, name="total")
    normalized = flow.divide(picked, row_total, name="normalized")
    result = flow.sum(normalized, axis=0, name="Y")
    trace = result.trace((1,))
    assert reads == []
    assert trace.complete
    assert {item.reference.coordinate: item.count for item in trace.roots} == {
        (1, 1): 4, (1, 0): 2, (1, 2): 2,
        (0, 1): 2, (0, 0): 1, (0, 2): 1,
    }
    divisions = [step for step in trace.steps if step.binary_operator == "divide"]
    assert len(divisions) == 3
    assert all(len(step.operands) == 2 and step.terms == () for step in divisions)
    with pytest.raises(rt.ValueBudgetExceeded):
        normalized.value((2, 1), max_terms=1)
    assert reads == []

    selected = reference[[1, 0, 1], ::-1]
    expected = (selected / selected.sum(axis=1, keepdims=True)).sum(axis=0)
    assert result.value((1,)) == pytest.approx(float(expected[1]))
    assert len(reads) == len(set(reads)) == reference.size
    reads.clear()
    renderer = RecordingRenderer()
    result.visualize((1,), renderer=renderer)
    np.testing.assert_allclose(renderer.panels[-1], expected, rtol=1e-6, atol=1e-6)
    assert len(reads) == len(set(reads)) == reference.size


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "divide"])
def test_cpu_backend_flow_binary_broadcast(backend, operation):
    _check_binary_broadcast(backend, operation)


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "divide"])
def test_cpu_backend_flow_binary_scalar_and_empty(backend, operation):
    _check_binary_scalar_and_empty(backend, operation)


def test_cpu_backend_flow_normalization_preserves_repeated_origins(backend, monkeypatch):
    _check_normalized_repeated_flow(backend, monkeypatch)


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "divide"])
def test_numpy_flow_binary_broadcast(operation):
    _check_binary_broadcast(np, operation)


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "divide"])
def test_numpy_flow_binary_scalar_and_empty(operation):
    _check_binary_scalar_and_empty(np, operation)


def test_numpy_flow_normalization_preserves_repeated_origins(monkeypatch):
    _check_normalized_repeated_flow(np, monkeypatch)


def test_cpu_backend_flow_division_rejects_a_zero_denominator(backend):
    flow = rt.Flow()
    numerator = flow.input(backend.array(np.array(1, dtype="float32")))
    denominator = flow.input(backend.array(np.array(0, dtype="float32")))
    result = flow.divide(numerator, denominator)
    assert result.trace().complete
    with pytest.raises(ZeroDivisionError):
        result.value(())
