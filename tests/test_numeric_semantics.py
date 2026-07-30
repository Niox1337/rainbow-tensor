"""Numerical previews describe Python scalar arithmetic without native-kernel claims."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import rainbow_tensor as rt


class ResultRenderer:
    """Inspect only output callbacks, so skipped evaluations cannot hide source reads."""

    name = "numeric-semantics-test"
    mime_type = "text/plain"

    def __init__(self):
        self.values = []

    def render_tensor(self, **kwargs):
        return "unused"

    def render_panels(self, **kwargs):
        output = kwargs["panels"][-1]
        self.values = [output["value_fn"](coord) for coord in np.ndindex(output["shape"])]
        return str(self.values)


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_float32_cancellation_uses_python_values_and_explains_difference(operation):
    array = np.array([1e8, 1, -1e8], dtype=np.float32)
    renderer = ResultRenderer()

    visual = operation(array, 0, renderer=renderer)

    native = np.sum(array) if operation is rt.sum else np.mean(array)
    assert native == 0
    assert renderer.values == [1.0 if operation is rt.sum else 1.0 / 3]
    assert visual.metadata["numeric_semantics"] == {
        "arithmetic": "python_scalar",
        "operand_sources": ["array_values"],
    }
    assert "Accumulation dtype, rounding, and overflow may differ" in visual.text


@pytest.mark.parametrize("operation", ["sum", "matmul", "einsum"])
def test_int64_overflow_is_python_integer_arithmetic_not_native_wraparound(operation):
    array = np.array([2**63 - 1, 1], dtype=np.int64)
    renderer = ResultRenderer()
    if operation == "sum":
        visual = rt.sum(array, 0, renderer=renderer)
        native = np.sum(array)
    elif operation == "matmul":
        ones = np.ones(2, dtype=np.int64)
        visual = rt.matmul(array, ones, renderer=renderer)
        native = np.matmul(array, ones)
    else:
        visual = rt.einsum("i->", array, renderer=renderer)
        native = np.einsum("i->", array)

    assert native == -(2**63)
    assert renderer.values == [2**63]
    assert isinstance(renderer.values[0], int)
    assert visual.metadata["numeric_semantics"]["arithmetic"] == "python_scalar"
    assert "Numerical model: Python scalar arithmetic." in visual.text


@pytest.mark.parametrize("operation", ["matmul", "einsum"])
@pytest.mark.parametrize("generated_operand", [0, 1])
def test_mixed_operands_identify_exact_placeholder_source(operation, generated_operand):
    operands = [np.array([[1, 2], [3, 4]]), np.ones((2, 1), dtype=np.int64)]
    operands[generated_operand] = operands[generated_operand].shape
    renderer = ResultRenderer()
    if operation == "matmul":
        visual = rt.matmul(*operands, renderer=renderer)
    else:
        visual = rt.einsum("ij,jk->ik", *operands, renderer=renderer)

    assert renderer.values == ([1, 5] if generated_operand == 0 else [2, 4])
    expected_sources = ["array_values", "array_values"]
    expected_sources[generated_operand] = "generated_values"
    assert visual.metadata["numeric_semantics"]["operand_sources"] == expected_sources
    assert f"Operand {generated_operand} uses generated row-major placeholder values" in visual.text
    assert "may differ from the input array backend" in visual.text


@pytest.mark.parametrize("operation", ["sum", "mean", "matmul", "einsum"])
def test_shape_only_operands_are_explicit_placeholders(operation):
    if operation == "sum":
        visual = rt.sum((3,), 0)
    elif operation == "mean":
        visual = rt.mean((3,), 0)
    elif operation == "matmul":
        visual = rt.matmul((3,), (3,))
    else:
        visual = rt.einsum("i->", (3,))

    sources = visual.metadata["numeric_semantics"]["operand_sources"]
    assert sources == ["generated_values"] * (2 if operation == "matmul" else 1)
    assert "Operand 0 uses generated row-major placeholder values from its shape." in visual.text
    assert "input array backend" not in visual.text


@pytest.mark.parametrize("operation", ["sum", "mean", "matmul", "einsum"])
def test_skipped_output_keeps_semantics_without_evaluating_any_source(operation):
    class UnreadableArray:
        shape = (3,)

        def __getitem__(self, coordinate):
            raise AssertionError("Skipped outputs must not read source values")

    array = UnreadableArray()
    renderer = ResultRenderer()
    options = {"max_terms": 1, "renderer": renderer}
    if operation == "sum":
        visual = rt.sum(array, 0, **options)
    elif operation == "mean":
        visual = rt.mean(array, 0, **options)
    elif operation == "matmul":
        visual = rt.matmul(array, (3,), **options)
    else:
        visual = rt.einsum("i->", array, **options)

    assert renderer.values == ["?"]
    assert visual.metadata["value_evaluation"]["status"] == "skipped"
    assert visual.metadata["numeric_semantics"]["arithmetic"] == "python_scalar"
    assert visual.metadata["numeric_semantics"]["operand_sources"][0] == "array_values"
    assert "were not computed" in visual.text


def test_shape_only_numerical_views_import_no_array_backend():
    source_root = str(Path(rt.__file__).resolve().parents[1])
    script = f"""
import sys
sys.path.insert(0, {source_root!r})
import rainbow_tensor as rt
views = [rt.sum((2,), 0), rt.mean((2,), 0), rt.matmul((2,), (2,)), rt.einsum('i->', (2,))]
assert all(view.metadata['numeric_semantics']['arithmetic'] == 'python_scalar' for view in views)
assert not set(('numpy', 'torch', 'jax', 'tensorflow')).intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
