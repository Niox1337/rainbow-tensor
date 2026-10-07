# Check the teaching contract

An explanation should match the package in the learner's Python environment.
Use `rt.capabilities()` to read its version, operation signatures, Flow methods,
and controller support before generating a lesson. This function returns a fresh
JSON-compatible dictionary without creating notebook widgets.

```python
import rainbow_tensor as rt

contract = rt.capabilities()
assert contract["package"]["version"] == rt.__version__
assert contract["operations"]["divide"]["explore"]
print(contract["operations"]["divide"]["signature"])
```

The <a href="../_static/capabilities.json">machine-readable reference</a> describes the
checkout used to build this documentation. Compare its package version with
the installed version. `schema_version` versions the document format itself.
The installed runtime is authoritative when those versions differ.

The operation inventory and explorer dispatcher share the public view registry.
Signatures come from the actual callables. A `flow_signature` of `null` means
the operation is an inspection view rather than a recorded Flow method.
Standalone walkthrough support is listed separately from support for a tracked
tensor containing several operations. Read the operation docstring for its
value and axis restrictions.

## Verify sources as well as numbers

This checked lesson verifies an operation's availability, the focused output,
and duplicate source participation in a recorded chain:

```{literalinclude} ../../examples/lessons/runtime_contract.py
:language: python
```

A visual of an already computed array cannot reconstruct earlier operations.
Record those operations explicitly through `Flow`. Its occurrence counts describe
paths through the recipe. They are not coefficients or derivatives.

## Explain numerical differences when they matter

Preview evaluation uses Python scalars. It does not dispatch NumPy, PyTorch,
JAX, or TensorFlow kernels. With unsigned byte inputs, NumPy can store
`250 + 10` as `4`, while the logical Python calculation produces `260`.
A zero denominator raises `ZeroDivisionError` in a preview, even when NumPy's
floating calculation would produce infinity or NaN.

```{literalinclude} ../../examples/lessons/numeric_semantics.py
:language: python
```

Use native NumPy computations to teach dtype promotion, rounding, overflow,
or warning policy. State the preview's numerical boundary in those lessons.
Ordinary small integer examples avoid these differences when the intended
topic is coordinate mapping or broadcasting.

## Run the conformance specimens

```bash
python scripts/check_lessons.py runtime-contract numeric-semantics
python -m pytest tests/test_teaching_contracts.py
```

The failure specimens deliberately contain an invented operation, an invalid
focus, a false history claim, collapsed repeated contributions, an incorrect
division assumption, and a dtype mismatch. They must fail the lesson checker.
The positive lessons must pass and display checked figures.

These are deterministic examples of teaching errors. They do not grade free-form
prose or measure a language model's accuracy. The checker executes registered,
trusted repository lessons, not arbitrary submitted Python.
