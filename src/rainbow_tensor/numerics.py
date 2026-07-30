"""Describe the existing scalar arithmetic used by numerical previews."""


def numeric_semantics(arrays):
    """Describe operand sources without importing or evaluating a backend.

    This is the preview's arithmetic model, not an assertion that values have
    been evaluated. ``value_evaluation`` separately records skipped outputs.
    """
    return {
        "arithmetic": "python_scalar",
        "operand_sources": [
            "generated_values" if isinstance(array, tuple) or not hasattr(array, "shape")
            else "array_values"
            for array in arrays
        ],
    }


def numeric_explanation(semantics):
    """Explain scalar arithmetic and identify generated placeholder operands."""
    lines = ["Numerical model: Python scalar arithmetic."]
    sources = semantics["operand_sources"]
    if "array_values" in sources:
        lines.append(
            "Accumulation dtype, rounding, and overflow may differ from the input array backend."
        )
    for operand, source in enumerate(sources):
        if source == "generated_values":
            lines.append(
                f"Operand {operand} uses generated row-major placeholder values from its shape."
            )
    return lines
