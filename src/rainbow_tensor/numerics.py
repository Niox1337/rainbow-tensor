"""Describe the existing scalar arithmetic used by numerical previews."""

from .explanations import t


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
    lines = [t("numeric.model")]
    sources = semantics["operand_sources"]
    if "array_values" in sources:
        lines.append(t("numeric.backend_difference"))
    for operand, source in enumerate(sources):
        if source == "generated_values":
            lines.append(t("numeric.generated", operand=operand))
    return lines
