"""Explain reduction shape predictions using explicit source-to-output axes."""

from ast import literal_eval
from dataclasses import dataclass
from math import prod

from ..explanations import t
from ..ops import reduce_result_shape


@dataclass(frozen=True, slots=True)
class ShapePrediction:
    """A parsed answer and its structural comparison with a reduction shape.

    The comparison never evaluates tensor values. ``code`` distinguishes an
    absent or malformed answer, an exact match, forgotten retained dimensions,
    an incorrect rank, and incorrect dimension lengths. Source axes remain
    available for explaining where each surviving output dimension came from.
    """

    code: str
    source_shape: tuple[int, ...]
    reduced_axes: tuple[int, ...]
    keepdims: bool
    expected: tuple[int, ...]
    predicted: tuple[int, ...] | None


def diagnose_prediction(text, source_shape, axes, keepdims):
    """Compare a tuple-shaped answer without executing the learner's text."""
    expected = reduce_result_shape(source_shape, axes, keepdims=keepdims)
    predicted = None
    code = "missing"
    if text.strip():
        try:
            predicted = literal_eval(text)
        except (ValueError, SyntaxError):
            predicted = None
        valid = isinstance(predicted, tuple) and all(
            type(size) is int and size >= 0 for size in predicted
        )
        if not valid:
            predicted = None
            code = "malformed"
        elif predicted == expected:
            code = "matches"
        elif keepdims and axes and predicted == reduce_result_shape(source_shape, axes):
            code = "keepdims"
        elif len(predicted) != len(expected):
            code = "rank"
        else:
            code = "dimension"
    return ShapePrediction(code, tuple(source_shape), tuple(axes), keepdims, expected, predicted)


def prediction_messages(feedback):
    """Translate the diagnosis and explain removed, retained, and empty axes."""
    lines = []
    if feedback.code == "matches":
        lines.append(t("playground.matches"))
    elif feedback.code == "malformed":
        lines.append(t("playground.feedback.malformed"))
    elif feedback.code == "keepdims":
        lines.append(t("playground.feedback.keepdims", axes=feedback.reduced_axes))
    elif feedback.code == "rank":
        lines.append(t(
            "playground.feedback.rank", predicted_rank=len(feedback.predicted),
            expected_rank=len(feedback.expected),
        ))
    elif feedback.code == "dimension":
        lines.extend(
            t("playground.feedback.dimension", axis=axis, predicted=predicted, expected=expected)
            for axis, (predicted, expected) in enumerate(zip(feedback.predicted, feedback.expected))
            if predicted != expected
        )
    if feedback.reduced_axes:
        key = "retained" if feedback.keepdims else "removed"
        lines.append(t(f"playground.feedback.{key}", axes=feedback.reduced_axes))
    else:
        lines.append(t("playground.feedback.no_axes", shape=feedback.source_shape))
    output_axis = 0
    for source_axis, size in enumerate(feedback.source_shape):
        if source_axis in feedback.reduced_axes:
            if feedback.keepdims:
                output_axis += 1
            continue
        lines.append(t(
            "playground.feedback.preserved", source_axis=source_axis,
            output_axis=output_axis, size=size,
        ))
        output_axis += 1
    if not feedback.expected:
        lines.append(t("playground.feedback.scalar"))
    if 0 in feedback.expected:
        lines.append(t("playground.feedback.empty"))
    elif any(feedback.source_shape[axis] == 0 for axis in feedback.reduced_axes):
        lines.append(t("playground.feedback.empty_groups", count=prod(feedback.expected)))
    return tuple(lines)
