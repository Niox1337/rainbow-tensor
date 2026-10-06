"""Copy explicit visual states without evaluating or retaining tensor inputs."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from html.parser import HTMLParser
from math import isfinite
from numbers import Integral, Real

from ..explanations import get_resolved_language, t
from ..visual import TensorVisual
from .recording import (
    _FORMAT,
    _LABEL_KEYS,
    _SCOPE,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_STATES,
    LessonRecording,
    _encoded,
    _limit,
)


class _CopyBudget:
    """Stop metadata traversal before building an unbounded JSON-compatible copy."""

    def __init__(self, limit):
        self.remaining = limit

    def consume(self, size):
        self.remaining -= size
        if self.remaining < 0:
            raise ValueError("lesson metadata exceeds max_bytes")


class _PlainFeedback(HTMLParser):
    """Keep visible feedback words without exporting active notebook HTML."""

    def __init__(self, markup):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.feed(markup)

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


def _plain(value, budget, depth=0):
    """Copy finite JSON data, preserving unsupported values as explicit unknowns."""
    if depth > 24:
        raise ValueError("lesson metadata exceeds the maximum nesting depth of 24")
    budget.consume(1)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        budget.consume(len(value.encode("utf-8")))
        return value
    if isinstance(value, Integral):
        integer = int(value)
        budget.consume(len(str(integer)))
        return str(integer) if abs(integer) > 2**53 - 1 else integer
    if isinstance(value, Real):
        number = float(value)
        return number if isfinite(number) else {"kind": "nonfinite", "value": str(number)}
    if isinstance(value, (tuple, list)):
        return [_plain(item, budget, depth + 1) for item in value]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return {"status": "unknown", "type": type(value).__name__}
            budget.consume(len(key.encode("utf-8")))
            result[key] = _plain(item, budget, depth + 1)
        return result
    if is_dataclass(value) and not isinstance(value, type):
        return _plain({field.name: getattr(value, field.name) for field in fields(value)},
                      budget, depth + 1)
    return {"status": "unknown", "type": type(value).__name__}


def _shape(shape):
    if shape is None:
        return None
    return [_shape(size) if isinstance(size, (tuple, list)) else str(size) for size in shape]


def _completeness(trace):
    if trace is None:
        return "unknown"
    return "complete" if trace.complete else "truncated"


def _capture(source, max_bytes):
    """Read only cached controller properties and detach one complete presentation."""
    from ..interactive import FocusExplorer
    from ..playground import ReductionPlayground
    from ..walkthrough import Walkthrough

    prediction = None
    if isinstance(source, TensorVisual):
        visual, kind = source, "visual"
    elif isinstance(source, Walkthrough):
        visual, kind = source.visual, "walkthrough"
    elif isinstance(source, ReductionPlayground):
        visual, kind = source.visual, "playground"
        feedback = _PlainFeedback(source.feedback_view.value)
        prediction = {
            "prompt": "\n".join((source.status.value, *feedback.parts)),
            "answer": source.prediction.value,
            "expression": source.expression, "revealed": source.revealed,
            "source_svg": source.input_visual.svg,
        }
    elif isinstance(source, FocusExplorer):
        visual, kind = source.visual, "explorer"
    else:
        raise TypeError("capture_lesson states must be TensorVisual or lesson controllers")
    if visual.mime_type != "image/svg+xml":
        raise ValueError("capture_lesson requires SVG visuals")
    budget = _CopyBudget(max_bytes)
    metadata = {}
    for key, value in visual.metadata.items():
        if key != "lesson":
            budget.consume(len(key.encode("utf-8")))
            metadata[key] = _plain(value, budget)
    if kind == "playground":
        metadata["prediction_feedback"] = _plain(getattr(source, "feedback", None), budget)
    metadata["trace"] = _plain(visual.trace, budget)
    metadata["provenance"] = _plain(visual.provenance, budget)
    lesson = None
    snapshot = visual.metadata.get("lesson")
    if snapshot is not None:
        lesson = _plain({
            "occurrence": snapshot.occurrence, "term": snapshot.term,
            "available_terms": snapshot.available_terms,
            "factor_occurrences": snapshot.factor_occurrences,
            "factor_values": snapshot.factor_values, "term_value": snapshot.term_value,
            "subtotal": snapshot.subtotal, "output_value": snapshot.output_value,
            "binary_operator": snapshot.binary_operator,
        }, budget)
    evaluation = visual.metadata.get("value_evaluation", {})
    value_status = {"evaluated": "complete", "skipped": "truncated"}.get(
        evaluation.get("status"), "unknown",
    )
    focus = visual.trace.output_coord if visual.trace is not None else None
    return {
        "kind": kind, "svg": visual.svg, "text": visual.text,
        "shape": _shape(visual.shape), "result_shape": _shape(visual.result_shape),
        "focus": _shape(focus), "metadata": metadata,
        "completeness": {
            "trace": _completeness(visual.trace),
            "ancestry": _completeness(visual.provenance), "values": value_status,
        },
        "lesson": lesson, "prediction": prediction,
    }


def capture_lesson(states, *, title=None, max_states=DEFAULT_MAX_STATES,
                   max_bytes=DEFAULT_MAX_BYTES):
    """Freeze explicitly chosen visuals or current controller states for offline replay.

    ``states`` is a finite iterable of ``TensorVisual``, ``FocusExplorer``,
    ``Walkthrough``, or ``ReductionPlayground`` objects. Each item is copied when
    yielded, so a generator may move a controller and yield it repeatedly.
    Capture itself never changes focus, reads array values, or enumerates tensor
    positions. At most ``max_states + 1`` items are requested to detect overflow.

    ``max_bytes`` bounds the complete UTF-8 JSON document, including captured
    SVG, explanations, metadata, and translated player labels. Exceeding either
    limit raises ``ValueError`` rather than silently omitting states. The HTML
    player adds fixed presentation code and image encoding overhead afterward.
    A hidden prediction keeps its captured result hidden until revealed offline.
    Numerical and structural omissions retain independent completeness flags.
    """
    _limit(max_states, "max_states")
    _limit(max_bytes, "max_bytes")
    if title is not None and not isinstance(title, str):
        raise TypeError("lesson title must be a string or None")
    document = {
        "format": _FORMAT, "version": 1, "scope": _SCOPE,
        "title": t("recording.title") if title is None else title,
        "language": get_resolved_language(),
        "labels": {key: t(f"recording.{key}") for key in _LABEL_KEYS}, "states": [],
    }
    remaining = max_bytes - len(_encoded(document, max_bytes).encode("utf-8"))
    for position, source in enumerate(states):
        if position >= max_states:
            raise ValueError(f"lesson recording exceeds max_states={max_states}")
        state = _capture(source, remaining)
        remaining -= len(_encoded(state, remaining).encode("utf-8")) + int(position > 0)
        document["states"].append(state)
    return LessonRecording._from_document(document, max_states=max_states, max_bytes=max_bytes)
