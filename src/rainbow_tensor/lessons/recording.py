"""Validate versioned lesson data without retaining arrays or live controllers."""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import isfinite

DEFAULT_MAX_STATES = 64
DEFAULT_MAX_BYTES = 5_000_000
_FORMAT = "rainbow-tensor-lesson"
_SCOPE = "captured_states_only"
_LABEL_KEYS = frozenset({
    "previous", "next", "state", "scope", "focus", "unknown", "complete", "truncated",
    "trace", "ancestry", "values", "details", "prediction", "reveal", "hide", "source",
    "result", "expression", "empty", "lesson", "keyboard",
})
_STATE_KEYS = frozenset({
    "kind", "svg", "text", "shape", "result_shape", "focus", "metadata", "completeness",
    "lesson", "prediction",
})
_POSITION = re.compile(r"(?:0|[1-9][0-9]*)\Z")


def _limit(value, name):
    """Reject ambiguous or unbounded recording limits before consuming input."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _encoded(document, max_bytes):
    """Serialize deterministically while limiting UTF-8 bytes, not character count."""
    chunks = []
    size = 0
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, sort_keys=True,
                               separators=(",", ":"))
    try:
        for chunk in encoder.iterencode(document):
            size += len(chunk.encode("utf-8"))
            if size > max_bytes:
                raise ValueError(f"lesson recording exceeds max_bytes={max_bytes}")
            chunks.append(chunk)
    except (TypeError, UnicodeError, RecursionError) as error:
        raise ValueError(f"lesson recording contains invalid JSON data: {error}") from error
    return "".join(chunks)


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def _text(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def _json_tree(value, depth=0):
    """Bound imported metadata depth and reject values outside ordinary JSON."""
    if depth > 24:
        raise ValueError("lesson metadata exceeds the maximum nesting depth of 24")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_tree(item, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_tree(item, depth + 1)
        return
    raise ValueError("lesson metadata must contain finite JSON values")


def _shape(value, name, *, nullable=False, nested=True):
    if value is None and nullable:
        return
    if not isinstance(value, list):
        raise ValueError(f"{name} must contain decimal dimension strings")
    for dimension in value:
        if nested and isinstance(dimension, list):
            _shape(dimension, name, nested=False)
        elif not isinstance(dimension, str) or not _POSITION.fullmatch(dimension):
            raise ValueError(f"{name} must contain nonnegative decimal strings")


def _svg(value, name):
    """Check SVG structure while rejecting entity declarations in imported images."""
    _text(value, name)
    if "<!DOCTYPE" in value.upper() or "<!ENTITY" in value.upper():
        raise ValueError(f"{name} must not declare XML entities or a document type")
    try:
        root = ET.fromstring(value)
    except ET.ParseError as error:
        raise ValueError(f"{name} is not valid SVG: {error}") from error
    if root.tag not in ("svg", "{http://www.w3.org/2000/svg}svg"):
        raise ValueError(f"{name} must have an SVG root element")


def _state(state):
    """Validate one frozen presentation without interpreting its source metadata."""
    _keys(state, _STATE_KEYS, "state")
    if state["kind"] not in ("visual", "explorer", "walkthrough", "playground"):
        raise ValueError("state has an unsupported kind")
    _svg(state["svg"], "state.svg")
    _text(state["text"], "state.text")
    _shape(state["shape"], "state.shape")
    _shape(state["result_shape"], "state.result_shape", nullable=True)
    _shape(state["focus"], "state.focus", nullable=True, nested=False)
    result = state["result_shape"]
    focus = state["focus"]
    if focus is not None and result is not None and all(isinstance(n, str) for n in result):
        if len(focus) != len(result) or any(int(c) >= int(n) for c, n in zip(focus, result)):
            raise ValueError("state.focus is outside the captured result shape")
    if not isinstance(state["metadata"], dict):
        raise ValueError("state.metadata must be an object")
    _json_tree(state["metadata"])
    _keys(state["completeness"], {"trace", "ancestry", "values"}, "state.completeness")
    if any(value not in ("unknown", "complete", "truncated")
           for value in state["completeness"].values()):
        raise ValueError("state.completeness contains an unsupported status")
    if state["lesson"] is not None:
        if not isinstance(state["lesson"], dict):
            raise ValueError("state.lesson must be an object or null")
        _json_tree(state["lesson"])
    prediction = state["prediction"]
    if prediction is not None:
        _keys(prediction, {"prompt", "answer", "expression", "revealed", "source_svg"},
              "state.prediction")
        for key in ("prompt", "answer", "expression"):
            _text(prediction[key], f"state.prediction.{key}")
        if not isinstance(prediction["revealed"], bool):
            raise ValueError("state.prediction.revealed must be a boolean")
        _svg(prediction["source_svg"], "state.prediction.source_svg")


def _document(document, max_states, max_bytes):
    """Validate format identity, state limits, and every replayed presentation."""
    _limit(max_states, "max_states")
    _limit(max_bytes, "max_bytes")
    _keys(document, {"format", "version", "scope", "title", "language", "labels", "states"},
          "lesson recording")
    if document["format"] != _FORMAT or type(document["version"]) is not int:
        raise ValueError("unrecognized lesson recording format")
    if document["version"] != 1:
        raise ValueError(f"unsupported lesson recording version {document['version']!r}")
    if document["scope"] != _SCOPE:
        raise ValueError("lesson recording must describe captured states only")
    _text(document["title"], "title")
    _text(document["language"], "language")
    _keys(document["labels"], _LABEL_KEYS, "labels")
    for value in document["labels"].values():
        _text(value, "label")
    if not isinstance(document["states"], list) or not document["states"]:
        raise ValueError("a lesson recording requires at least one captured state")
    if len(document["states"]) > max_states:
        raise ValueError(f"lesson recording exceeds max_states={max_states}")
    serialized = _encoded(document, max_bytes)
    for state in document["states"]:
        _state(state)
    return serialized


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate lesson recording key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"lesson recording contains non-finite JSON constant {value}")


@dataclass(frozen=True, slots=True, init=False)
class LessonRecording:
    """Immutable, versioned presentations of explicitly captured lesson states.

    Create a recording with :func:`capture_lesson` or :meth:`from_json`.
    Recordings retain no input arrays, callbacks, or notebook connections.
    Coordinates and dimensions use decimal strings in JSON to preserve integers
    beyond JavaScript's exact range. Metadata describes the captured revision
    only. It is never imported as Python objects or executed as code.
    """

    _json: str

    def __init__(self):
        raise TypeError("create a lesson recording with capture_lesson or from_json")

    @classmethod
    def _from_document(cls, document, *, max_states=DEFAULT_MAX_STATES,
                       max_bytes=DEFAULT_MAX_BYTES):
        recording = object.__new__(cls)
        object.__setattr__(recording, "_json", _document(document, max_states, max_bytes))
        return recording

    @classmethod
    def from_json(cls, text, *, max_states=DEFAULT_MAX_STATES, max_bytes=DEFAULT_MAX_BYTES):
        """Load validated UTF-8 JSON within explicit state and byte limits.

        Unknown versions, duplicate keys, non-finite numbers, malformed SVG,
        invalid coordinates, and nested metadata beyond 24 levels are rejected.
        The byte limit applies before parsing and includes all SVG and text.
        """
        _limit(max_states, "max_states")
        _limit(max_bytes, "max_bytes")
        _text(text, "lesson JSON")
        try:
            if len(text.encode("utf-8")) > max_bytes:
                raise ValueError(f"lesson recording exceeds max_bytes={max_bytes}")
            document = json.loads(text, object_pairs_hook=_unique_pairs,
                                  parse_constant=_invalid_constant)
        except (UnicodeError, RecursionError) as error:
            raise ValueError(f"invalid lesson recording JSON: {error}") from error
        return cls._from_document(document, max_states=max_states, max_bytes=max_bytes)

    @property
    def state_count(self):
        """Return the number of captured presentations, not possible tensor positions."""
        return len(self.to_dict()["states"])

    def to_dict(self):
        """Return detached JSON data that callers may inspect or edit independently."""
        return json.loads(self._json)

    def to_json(self):
        """Return deterministic version-one JSON with captured text and SVG intact."""
        return self._json
