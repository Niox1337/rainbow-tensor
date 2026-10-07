"""Portable lesson JSON rejects ambiguous data and preserves captured revisions."""

import json
from dataclasses import FrozenInstanceError

import pytest

from rainbow_tensor.lessons import LessonRecording
from rainbow_tensor.lessons.recording import _LABEL_KEYS


@pytest.fixture
def document():
    """Build the smallest complete version-one document for import boundary tests."""
    return {
        "format": "rainbow-tensor-lesson", "version": 1, "scope": "captured_states_only",
        "title": "A captured sum", "language": "en",
        "labels": dict.fromkeys(_LABEL_KEYS, "Label"),
        "states": [{
            "kind": "visual", "svg": '<svg xmlns="http://www.w3.org/2000/svg"/>',
            "text": "Two recorded contributions", "shape": ["2", "3"],
            "result_shape": ["2"], "focus": ["1"], "metadata": {},
            "completeness": {"trace": "complete", "ancestry": "unknown", "values": "unknown"},
            "lesson": None, "prediction": None,
        }],
    }


def load(document, **limits):
    return LessonRecording.from_json(json.dumps(document, ensure_ascii=False), **limits)


def test_round_trip_is_deterministic_and_detached_from_mutations(document):
    recording = load(document)
    assert recording.state_count == 1
    assert recording.to_dict() == document
    assert LessonRecording.from_json(recording.to_json()).to_json() == recording.to_json()
    document["states"][0]["text"] = "Changed after capture"
    detached = recording.to_dict()
    detached["states"].clear()
    assert recording.to_dict()["states"][0]["text"] == "Two recorded contributions"
    with pytest.raises(FrozenInstanceError):
        recording._json = "{}"


@pytest.mark.parametrize("key,value", [
    ("version", 2), ("version", True), ("format", "another-format"),
    ("scope", "all_tensor_positions"), ("title", None), ("language", 1),
    ("labels", {}), ("states", []), ("states", None),
])
def test_import_rejects_unknown_or_malformed_document_contract(document, key, value):
    document[key] = value
    with pytest.raises(ValueError):
        load(document)


@pytest.mark.parametrize("key,value", [
    ("kind", "live_callback"), ("svg", "<html/>"), ("svg", "<svg>"),
    ("text", ["not a string"]), ("shape", [2, 3]), ("shape", ["-1"]),
    ("result_shape", ["03"]), ("focus", ["2"]), ("focus", []),
    ("metadata", []), ("completeness", {"trace": "complete"}),
    ("lesson", "not an object"), ("prediction", {"revealed": True}),
])
def test_import_rejects_malformed_states(document, key, value):
    document["states"][0][key] = value
    with pytest.raises(ValueError):
        load(document)


def test_large_coordinates_survive_json_and_scalar_and_empty_states_remain_distinct(document):
    state = document["states"][0]
    state["result_shape"] = [str(2**64)]
    state["focus"] = [str(2**63 + 1)]
    assert load(document).to_dict()["states"][0]["focus"] == ["9223372036854775809"]
    state["result_shape"], state["focus"] = [], []
    assert load(document).to_dict()["states"][0]["focus"] == []
    state["result_shape"], state["focus"] = ["0"], None
    assert load(document).to_dict()["states"][0]["focus"] is None


def test_limits_count_utf8_bytes_and_all_states(document):
    document["states"][0]["text"] = "λ 😀"
    serialized = load(document).to_json()
    exact = len(serialized.encode("utf-8"))
    assert LessonRecording.from_json(serialized, max_bytes=exact).to_json() == serialized
    with pytest.raises(ValueError, match="max_bytes"):
        LessonRecording.from_json(serialized, max_bytes=exact - 1)
    document["states"] *= 2
    with pytest.raises(ValueError, match="max_states"):
        load(document, max_states=1)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
@pytest.mark.parametrize("name", ["max_states", "max_bytes"])
def test_limits_require_positive_integers(document, name, limit):
    with pytest.raises(ValueError, match=name):
        load(document, **{name: limit})


def test_duplicate_keys_and_nonfinite_numbers_are_rejected_before_schema_checks():
    with pytest.raises(ValueError, match="duplicate"):
        LessonRecording.from_json('{"version":1,"version":1}')
    with pytest.raises(ValueError, match="non-finite"):
        LessonRecording.from_json('{"metadata":NaN}')
    with pytest.raises(ValueError, match="max_bytes"):
        LessonRecording.from_json("not valid JSON", max_bytes=1)


def test_deep_metadata_and_svg_entity_declarations_are_rejected(document):
    nested = {}
    for _ in range(26):
        nested = {"nested": nested}
    document["states"][0]["metadata"] = nested
    with pytest.raises(ValueError, match="nesting depth"):
        load(document)
    document["states"][0]["metadata"] = {}
    document["states"][0]["svg"] = '<!DOCTYPE svg [<!ENTITY a "value">]><svg>&a;</svg>'
    with pytest.raises(ValueError, match="entities"):
        load(document)


def test_import_does_not_claim_unrecorded_states_or_numerical_completeness(document):
    state = document["states"][0]
    state["completeness"] = {"trace": "truncated", "ancestry": "truncated", "values": "unknown"}
    state["metadata"] = {"truncated_reasons": ["max_depth"], "operation": "sum"}
    imported = load(document).to_dict()
    assert imported["scope"] == "captured_states_only"
    assert imported["states"][0]["completeness"]["values"] == "unknown"
    assert imported["states"][0]["metadata"]["truncated_reasons"] == ["max_depth"]
