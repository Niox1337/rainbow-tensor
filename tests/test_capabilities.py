"""Installed capability documents describe callable teaching operations faithfully."""

import json
from inspect import signature

import rainbow_tensor as rt
from rainbow_tensor import views
from rainbow_tensor.capabilities import capabilities, explorable_operations


def test_inventory_matches_public_operations_and_actual_signatures():
    """Agents receive the installed signatures rather than a hand-maintained copy."""
    document = capabilities()
    assert document["schema_version"] == 1
    assert document["package"] == {"name": "rainbow-tensor", "version": rt.__version__}
    assert set(document["operations"]) == {*views.__all__, "memory"}
    for name, record in document["operations"].items():
        function = getattr(views, name) if name != "memory" else rt.memory
        assert record["signature"] == str(signature(function))
        assert record["explore"] == (function in explorable_operations())
        assert (record["flow_signature"] is not None) == callable(getattr(rt.Flow, name, None))
    assert "softmax" not in document["operations"]
    assert not document["operations"]["shape"]["explore"]
    assert not document["operations"]["memory"]["explore"]


def test_capabilities_round_trip_without_shared_mutable_state():
    """A consumer may annotate its copy without changing the next caller's contract."""
    document = capabilities()
    assert json.loads(json.dumps(document, allow_nan=False)) == document
    document["operations"]["sum"]["signature"] = "invented()"
    document["controllers"]["explore"]["operations"].clear()
    fresh = capabilities()
    assert fresh["operations"]["sum"]["signature"] == str(signature(rt.sum))
    assert "sum" in fresh["controllers"]["explore"]["operations"]


def test_controller_scope_and_numeric_boundaries_are_explicit():
    """A lesson cannot infer backend execution or unsupported standalone walkthroughs."""
    document = capabilities()
    assert document["controllers"]["walkthrough"]["operations"] == ["matmul", "mean", "sum"]
    assert document["controllers"]["walkthrough"]["tracked_tensor"]
    assert document["semantics"]["arithmetic"] == "python_scalars"
    assert document["semantics"]["structural_trace_reads_values"] is False
    assert document["semantics"]["backend_kernels"] is False
    assert document["semantics"]["divide_by_zero"] == "raises_ZeroDivisionError"


def test_selection_rules_and_recording_limits_match_the_runtime():
    """Portable output and data-dependent choices carry their own explicit limits."""
    from rainbow_tensor.lessons.recording import DEFAULT_MAX_BYTES, DEFAULT_MAX_STATES

    document = capabilities()
    assert document["operations"]["where"]["explore"]
    assert not document["operations"]["where"]["walkthrough"]
    assert document["operations"]["argmax"]["flow_signature"] is not None
    assert document["semantics"]["where_evaluation"] == "condition_and_both_branches"
    assert document["semantics"]["extrema_nan"] == "first_nan"
    recording = document["exports"]["lesson"]
    assert recording["max_bytes"] == DEFAULT_MAX_BYTES
    assert recording["max_states"] == DEFAULT_MAX_STATES
    assert recording["save_format"] == "html"
    assert recording["requires_live_kernel"] is False
