"""Renderer registry tests."""

import pytest

import rainbow_tensor as rt


class RecordingRenderer:
    name = "recording"
    mime_type = "text/plain"

    def __init__(self):
        self.calls = []

    def render_tensor(self, **kwargs):
        self.calls.append(("tensor", kwargs))
        return "one tensor"

    def render_panels(self, **kwargs):
        self.calls.append(("panels", kwargs))
        return "many tensors"


def test_shape_accepts_renderer_object():
    renderer = RecordingRenderer()
    visual = rt.shape((2, 3), renderer=renderer)

    assert visual.svg == "one tensor"
    assert visual.content == "one tensor"
    assert visual.renderer == "recording"
    assert visual.mime_type == "text/plain"
    assert visual._repr_svg_() is None
    kind, kwargs = renderer.calls[0]
    assert kind == "tensor"
    assert kwargs["shape"] == (2, 3)


def test_panel_operation_accepts_renderer_object():
    renderer = RecordingRenderer()
    visual = rt.reshape((2, 3), (3, 2), renderer=renderer)

    assert visual.svg == "many tensors"
    assert visual.result_shape == (3, 2)
    kind, kwargs = renderer.calls[0]
    assert kind == "panels"
    assert len(kwargs["panels"]) == 2


def test_renderer_registry_and_default_round_trip():
    renderer = RecordingRenderer()
    old = rt.get_default_renderer()
    rt.register_renderer(renderer)

    try:
        assert rt.resolve_renderer("recording") is renderer
        rt.set_default_renderer("recording")
        assert rt.shape((2,)).renderer == "recording"
    finally:
        rt.set_default_renderer(old)


def test_renderer_registry_rejects_missing_methods():
    class BadRenderer:
        name = "bad"

    with pytest.raises(TypeError):
        rt.register_renderer(BadRenderer())
