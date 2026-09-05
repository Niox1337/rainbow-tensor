"""Exercise translated output across the public visual operations."""

from xml.etree import ElementTree as ET

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.explanations import MESSAGES
from rainbow_tensor.render_svg import _text_width, render_panels


@pytest.mark.parametrize("operation", [
    lambda: rt.shape(np.arange(24).reshape(2, 3, 4)),
    lambda: rt.index((2, 3), (0, slice(None)), show_result=True),
    lambda: rt.index((2, 3), [[True, False, True], [False, True, False]]),
    lambda: rt.reshape((2, 3), (3, 2)),
    lambda: rt.transpose((2, 3)),
    lambda: rt.swapaxes((2, 3, 4), 0, 2),
    lambda: rt.moveaxis((2, 3, 4), 0, 2),
    lambda: rt.squeeze((1, 3, 1)),
    lambda: rt.expand_dims((2, 3), 1),
    lambda: rt.repeat((2, 3), 2, axis=1),
    lambda: rt.take((2, 3), [2, 0], axis=1),
    lambda: rt.stack([(2, 3), (2, 3)]),
    lambda: rt.concatenate([(2, 3), (2, 3)]),
    lambda: rt.broadcast((2, 1), (1, 3)),
    lambda: rt.sum((2, 3, 4), axis=(0, 2), keepdims=True),
    lambda: rt.mean((2, 0), axis=1),
    lambda: rt.matmul((2, 0), (0, 3)),
    lambda: rt.einsum("i,i->", (3,), (3,), focus=()),
])
def test_chinese_visuals_keep_shapes_and_translate_display(operation):
    english = operation()
    rt.set_language("zh-CN")
    chinese = operation()
    root = ET.fromstring(chinese.svg)
    assert root.attrib["lang"] == "zh"
    assert chinese.shape == english.shape
    assert chinese.result_shape == english.result_shape
    assert any("\u4e00" <= char <= "\u9fff" for char in chinese.svg)
    assert "value " not in chinese.svg
    if english.text:
        assert chinese.text != english.text


def test_translated_hover_and_empty_accessibility_labels():
    rt.set_language("zh")
    scalar = ET.fromstring(rt.shape(np.array(7)).svg)
    titles = [node.text for node in scalar.iter() if node.tag.endswith("title")]
    assert titles == ["[]  值 7  ·  展平索引 0"]
    empty = ET.fromstring(rt.shape((2, 0)).svg)
    note = next(node for node in empty.iter() if node.attrib.get("role") == "note")
    assert note.attrib["aria-label"] == "空张量形状 (2, 0)，没有元素"
    assert "没有元素" in "".join(empty.itertext())


def test_translation_text_is_escaped_in_svg(monkeypatch):
    monkeypatch.setitem(MESSAGES, "xx", {"label.shape": '<shape & "values">'})
    rt.set_language("xx")
    svg = rt.shape((2,)).svg
    root = ET.fromstring(svg)
    assert root.attrib["aria-label"] == '<shape & "values"> (2)'
    assert '<shape & "values">' not in svg


def test_long_translated_caption_expands_its_panel():
    caption = "源张量" * 12
    root = ET.fromstring(render_panels([
        {"shape": (1,), "caption_parts": [(caption, "#000000")]},
    ], theme="light"))
    assert float(root.attrib["width"]) >= _text_width(caption, 14, 0.55) + 39


def test_wide_and_combining_glyph_measurements():
    assert _text_width("形状", 10, 0.5) == 20
    assert _text_width("cafe\u0301", 10, 0.5) == _text_width("cafe", 10, 0.5)


def test_shipped_chinese_catalog_covers_every_english_key():
    assert MESSAGES["zh"].keys() == MESSAGES["en"].keys()
