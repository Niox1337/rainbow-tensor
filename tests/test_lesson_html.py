"""Portable HTML keeps imported text inert and preserves explicit output paths."""

import base64
import hashlib
import json
from html.parser import HTMLParser

import pytest

from rainbow_tensor.lessons import LessonRecording, capture_lesson
from rainbow_tensor.visual import TensorVisual


class ParsedPlayer(HTMLParser):
    """Inspect actual HTML parsing boundaries rather than matching encoded strings."""

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.scripts = []
        self.policy = None
        self.tags = []
        self.active = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attributes = dict(attrs)
        if tag == "script":
            self.scripts.append([attributes, ""])
            self.active = self.scripts[-1]
        if tag == "meta" and attributes.get("http-equiv") == "Content-Security-Policy":
            self.policy = attributes["content"]

    def handle_data(self, data):
        if self.active is not None:
            self.active[1] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.active = None


def test_hostile_svg_text_title_and_metadata_never_create_active_markup():
    attack = '</script><img src="https://invalid.example/" onerror="window.injected=1">'
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><script>window.injected=1</script></svg>'
    visual = TensorVisual(svg, (), explanation=[attack, "quotes ' \" & λ\u2028\u2029"],
                          metadata={"text": attack})
    recording = capture_lesson([visual], title=attack)
    player = ParsedPlayer(recording.to_html())
    assert len(player.scripts) == 2
    assert "svg" not in player.tags
    assert "iframe" not in player.tags
    data = json.loads(player.scripts[0][1])
    assert data["title"] == attack
    assert data["states"][0]["text"] == visual.text
    assert data["states"][0]["metadata"]["text"] == attack
    image = data["states"][0]["image"]
    assert image.startswith("data:image/svg+xml;base64,")
    assert base64.b64decode(image.split(",", 1)[1]).decode() == svg
    assert "\\u003c/script\\u003e" in player.scripts[0][1]
    assert "\\u2028\\u2029" in player.scripts[0][1]


def test_content_policy_permits_only_the_exact_player_script_and_embedded_images():
    recording = capture_lesson([TensorVisual("<svg/>", ())])
    player = ParsedPlayer(recording.to_html())
    script = player.scripts[1][1]
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert f"script-src 'sha256-{digest}'" in player.policy
    assert "default-src 'none'" in player.policy
    assert "connect-src 'none'" in player.policy
    assert "img-src data:" in player.policy
    assert "'unsafe-eval'" not in player.policy
    assert "innerHTML" not in script
    assert "textContent" in script


def test_save_accepts_path_objects_and_writes_deterministic_utf8_html(tmp_path):
    recording = capture_lesson([TensorVisual("<svg/>", (), explanation=["λ 😀"])])
    path = tmp_path / "A lesson with spaces.html"
    assert recording.save(path) == path
    assert path.read_text(encoding="utf-8") == recording.to_html()
    assert "λ 😀" in path.read_text(encoding="utf-8")
    assert LessonRecording.from_json(recording.to_json()).to_html() == recording.to_html()
    path.write_text("old content", encoding="utf-8")
    recording.save(str(path))
    assert path.read_text(encoding="utf-8") == recording.to_html()


def test_save_reports_missing_parent_without_creating_unrequested_directories(tmp_path):
    recording = capture_lesson([TensorVisual("<svg/>", ())])
    with pytest.raises(FileNotFoundError):
        recording.save(tmp_path / "missing" / "lesson.html")
    assert not (tmp_path / "missing").exists()
