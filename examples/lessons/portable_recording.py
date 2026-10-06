"""Freeze two selected outputs into a finite lesson that needs no live kernel."""

from pathlib import Path

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

values = np.arange(1, 7).reshape(2, 3)
explorer = rt.explore(rt.sum, values, axis=1, focus=(0,))
display(explorer.visual)


def selected_states():
    """Yield the same controller after each explicitly chosen focus change."""
    for coordinate in ((0,), (1,)):
        explorer.set_focus(coordinate)
        yield explorer


recording = rt.capture_lesson(selected_states(), title="Two row sums", max_states=2)
assert recording.state_count == 2
document = recording.to_dict()
assert document["scope"] == "captured_states_only"
assert [state["focus"] for state in document["states"]] == [["0"], ["1"]]
assert document["states"][0]["text"] != document["states"][1]["text"]
assert rt.LessonRecording.from_json(recording.to_json()).to_dict() == document
recording.save("row-sums.html")
Path("row-sums.json").write_text(recording.to_json(), encoding="utf-8")
assert Path("row-sums.html").read_text(encoding="utf-8").startswith("<!doctype html>")
assert Path("row-sums.json").read_text(encoding="utf-8") == recording.to_json()
display(explorer.visual)

# Subsequent changes do not alter the stored states or require a Python kernel.
explorer.set_focus((0,))
assert recording.to_dict() == document
explorer.close()
