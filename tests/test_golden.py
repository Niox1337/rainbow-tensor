"""Golden snapshot tests.

Each case renders a canonical input and compares the SVG byte for byte against
a stored golden file. This catches any unintended change to the rendered
output. When a change is intentional, regenerate the goldens with::

    RT_UPDATE_GOLDEN=1 pytest tests/test_golden.py

and review the diff before committing the new files.
"""

import os
from pathlib import Path

import pytest

import rainbow_tensor as rt

GOLDEN_DIR = Path(__file__).parent / "golden"

# Canonical render cases, kept deterministic so the SVG is a pure function of
# the inputs.
CASES = {
    "shape_1d_light": lambda: rt.show_shape((3,)),
    "shape_2d_light": lambda: rt.show_shape((2, 3)),
    "shape_3d_light": lambda: rt.show_shape((2, 2, 2)),
    "shape_3d_dark": lambda: rt.show_shape((2, 2, 2), theme="dark"),
    "index_3d_light": lambda: rt.show_index((2, 2, 2), (0, slice(None), 1)),
    "index_2d_dark": lambda: rt.show_index((3, 4), (1, slice(None)), theme="dark"),
    "truncate_1d_light": lambda: rt.show_shape((30,)),
    "truncate_2d_light": lambda: rt.show_shape((20, 6)),
}


def _render(name):
    return CASES[name]().svg


@pytest.mark.parametrize("name", sorted(CASES))
def test_render_matches_golden(name):
    rendered = _render(name)
    path = GOLDEN_DIR / f"{name}.svg"

    if os.environ.get("RT_UPDATE_GOLDEN"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        pytest.skip(f"updated golden {name}")

    assert path.exists(), (
        f"missing golden for {name}, regenerate with RT_UPDATE_GOLDEN=1 pytest"
    )
    assert rendered == path.read_text(encoding="utf-8")
