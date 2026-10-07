"""Incorrect teaching specimens fail the same checker as published examples."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_lessons.py"
    spec = importlib.util.spec_from_file_location("teaching_contract_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("body, error", [
    ("assert rt.capabilities()['operations']['softmax']", KeyError),
    ("display(rt.sum(x, axis=1, focus=(9,)))\nassert True", IndexError),
    ("visual = rt.sum(x, axis=1)\nassert visual.provenance is not None", AssertionError),
    (
        "flow = rt.Flow()\na = flow.input(x)\nb = flow.index(a, ([1, 0, 1], 2))\n"
        "y = flow.sum(b)\n"
        "assert {r.reference.coordinate: r.count for r in y.trace(()).roots} == "
        "{(1, 2): 1, (0, 2): 1}",
        AssertionError,
    ),
    ("display(rt.divide(x, 0))\nassert True", ZeroDivisionError),
    (
        "flow = rt.Flow()\na = flow.input(np.array([250], dtype=np.uint8))\n"
        "b = flow.input(np.array([10], dtype=np.uint8))\n"
        "assert flow.add(a, b).value((0,)) == 4",
        AssertionError,
    ),
])
def test_misleading_generated_specimens_are_rejected(checker, tmp_path, body, error):
    """Invalid APIs, focus, history, duplicate counts and dtype claims cannot pass."""
    source = tmp_path / "incorrect_lesson.py"
    source.write_text(
        "import numpy as np\nfrom IPython.display import display\n"
        "import rainbow_tensor as rt\nx = np.arange(1, 7).reshape(2, 3)\n" + body + "\n",
        encoding="utf-8",
    )
    with pytest.raises(error):
        checker.execute_lesson(source, tmp_path)


@pytest.mark.parametrize("name", ["runtime-contract", "numeric-semantics"])
def test_teaching_contract_lessons_are_executable(checker, name):
    report = checker.check_lesson(name)
    assert report["figures"] == 2
    assert report["assertions"] >= 4
