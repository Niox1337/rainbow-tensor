"""The documented lesson checker fails clearly and keeps temporary work isolated."""

import importlib.util
import sys
import traceback
from pathlib import Path

import IPython.display
import pytest

import rainbow_tensor as rt


@pytest.fixture
def checker():
    """Load the checkout's contributor command without running its CLI."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_lessons.py"
    spec = importlib.util.spec_from_file_location("lesson_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_lesson_runs_from_its_own_checkout(checker, tmp_path, monkeypatch):
    """An unrelated current directory cannot change the source or receive exports."""
    monkeypatch.chdir(tmp_path)
    report = checker.check_lesson("operation-origins")
    assert report["figures"] == 2
    assert report["assertions"] >= 10
    assert list(tmp_path.iterdir()) == []


def test_unknown_cli_name_cannot_execute_a_path(checker, capsys):
    """The command accepts lesson names, never a caller-supplied Python filename."""
    with pytest.raises(SystemExit) as caught:
        checker.main(["../../unexpected.py"])
    assert caught.value.code == 2
    assert "unknown lesson '../../unexpected.py'" in capsys.readouterr().err


def test_registered_source_cannot_escape_lesson_directory(checker, tmp_path, monkeypatch):
    """A broken registry entry cannot resolve a source outside the lesson directory."""
    lesson_directory = tmp_path / "lessons"
    lesson_directory.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("raise AssertionError('must not run')\n", encoding="utf-8")
    monkeypatch.setattr(checker, "LESSON_DIRECTORY", lesson_directory)
    monkeypatch.setattr(checker, "LESSONS", {"escape": "../outside.py"})
    with pytest.raises(ValueError, match="outside its directory"):
        checker.check_lesson("escape")


def test_missing_registered_source_fails_the_command(checker, tmp_path, monkeypatch, capsys):
    """An omitted packaged lesson is a validation failure rather than a silent skip."""
    monkeypatch.setattr(checker, "LESSON_DIRECTORY", tmp_path)
    assert checker.main(["reduction"]) == 1
    assert "Lesson source is missing" in capsys.readouterr().err


def test_failed_assertion_names_source_line_and_closes_widget(checker, tmp_path, monkeypatch):
    """A lesson failure restores display settings and releases its live controls."""
    source = tmp_path / "wrong_answer.py"
    source.write_text(
        "import rainbow_tensor as rt\n"
        "explorer = rt.explore(rt.sum, (2, 3), axis=1)\n"
        "assert False, 'the predicted answer is wrong'\n",
        encoding="utf-8",
    )
    explorers = []
    original_explore = rt.explore

    def record(*args, **kwargs):
        explorer = original_explore(*args, **kwargs)
        explorers.append(explorer)
        return explorer

    monkeypatch.setattr(rt, "explore", record)
    original_display = IPython.display.display
    original_language = rt.get_language()
    original_theme = rt.get_default_theme()
    with pytest.raises(AssertionError, match="predicted answer") as caught:
        checker.execute_lesson(source, tmp_path)
    last_frame = traceback.extract_tb(caught.value.__traceback__)[-1]
    assert Path(last_frame.filename) == source
    assert last_frame.lineno == 3
    assert IPython.display.display is original_display
    assert rt.get_language() == original_language
    assert rt.get_default_theme() is original_theme
    assert len(explorers) == 1
    with pytest.raises(RuntimeError, match="closed"):
        explorers[0].set_focus((0,))


def test_lesson_without_assertions_is_rejected_before_execution(checker, tmp_path):
    """A file that merely renders something is not a checked teaching example."""
    source = tmp_path / "unchecked.py"
    source.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no executable assertions"):
        checker.execute_lesson(source, tmp_path)


def test_lesson_must_display_a_figure(checker, tmp_path):
    """Numeric assertions alone do not meet the visualization lesson contract."""
    source = tmp_path / "no_figure.py"
    source.write_text("assert 1 + 2 == 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not display a visualization"):
        checker.execute_lesson(source, tmp_path)


def test_timeout_removes_temporary_working_directory(checker, monkeypatch):
    """A stopped child cannot leave its exported lesson files in the checkout."""
    created = []
    original_temporary_directory = checker.tempfile.TemporaryDirectory

    def record_directory(**kwargs):
        temporary = original_temporary_directory(**kwargs)
        created.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(checker.tempfile, "TemporaryDirectory", record_directory)
    with pytest.raises(RuntimeError, match="exceeded"):
        checker.check_lesson("reduction", timeout=0.000001)
    assert len(created) == 1
    assert not created[0].exists()


@pytest.mark.parametrize("value", ["0", "-2", "nan", "inf", "not-a-number"])
def test_invalid_timeout_is_a_usage_error(checker, value, capsys):
    """Invalid time limits are rejected before an expensive worker can start."""
    with pytest.raises(SystemExit) as caught:
        checker.main(["--timeout", value])
    assert caught.value.code == 2
    assert "timeout must" in capsys.readouterr().err


def test_optimized_python_cannot_skip_lesson_assertions(checker, tmp_path, monkeypatch):
    """A worker must not report success when Python has disabled assert statements."""
    source = tmp_path / "assertions.py"
    source.write_text("assert False\n", encoding="utf-8")
    monkeypatch.setattr(sys, "flags", type("Flags", (), {"optimize": 1})())
    with pytest.raises(RuntimeError, match="without -O"):
        checker.execute_lesson(source, tmp_path)


def test_copied_runner_uses_installed_version_without_source(checker, tmp_path, monkeypatch):
    """The installed-package validation tree needs no copied source package for its CLI."""
    monkeypatch.setattr(checker, "ROOT", tmp_path)

    def installed_version(name):
        assert name == "rainbow-tensor"
        return "9.8.7"

    monkeypatch.setattr(checker, "version", installed_version)
    assert checker._version() == "rainbow-tensor lesson checker 9.8.7"
