"""Distribution validation copies lesson fixtures without exposing package source."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def distribution_checker():
    """Load the contributor packaging checks without installing a distribution."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_distribution.py"
    spec = importlib.util.spec_from_file_location("distribution_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(path):
    """Create a small sdist-shaped tree with a package that must not be copied."""
    files = {
        "tests/test_golden.py": "assert True\n",
        "tests/golden/example.svg": "<svg />\n",
        "scripts/check_distribution.py": "# distribution validator\n",
        "scripts/check_lessons.py": "# lesson validator\n",
        "examples/lessons/example.py": "assert 1 + 2 == 3\n",
        "src/rainbow_tensor/__init__.py": "raise AssertionError('unpacked package imported')\n",
    }
    for name, text in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return path


def test_validation_copy_contains_checks_but_no_package_source(distribution_checker, tmp_path):
    """The installed wheel remains the only available package in a copied validation tree."""
    source = make_source(tmp_path / "source")
    destination = distribution_checker._copy_validation_files(source, tmp_path / "validation")
    assert not (destination / "src").exists()
    assert not (destination / "rainbow_tensor").exists()
    for name in (
        "tests/test_golden.py", "tests/golden/example.svg",
        "scripts/check_distribution.py", "scripts/check_lessons.py",
        "examples/lessons/example.py",
    ):
        assert (destination / name).read_bytes() == (source / name).read_bytes()


@pytest.mark.parametrize(
    "missing",
    ["scripts/check_lessons.py", "examples/lessons/example.py"],
)
def test_missing_packaged_lesson_files_fail_validation(
    distribution_checker, tmp_path, missing,
):
    """Missing lesson infrastructure must fail release validation before installation."""
    checkout = make_source(tmp_path / "checkout")
    source = make_source(tmp_path / "source")
    (source / missing).unlink()
    with pytest.raises(ValueError, match="missing validation files") as caught:
        distribution_checker._check_source_files(source, checkout)
    assert str(Path(missing)) in str(caught.value)
