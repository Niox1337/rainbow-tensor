"""Check the documented lessons against this checkout in fresh Python processes.

Only the named, checked-in lesson sources are executed. Every lesson receives a
temporary working directory, a timeout, and deterministic display settings.
This is a contributor validation tool, not a sandbox for untrusted Python.
"""

import argparse
import ast
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSON_DIRECTORY = ROOT / "examples" / "lessons"
LESSONS = {
    "reduction": "reduction.py",
    "repeated-index": "repeated_index.py",
    "broadcast": "broadcast.py",
    "operation-origins": "operation_origins.py",
    "scalars-and-empties": "scalars_and_empties.py",
    "bounded-work": "bounded_work.py",
    "elementwise": "elementwise.py",
    "guided-terms": "guided_terms.py",
    "row-normalization": "row_normalization.py",
    "reduction-axes": "reduction_axes.py",
}


def lesson_source(name):
    """Resolve a registered source without accepting arbitrary paths from the CLI."""
    try:
        path = (LESSON_DIRECTORY / LESSONS[name]).resolve()
    except KeyError as error:
        raise ValueError(f"Unknown lesson {name!r}") from error
    if not path.is_relative_to(LESSON_DIRECTORY.resolve()) or not path.is_file():
        raise ValueError(f"Lesson source is missing or outside its directory: {name}")
    return path


def execute_lesson(source, output_directory):
    """Execute trusted lesson code, validate displayed SVGs, and close its controls.

    The original filename is retained in compiled code so an assertion failure
    points directly to the lesson line shown in the documentation. Display is
    captured without a browser. Browser event handling has separate tests.
    """
    import IPython.display

    import rainbow_tensor as rt
    from rainbow_tensor.playground import ReductionPlayground
    from rainbow_tensor.walkthrough import Walkthrough

    source = Path(source).resolve()
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    assertions = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    if not assertions:
        raise ValueError(f"Lesson has no executable assertions: {source}")
    if sys.flags.optimize:
        raise RuntimeError("Lesson assertions require Python without -O or -OO")

    output_directory = Path(output_directory).resolve()
    namespace = {"__name__": "__lesson__", "__file__": str(source)}
    displayed = []
    controls = []
    original_display = IPython.display.display
    original_language = rt.get_language()
    original_theme = rt.get_default_theme()

    def capture(*objects, **kwargs):
        """Save each displayed view before a later focus change replaces it."""
        for item in objects:
            visual = item if isinstance(item, rt.TensorVisual) else getattr(item, "visual", None)
            if not isinstance(visual, rt.TensorVisual):
                raise TypeError("A verified lesson must display tensor visuals or explorers")
            if callable(getattr(item, "close", None)):
                controls.append(item)
            if ET.fromstring(visual.svg).tag != "{http://www.w3.org/2000/svg}svg":
                raise ValueError("A lesson displayed content that is not an SVG figure")
            number = len(displayed) + 1
            svg_path = output_directory / f"figure-{number:02d}.svg"
            text_path = output_directory / f"figure-{number:02d}.txt"
            visual.save(svg_path)
            text_path.write_text(visual.text, encoding="utf-8")
            if svg_path.read_text(encoding="utf-8") != visual.svg:
                raise AssertionError("Saved SVG differs from the displayed figure")
            if text_path.read_text(encoding="utf-8") != visual.text:
                raise AssertionError("Saved explanation differs from the displayed text")
            displayed.append(svg_path.name)

    try:
        rt.set_language("en")
        rt.set_default_theme("light")
        IPython.display.display = capture
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree, str(source), "exec"), namespace)
        if not displayed:
            raise ValueError(f"Lesson did not display a visualization: {source}")
        return {"figures": len(displayed), "assertions": assertions}
    finally:
        # Namespace discovery also closes an explorer created before a failed assert.
        controls.extend(
            item for item in namespace.values()
            if isinstance(item, (rt.FocusExplorer, Walkthrough, ReductionPlayground))
        )
        closed = set()
        for control in controls:
            if id(control) not in closed:
                control.close()
                closed.add(id(control))
        IPython.display.display = original_display
        rt.set_language(original_language)
        rt.set_default_theme(original_theme)


def check_lesson(name, *, timeout=30):
    """Run one named lesson in a temporary directory, including failure cleanup."""
    lesson_source(name)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("The lesson timeout must be a positive finite number")
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONOPTIMIZE"):
        environment.pop(key, None)
    environment["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory(prefix=f"rainbow-lesson-{name}-") as directory:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker", name],
                cwd=directory,
                env=environment,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"{name}: exceeded {timeout:g} seconds ({lesson_source(name)})"
            ) from error
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"{name}: lesson failed\n{detail}")
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{name}: worker returned an invalid validation report") from error
    return report


def _positive_seconds(value):
    """Reject invalid timeouts before launching any lesson process."""
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a number") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return seconds


def _version():
    """Report the checkout version, or the installed version in copied validation files."""
    initialization = ROOT / "src" / "rainbow_tensor" / "__init__.py"
    if not initialization.is_file():
        return f"rainbow-tensor lesson checker {version('rainbow-tensor')}"
    tree = ast.parse(initialization.read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            return f"rainbow-tensor lesson checker {ast.literal_eval(node.value)}"
    raise ValueError("The checkout does not declare its package version")


def main(argv=None):
    """Run all lessons by default or select registered names for a focused check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lessons", nargs="*", metavar="LESSON", help="named lessons, default: all")
    parser.add_argument("--list", action="store_true", help="list the registered lesson names")
    parser.add_argument(
        "--timeout", type=_positive_seconds, default=30,
        help="maximum seconds per lesson, default: 30",
    )
    parser.add_argument("--version", action="version", version=_version())
    parser.add_argument("--worker", choices=LESSONS, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.worker:
        if (ROOT / "src" / "rainbow_tensor" / "__init__.py").is_file():
            sys.path.insert(0, str(ROOT / "src"))
        report = execute_lesson(lesson_source(options.worker), Path.cwd())
        print(json.dumps(report))
        return 0
    if options.list:
        print("\n".join(LESSONS))
        return 0
    names = options.lessons or list(LESSONS)
    unknown = [name for name in names if name not in LESSONS]
    if unknown:
        parser.error(f"unknown lesson {unknown[0]!r}, use --list to see available lessons")
    failed = []
    for name in names:
        try:
            report = check_lesson(name, timeout=options.timeout)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"FAIL {error}", file=sys.stderr, flush=True)
            failed.append(name)
        else:
            print(f"PASS {name}: {report['figures']} figures", flush=True)
    if failed:
        print(f"{len(failed)} of {len(names)} lessons failed", file=sys.stderr)
        return 1
    print(f"All {len(names)} lessons passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Lesson validation interrupted", file=sys.stderr)
        raise SystemExit(130) from None
