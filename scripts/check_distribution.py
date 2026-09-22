"""Validate built distributions using an isolated, non-editable installation.

Run after ``python -m build``. The wheel is checked outside the checkout, then
replaced by an installation built from the extracted sdist. The sdist's own
tests run against that installed package, including its shipped SVG fixtures.
Dependencies are installed into a temporary virtual environment.
"""

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
import venv
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

SMOKE = """
import importlib.metadata
from importlib.resources import files
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import rainbow_tensor as rt

package_path = Path(rt.__file__).resolve()
assert package_path.is_relative_to(Path(sys.prefix).resolve()), package_path
assert importlib.metadata.version("rainbow-tensor") == sys.argv[1]
assert rt.__version__ == sys.argv[1]
assert "ipywidgets" not in sys.modules
assert "anywidget" not in sys.modules
assert "export default { render }" in files("rainbow_tensor").joinpath(
    "_widgets/explorer.js"
).read_text(encoding="utf-8")
assert rt.get_language() == "auto"
assert rt.get_default_theme().name == "auto"
assert {"en", "zh"} <= set(rt.available_languages())
rt.set_language("zh-CN")
assert rt.get_resolved_language() == "zh"
assert 'lang="zh"' in rt.shape((2, 0)).svg
assert rt.shape((2, 0)).svg != rt.shape((2, 0), theme="light").svg
rt.set_language("en")
array = np.arange(6).reshape(2, 3)
cube = np.arange(24).reshape(2, 3, 4)
visuals = [
    rt.shape(array),
    rt.index(array, (slice(None), 1)),
    rt.sum(array, axis=1),
    rt.matmul(array, np.arange(6).reshape(3, 2)),
    rt.sum(array, keepdims=True),
    rt.mean(array, axis=(), keepdims=True),
    rt.sum(cube, axis=(0, 2), keepdims=True, focus=(0, 2, 0)),
]
assert visuals[0].shape == (2, 3)
assert [visual.result_shape for visual in visuals[1:]] == [
    (2,), (2,), (2, 2), (1, 1), (2, 3), (1, 3, 1),
]
assert visuals[-1].trace.output_coord == (0, 2, 0)
assert visuals[-1].trace.term_count == 8
indexed = rt.index(array, ([1, 0, 1], slice(None, None, -1)), focus=(2, 1))
assert indexed.trace.output_coord == (2, 1)
assert indexed.trace.terms[0][0].coordinate == (1, 1)
assert "Output (2, 1) reads source (1, 1)." in indexed.text
visuals.append(indexed)
flow = rt.Flow()
source = flow.input(array + 1, name="X")
sampled = flow.index(source, ([1, 0, 1], slice(None, None, -1)), name="A")
turned = flow.transpose(sampled, name="B")
result = flow.sum(turned, axis=1, name="Y")
chained = result.visualize(focus=(0,))
assert result.value((0,)) == 15
assert chained.provenance.complete
assert {root.reference.coordinate: root.count for root in chained.provenance.roots} == {
    (1, 2): 2, (0, 2): 1,
}
visuals.append(chained)
scalar = rt.sum(np.array(7))
empty = rt.sum(np.empty((2, 0, 3)), axis=2)
empty_sum = rt.sum(np.empty((2, 0, 3)), axis=1)
empty_mean = rt.mean(np.empty((2, 0, 3)), axis=1)
assert scalar.result_shape == scalar.trace.output_coord == ()
assert scalar.trace.terms[0][0].coordinate == ()
assert empty.result_shape == (2, 0) and empty.trace is None
assert "No elements" in empty.svg
assert empty.metadata["value_evaluation"]["total_terms"] == 0
assert empty_sum.trace.term_count == 0 and empty_sum.trace.divisor == 1
assert empty_mean.trace.term_count == empty_mean.trace.divisor == 0
assert "NaN" in empty_mean.text and "/ 0" not in empty_mean.text
visuals.extend([rt.shape(()), rt.shape((0,)), scalar, empty, empty_sum, empty_mean])
for visual in visuals:
    assert ElementTree.fromstring(visual.svg).tag.endswith("svg")
    assert visual.mime_type == "image/svg+xml"
assert visuals[-1].text
path = Path("smoke.svg")
visuals[-1].save(path)
assert path.read_text(encoding="utf-8") == visuals[-1].svg
flow_explorer = rt.explore(result)
try:
    assert "data-rt-coordinate" in flow_explorer.figure.value
    assert "data-rt-coordinate" not in flow_explorer.visual.svg
    flow_explorer.figure._handle_custom_msg({
        "type": "focus", "revision": flow_explorer.figure.revision, "coordinate": "[2]",
    }, [])
    assert flow_explorer.focus == (2,)
    roots = flow_explorer.visual.provenance.roots
    assert {root.reference.coordinate: root.count for root in roots} == {(1, 0): 2, (0, 0): 1}
finally:
    flow_explorer.close()
explorer = rt.explore(rt.mean, cube, axis=(-1, 0), keepdims=True)
try:
    assert tuple(control.max for control in explorer.coordinates) == (0, 2, 0)
    explorer.coordinates[1].value = 2
    explorer.update_button.click()
    visual = explorer.visual
    assert explorer.focus == (0, 2, 0)
    assert visual.trace.output_coord == (0, 2, 0)
    assert visual.svg == rt.mean(
        cube, axis=(-1, 0), keepdims=True, focus=(0, 2, 0),
    ).svg
finally:
    explorer.close()
indexed_explorer = rt.explore(rt.index, array, ([1, 0, 1], slice(None, None, -1)))
try:
    assert indexed_explorer.focus == (0, 0)
    indexed_explorer.coordinates[0].value = 2
    indexed_explorer.update_button.click()
    assert indexed_explorer.focus == (2, 0)
    assert indexed_explorer.visual.trace.terms[0][0].coordinate == (1, 2)
    assert indexed_explorer.visual.svg == rt.index(
        array, ([1, 0, 1], slice(None, None, -1)), focus=(2, 0),
    ).svg
finally:
    indexed_explorer.close()
empty_explorer = rt.explore(rt.sum, (2, 0, 3), axis=2)
try:
    assert empty_explorer.focus is None
    assert empty_explorer.coordinates == ()
    assert empty_explorer.update_button.disabled
    assert empty_explorer.visual.trace is None
finally:
    empty_explorer.close()
print(f"Installed package smoke passed: {package_path}")
"""


def _run(command, *, cwd, env):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _one_artifact(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {directory}, found {len(matches)}")
    return matches[0].resolve()


def _extract_source(archive_path, destination):
    """Extract regular source files without allowing paths outside destination."""
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported sdist member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not (roots[0] / "pyproject.toml").is_file():
        raise ValueError("The sdist must contain one project with pyproject.toml")
    return roots[0]


def _check_source_files(source, checkout):
    expected = {
        path.relative_to(checkout)
        for path in (checkout / "tests").rglob("*")
        if path.is_file() and path.suffix in {".py", ".svg"}
    }
    required = {Path("tests/test_golden.py"), Path("scripts/check_distribution.py")}
    missing = sorted(str(path) for path in expected | required if not (source / path).is_file())
    if missing:
        raise ValueError("The sdist is missing validation files: " + ", ".join(missing))
    fixtures = list((source / "tests/golden").glob("*.svg"))
    if not fixtures:
        raise ValueError("The sdist contains no golden SVG fixtures")
    print(f"Sdist includes {len(fixtures)} golden SVG fixtures and all checkout test files")


def check_distribution(directory):
    """Check static and interactive rendering from a wheel and matching sdist."""
    wheel = _one_artifact(directory, "rainbow_tensor-*.whl")
    sdist = _one_artifact(directory, "rainbow_tensor-*.tar.gz")
    with ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError("The wheel must contain exactly one package metadata file")
        wheel_metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    version = wheel_metadata["Version"]
    if wheel_metadata["Name"] != "rainbow-tensor" or not version:
        raise ValueError("Unexpected wheel package metadata")

    env = os.environ.copy()
    for key in (
        "PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
        "RAINBOW_TEST_BACKEND", "RT_UPDATE_GOLDEN",
    ):
        env.pop(key, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"

    with tempfile.TemporaryDirectory(prefix="rainbow-distribution-") as temporary:
        workspace = Path(temporary).resolve()
        extracted = workspace / "source"
        extracted.mkdir()
        source = _extract_source(sdist, extracted)
        _check_source_files(source, Path(__file__).resolve().parents[1])
        source_metadata = BytesParser().parsebytes((source / "PKG-INFO").read_bytes())
        if (source_metadata["Name"], source_metadata["Version"]) != ("rainbow-tensor", version):
            raise ValueError("Wheel and sdist package metadata do not match")

        environment = workspace / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        working = workspace / "working"
        working.mkdir()
        pip = [str(python), "-I", "-m", "pip"]
        _run([*pip, "install", str(wheel), "pytest>=7"], cwd=working, env=env)
        smoke_path = working / "smoke.py"
        smoke_path.write_text(SMOKE, encoding="utf-8")
        smoke = [str(python), "-I", "-B", str(smoke_path), version]
        _run(smoke, cwd=working, env=env)
        _run(
            [*pip, "install", "--no-deps", "--force-reinstall", str(source)],
            cwd=working,
            env=env,
        )
        _run(smoke, cwd=working, env=env)
        _run(
            [
                str(python), "-I", "-B", "-m", "pytest", str(source / "tests"),
                "-q", "--import-mode=importlib", "-p", "no:cacheprovider",
            ],
            cwd=working,
            env=env,
        )
    print(f"Distribution validation passed for rainbow-tensor {version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir", type=Path, default=Path("dist"),
        help="directory containing exactly one wheel and one sdist (default: dist)",
    )
    args = parser.parse_args()
    try:
        check_distribution(args.dist_dir)
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as error:
        parser.exit(1, f"Distribution validation failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Distribution validation interrupted\n")


if __name__ == "__main__":
    main()
