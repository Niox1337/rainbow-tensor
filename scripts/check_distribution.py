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
array = np.arange(6).reshape(2, 3)
visuals = [
    rt.shape(array),
    rt.index(array, (slice(None), 1)),
    rt.sum(array, axis=1),
    rt.matmul(array, np.arange(6).reshape(3, 2)),
]
assert visuals[0].shape == (2, 3)
assert [visual.result_shape for visual in visuals[1:]] == [(2,), (2,), (2, 2)]
for visual in visuals:
    assert ElementTree.fromstring(visual.svg).tag.endswith("svg")
    assert visual.mime_type == "image/svg+xml"
assert visuals[-1].text
path = Path("smoke.svg")
visuals[-1].save(path)
assert path.read_text(encoding="utf-8") == visuals[-1].svg
if sys.argv[2] == "interactive":
    explorer = rt.explore(rt.sum, array, axis=1)
    try:
        visual = explorer.set_focus((1,))
        assert explorer.focus == (1,)
        assert visual.trace.output_coord == (1,)
        assert visual.svg == rt.sum(array, axis=1, focus=(1,)).svg
    finally:
        explorer.close()
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


def check_distribution(directory, *, interactive=False):
    """Check one wheel and matching sdist without importing the working tree."""
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
        mode = "interactive" if interactive else "static"
        requirement = str(wheel) + ("[interactive]" if interactive else "")
        pip = [str(python), "-I", "-m", "pip"]
        _run([*pip, "install", requirement, "pytest>=7"], cwd=working, env=env)
        smoke_path = working / "smoke.py"
        smoke_path.write_text(SMOKE, encoding="utf-8")
        smoke = [str(python), "-I", "-B", str(smoke_path), version, mode]
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
    parser.add_argument(
        "--interactive", action="store_true",
        help="install the interactive extra and check widget updates and cleanup",
    )
    args = parser.parse_args()
    try:
        check_distribution(args.dist_dir, interactive=args.interactive)
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as error:
        parser.exit(1, f"Distribution validation failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Distribution validation interrupted\n")


if __name__ == "__main__":
    main()
