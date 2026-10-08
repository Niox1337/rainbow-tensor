"""Exercise the package in a real JupyterLab kernel and headless Chromium.

Install scripts/requirements-browser.txt and run playwright install chromium.
The check creates an isolated notebook, authenticated localhost server, and
temporary kernel specification using this Python interpreter. Browser actions
cross the actual Jupyter widget communication channel. No mock model is used.
"""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from threading import Event
from urllib.error import URLError
from urllib.request import Request, urlopen

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''import numpy as np
import rainbow_tensor as rt
from IPython.display import display

rt.set_language("en")
assert hasattr(rt, "walkthrough") and hasattr(rt, "reduction_playground")
views = []

def show(name, view):
    view.widget.add_class("rt-smoke-" + name)
    views.append(view)
    display(view)

x = np.arange(1, 7).reshape(2, 3)
selection = ([1, 0, 1], slice(None, None, -1))
index_explorer = rt.explore(rt.index, x, selection)
show("index", index_explorer)

flow = rt.Flow()
source = flow.input(x, name="X")
sampled = flow.index(source, selection, name="S")
result = flow.sum(flow.transpose(sampled, name="T"), axis=1, name="Y")
flow_explorer = rt.explore(result)
show("flow", flow_explorer)

scalar_explorer = rt.explore(rt.sum, np.array([3, 4]))
show("scalar", scalar_explorer)
empty_explorer = rt.explore(rt.sum, np.empty((2, 0, 3)), axis=2)
show("empty", empty_explorer)

lesson = rt.walkthrough(result)
show("lesson", lesson)
condition = flow.greater(source, flow.input(4, name="Threshold"), name="Above")
filtered = flow.where(condition, source, flow.input(0, name="Zero"), name="Filtered")
selection_lesson = rt.walkthrough(filtered, focus=(1, 2))
show("selection", selection_lesson)
winner = flow.argmax(flow.input(np.array([3., 9., 9.]), name="Candidates"), name="Winner")
extrema_lesson = rt.walkthrough(winner)
show("extrema", extrema_lesson)
playground = rt.reduction_playground(rt.mean, np.arange(24).reshape(2, 3, 4), axis=1)
show("playground", playground)
print("RAINBOW_WIDGETS_READY")
'''
VERIFY = '''assert index_explorer.focus == (2, 1)
assert index_explorer.figure.revision >= 4
assert index_explorer.visual.trace.terms[0][0].coordinate == (1, 1)
assert flow_explorer.focus == (2,)
counts = {root.reference.coordinate: root.count
          for root in flow_explorer.visual.provenance.roots}
assert counts == {(1, 0): 2, (0, 0): 1}
assert result.value((2,)) == 9
assert scalar_explorer.focus == ()
assert empty_explorer.focus is None
assert empty_explorer.update_button.disabled
assert lesson.snapshot.occurrence == 1
assert lesson.snapshot.step.reference.coordinate == (0, 0)
assert lesson.snapshot.output_value == 6
assert lesson.calculation_paths.value == 1
assert selection_lesson.focus == (0, 0)
assert selection_lesson.snapshot.output_value == 0
assert selection_lesson.snapshot.selected_source.reason == "condition_false"
assert extrema_lesson.snapshot.term == 2
assert extrema_lesson.snapshot.term_value == 9
assert extrema_lesson.snapshot.output_value == 1
assert extrema_lesson.snapshot.selected_source.position == 1
assert extrema_lesson.snapshot.subtotal is None
assert playground.axis == (0, 2)
assert playground.keepdims is True
assert playground.explorer.result_shape == (1, 3, 1)
assert playground.revealed
assert playground.explorer.visual.trace.divisor == 8
assert playground.explorer.focus == (0, 1, 0)
assert playground.feedback.code == "matches"
print("RAINBOW_JUPYTER_CHECKS_PASSED")
'''
CLEANUP = '''for view in views:
    view.close()
print("RAINBOW_WIDGETS_CLOSED")
'''


def write_environment(directory, token, port):
    """Create a temporary notebook, kernel spec, and server-only authentication."""
    data = directory / "data"
    config = directory / "config"
    runtime = directory / "runtime"
    notebooks = directory / "notebooks"
    kernel = data / "kernels" / "rainbow-smoke"
    settings = directory / "settings"
    notebook_settings = settings / "@jupyterlab" / "notebook-extension"
    for path in (config, runtime, notebooks, kernel, notebook_settings):
        path.mkdir(parents=True)
    (notebook_settings / "tracker.jupyterlab-settings").write_text(
        json.dumps({"windowingMode": "none"}), encoding="utf-8",
    )
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Rainbow smoke", "language": "python",
        "env": {"PYTHONPATH": str(ROOT / "src")},
    }), encoding="utf-8")
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {
            "display_name": "Rainbow smoke", "language": "python", "name": "rainbow-smoke",
        }},
        "cells": [{
            "cell_type": "code", "id": f"smoke-{index}", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source,
        } for index, source in enumerate((SETUP, VERIFY, CLEANUP))],
    }
    (notebooks / "smoke.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    (config / "jupyter_server_config.json").write_text(json.dumps({
        "IdentityProvider": {"token": token},
        "ServerApp": {
            "ip": "127.0.0.1", "port": port, "port_retries": 0,
            "open_browser": False, "root_dir": str(notebooks), "allow_remote_access": False,
        },
    }), encoding="utf-8")
    return {
        **os.environ, "JUPYTER_CONFIG_DIR": str(config), "JUPYTER_DATA_DIR": str(data),
        "JUPYTER_RUNTIME_DIR": str(runtime), "JUPYTER_PATH": str(data),
        "JUPYTERLAB_SETTINGS_DIR": str(settings),
    }


def request(base, token, path, method="GET"):
    """Call only the owned localhost server without exposing its token in URLs."""
    message = Request(base + path, headers={"Authorization": "token " + token}, method=method)
    with urlopen(message, timeout=2) as response:
        body = response.read()
        return json.loads(body) if body else None


def wait_ready(process, base, token):
    """Wait for authenticated server readiness and stop promptly on startup failure."""
    deadline = time.monotonic() + 90
    retry = Event()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("the temporary Jupyter server exited during startup")
        try:
            request(base, token, "/api/status")
            return
        except (URLError, TimeoutError, OSError):
            retry.wait(0.1)
    raise TimeoutError("the temporary Jupyter server did not become ready")


def stop_server(process, base, token):
    """Shut down the owned kernel and server, forcibly reaping only this process tree if needed."""
    if process.poll() is not None:
        return
    try:
        for kernel in request(base, token, "/api/kernels"):
            request(base, token, "/api/kernels/" + kernel["id"], "DELETE")
        request(base, token, "/api/shutdown", "POST")
        process.wait(timeout=15)
        return
    except (URLError, TimeoutError, OSError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, check=False,
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def cell(container, coordinate):
    """Locate a rendered result cell using its exact serialized coordinate."""
    payload = json.dumps(coordinate, separators=(",", ":"))
    return container.locator(f'[data-rt-coordinate=\'{payload}\']')


def run_cell(page, index, marker):
    """Execute a saved notebook cell through the editor and wait for its actual output."""
    notebook_cell = page.locator(".jp-Notebook .jp-CodeCell").nth(index)
    notebook_cell.locator(".cm-content").click()
    page.get_by_role("button", name="Run this cell and advance (Shift+Enter)", exact=True).click()
    page.wait_for_function("""({index, marker}) => {
      const cell = document.querySelectorAll('.jp-Notebook .jp-CodeCell')[index]
      const text = cell?.querySelector('.jp-OutputArea')?.textContent || ''
      return text.includes(marker) || text.includes('Traceback')
    }""", arg={"index": index, "marker": marker}, timeout=90_000)
    output = notebook_cell.locator(".jp-OutputArea").inner_text()
    if marker not in output:
        raise RuntimeError(f"Notebook cell {index} failed:\n{output[-6000:]}")


def check_notebook(page, base, token):
    """Exercise native widget controls and assert the resulting kernel state through the UI."""
    page.goto(base + "/lab/tree/smoke.ipynb", wait_until="domcontentloaded")
    page.locator("#password_input").fill(token)
    page.locator("#password_input").press("Enter")
    expect(page.locator(".jp-Notebook .jp-CodeCell")).to_have_count(3, timeout=90_000)
    expect(page.get_by_role("button", name="Rainbow smoke | Idle", exact=True)).to_be_visible(
        timeout=90_000,
    )
    print("JupyterLab kernel ready.", flush=True)
    run_cell(page, 0, "RAINBOW_WIDGETS_READY")
    print("Notebook widgets rendered.", flush=True)
    dismiss_news = page.get_by_role("button", name="No", exact=True)
    if dismiss_news.is_visible():
        dismiss_news.click()

    index = page.locator(".rt-smoke-index")
    cell(index, (2, 0)).click()
    expect(cell(index, (2, 0))).to_have_attribute("aria-pressed", "true")
    page.keyboard.press("ArrowRight")
    expect(cell(index, (2, 1))).to_have_attribute("aria-pressed", "true")
    expect(cell(index, (2, 1))).to_be_focused()
    page.keyboard.press("Enter")
    expect(index.locator('[aria-live="polite"]')).to_contain_text("(2, 1)")

    flow = page.locator(".rt-smoke-flow")
    cell(flow, (2,)).click()
    expect(cell(flow, (2,))).to_have_attribute("aria-pressed", "true")
    expect(flow).to_contain_text("X[1, 0]")
    scalar = page.locator(".rt-smoke-scalar")
    cell(scalar, ()).click()
    expect(cell(scalar, ())).to_have_attribute("aria-pressed", "true")
    empty = page.locator(".rt-smoke-empty")
    expect(empty.locator("[data-rt-coordinate]")).to_have_count(0)
    expect(empty.get_by_role("button", name="Update focus")).to_be_disabled()

    lesson = page.locator(".rt-smoke-lesson")
    lesson.get_by_role("button", name="Next term", exact=True).click()
    expect(lesson).to_contain_text("Term 2 of 3")
    expect(lesson).to_contain_text("Running numerator through this term: 9")
    lesson.get_by_role("combobox", name="Occurrence", exact=True).select_option(index=1)
    expect(lesson).to_contain_text("Occurrence 1:")
    expect(lesson).to_contain_text("T[0, 0] = 6")
    paths = lesson.get_by_role("listbox", name="Calculation paths", exact=True)
    paths.select_option(index=0)
    expect(lesson).to_contain_text("Occurrence 0:")
    paths.focus()
    expect(paths).to_be_focused()
    paths.press("ArrowDown")
    expect(paths).to_have_value("1")
    expect(lesson).to_contain_text("Occurrence 1:")
    expect(lesson).to_contain_text("Y[0] ← summand 1 ← T[0, 0]")
    paths.press("ArrowDown")
    expect(lesson).to_contain_text("Occurrence 2:")
    paths.press("ArrowUp")
    expect(lesson).to_contain_text("Occurrence 1:")

    selection = page.locator(".rt-smoke-selection")
    expect(selection).to_contain_text("the condition is true")
    selection_paths = selection.get_by_role("listbox", name="Calculation paths", exact=True)
    selection_paths.select_option(index=1)
    expect(selection).to_contain_text("(greater)")
    expect(selection).to_contain_text("condition ← Above[1, 2]")
    selection_paths.select_option(index=0)
    expect(selection).to_contain_text("Occurrence 0:")
    cell(selection, (0, 0)).click()
    expect(selection).to_contain_text("the condition is false")
    expect(selection).to_contain_text("candidate dependencies")

    extrema = page.locator(".rt-smoke-extrema")
    expect(extrema).to_contain_text("the first maximum wins")
    extrema.get_by_role("button", name="Next term", exact=True).click()
    expect(extrema).to_contain_text("candidate 2")
    extrema.get_by_role("button", name="Next term", exact=True).click()
    expect(extrema).to_contain_text("candidate 3")
    expect(extrema).to_contain_text("Winner[()] selects Candidates[1]")

    playground = page.locator(".rt-smoke-playground")
    expect(playground.locator(".rt-focus-explorer")).to_have_count(0)
    prediction = playground.get_by_role("textbox", name="Your prediction", exact=True)
    prediction.fill("(2, 9)")
    label_sizes = prediction.evaluate("""input => Array.from(input.labels, label => ({
        width: label.clientWidth, content: label.scrollWidth,
    }))""")
    assert label_sizes and all(
        label["width"] > 0 and label["width"] + 1 >= label["content"]
        for label in label_sizes
    ), f"Prediction label is clipped: {label_sizes}"
    playground.get_by_role("button", name="Reveal result", exact=True).click()
    expect(playground).to_contain_text("Compare output axis 1: predicted 9, expected 4")
    expect(playground).to_contain_text("Source axis 2 becomes output axis 1 with length 4")
    playground.get_by_role("button", name="Predict again", exact=True).click()
    prediction.fill("(2, 4)")
    playground.get_by_role("button", name="Reveal result", exact=True).click()
    expect(playground).to_contain_text("Your predicted shape matches.")
    expect(playground.locator(".rt-focus-explorer")).to_have_count(1)
    playground.locator("select").nth(0).select_option(label="Choose axes")
    playground.locator("select").nth(1).select_option(index=[0, 2])
    playground.get_by_role("checkbox", name="Keep dimensions", exact=True).check()
    playground.get_by_role("button", name="Apply parameters", exact=True).click()
    expect(playground.locator(".rt-focus-explorer")).to_have_count(0)
    expect(playground).to_contain_text("axis=(0, 2), keepdims=True")
    playground.get_by_role("textbox", name="Your prediction", exact=True).fill("(3,)")
    playground.get_by_role("button", name="Reveal result", exact=True).click()
    expect(playground).to_contain_text("Compare keepdims: reduced source axes (0, 2)")
    playground.get_by_role("button", name="Predict again", exact=True).click()
    playground.get_by_role("textbox", name="Your prediction", exact=True).fill("(1, 3, 1)")
    playground.get_by_role("button", name="Reveal result", exact=True).click()
    expect(playground).to_contain_text("Result shape: (1, 3, 1)")
    cell(playground, (0, 1, 0)).click()
    expect(cell(playground, (0, 1, 0))).to_have_attribute("aria-pressed", "true")
    run_cell(page, 1, "RAINBOW_JUPYTER_CHECKS_PASSED")
    print("Browser controls and kernel assertions passed.", flush=True)


def main():
    """Run an isolated live-host smoke check and always release owned resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path, help="Save the verified notebook view as PNG")
    args = parser.parse_args()
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    expect.set_options(timeout=30_000)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    token = secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory(prefix="rainbow-jupyter-") as temporary:
        directory = Path(temporary)
        environment = write_environment(directory, token, port)
        with (directory / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "jupyterlab", "--no-browser"],
                env=environment, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                wait_ready(process, base, token)
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    try:
                        page = browser.new_page(viewport={"width": 1440, "height": 1000})
                        page.set_default_timeout(30_000)
                        try:
                            check_notebook(page, base, token)
                            if args.screenshot:
                                page.screenshot(path=str(args.screenshot), full_page=True)
                                for name in ("lesson", "playground", "selection", "extrema"):
                                    page.locator(".rt-smoke-" + name).scroll_into_view_if_needed()
                                    page.screenshot(path=str(args.screenshot.with_name(
                                        args.screenshot.stem + "-" + name + ".png"
                                    )))
                            run_cell(page, 2, "RAINBOW_WIDGETS_CLOSED")
                        except Exception:
                            if args.screenshot:
                                page.screenshot(path=str(args.screenshot.with_suffix(".failure.png")))
                                args.screenshot.with_suffix(".failure.txt").write_text(
                                    page.locator("body").inner_text(), encoding="utf-8",
                                )
                            raise
                    finally:
                        browser.close()
            except Exception as error:
                raise SystemExit(str(error).replace(token, "[redacted]")) from None
            finally:
                stop_server(process, base, token)
    print("JupyterLab live-kernel checks passed: selection, keyboard, Flow, lessons, playground.")


if __name__ == "__main__":
    main()
