"""Verify standalone lesson replay in offline Chromium without a notebook kernel.

Install scripts/requirements-browser.txt and run playwright install chromium.
The checks open saved HTML files with networking disabled, including imported
hostile text, adaptive SVG themes, prediction reveal, and keyboard navigation.
"""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "rainbow_tensor").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from playwright.sync_api import expect, sync_playwright  # noqa: E402

import rainbow_tensor as rt  # noqa: E402
from rainbow_tensor.lessons import capture_lesson  # noqa: E402
from rainbow_tensor.visual import TensorVisual  # noqa: E402


def _screenshot(page, destination):
    """Save an explicitly requested screenshot without leaving default artifacts."""
    if destination is not None:
        page.screenshot(path=str(destination), full_page=True)


def check_navigation(context, directory, screenshot=None):
    """Replay only authored positions and retain exact large coordinates offline."""
    size = 2**54 + 5
    recording = capture_lesson([
        rt.sum(np.arange(6).reshape(2, 3), axis=1, focus=(1,)),
        rt.reshape((size,), (size,), focus=(size - 1,)),
        rt.reshape((), (), focus=()),
        rt.reshape((0,), (0,)),
    ])
    destination = directory / "navigation.html"
    recording.save(destination)
    page = context.new_page()
    try:
        page.goto(destination.as_uri())
        expect(page.locator("#state-number")).to_have_text("Captured state 1 / 4")
        expect(page.locator("#previous")).to_be_disabled()
        page.locator("#next").click()
        expect(page.locator("#focus")).to_have_text(f"Focused output: ({size - 1},)")
        page.locator("#lesson-player").focus()
        page.keyboard.press("ArrowRight")
        expect(page.locator("#focus")).to_have_text("Focused output: ()")
        page.keyboard.press("End")
        expect(page.locator("#focus")).to_have_text("Focused output: No focused element")
        expect(page.locator("#next")).to_be_disabled()
        page.keyboard.press("Home")
        expect(page.locator("#state-number")).to_have_text("Captured state 1 / 4")
        assert page.locator("#result-image").evaluate(
            "image => image.complete && image.naturalWidth > 0",
        )
    finally:
        _screenshot(page, screenshot)
        page.close()


def check_prediction(context, directory, screenshot=None):
    """Hide captured answers until reveal without changing the author's saved states."""
    playground = rt.reduction_playground(rt.sum, np.arange(6).reshape(2, 3), axis=1)
    try:
        playground.prediction.value = "(3,)"
        recording = capture_lesson([playground])
    finally:
        playground.close()
    destination = directory / "prediction.html"
    recording.save(destination)
    page = context.new_page()
    try:
        page.goto(destination.as_uri())
        expect(page.locator("#prediction-answer")).to_have_text("(3,)")
        expect(page.locator("#result")).to_be_hidden()
        expect(page.locator("#focus")).to_be_hidden()
        page.get_by_role("button", name="Reveal captured result").click()
        expect(page.locator("#result")).to_be_visible()
        expect(page.locator("#reveal")).to_have_attribute("aria-expanded", "true")
        page.get_by_role("button", name="Hide captured result").click()
        expect(page.locator("#result")).to_be_hidden()
        assert not recording.to_dict()["states"][0]["prediction"]["revealed"]
    finally:
        _screenshot(page, screenshot)
        page.close()


def check_untrusted_content(context, directory, screenshot=None):
    """Imported SVG scripts, event handlers, and external image loads stay inert."""
    attack = '</script><img src="https://invalid.example/pixel" onerror="window.injected=1">'
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" '
        'onload="parent.injected=1"><script>parent.injected=1</script>'
        '<image href="https://invalid.example/remote.svg" width="40" height="40"/>'
        '<rect width="40" height="40" fill="red"/></svg>'
    )
    recording = capture_lesson([TensorVisual(svg, (), explanation=[attack])], title=attack)
    destination = directory / "untrusted.html"
    recording.save(destination)
    page = context.new_page()
    errors = []
    requests = []
    page.on("pageerror", lambda error: errors.append(error))
    page.on("request", lambda request: requests.append(request.url))
    try:
        page.goto(destination.as_uri())
        expect(page.locator("#explanation")).to_have_text(attack)
        expect(page.locator("h1")).to_have_text(attack)
        page.wait_for_function("document.getElementById('result-image').complete")
        assert page.evaluate("window.injected === undefined")
        assert not any(url.startswith(("http:", "https:")) for url in requests), requests
        assert not errors, errors
        expect(page.locator("svg")).to_have_count(0)
    finally:
        _screenshot(page, screenshot)
        page.close()


def check_adaptive_theme(context, directory, screenshot=None):
    """Automatic SVG colours still follow the browser scheme inside inert images."""
    recording = capture_lesson([rt.shape(np.arange(6).reshape(2, 3), theme="auto")])
    destination = directory / "theme.html"
    recording.save(destination)
    page = context.new_page()
    try:
        page.emulate_media(color_scheme="light")
        page.goto(destination.as_uri())
        page.wait_for_function("document.getElementById('result-image').naturalWidth > 0")
        light = page.locator("#result-image").screenshot()
        page.emulate_media(color_scheme="dark")
        dark = page.locator("#result-image").screenshot()
        assert light != dark, "adaptive SVG did not change with the browser colour scheme"
        assert page.locator("body").evaluate("body => getComputedStyle(body).backgroundColor") == (
            "rgb(17, 22, 33)"
        )
    finally:
        _screenshot(page, screenshot)
        page.close()


def main():
    """Run offline browser checks and release every browser and captured controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"rainbow-tensor {rt.__version__}")
    parser.add_argument(
        "--screenshot", type=Path,
        help="Save one PNG per check using this path as the filename prefix",
    )
    args = parser.parse_args()
    if args.screenshot:
        if args.screenshot.suffix.lower() != ".png":
            parser.error("--screenshot must end in .png")
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    rt.set_language("en")
    with TemporaryDirectory(prefix="rainbow-offline-lessons-") as temporary:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(offline=True)
            try:
                for check in (
                    check_navigation, check_prediction,
                    check_untrusted_content, check_adaptive_theme,
                ):
                    screenshot = None if args.screenshot is None else args.screenshot.with_name(
                        args.screenshot.stem + "-" + check.__name__.removeprefix("check_") + ".png",
                    )
                    check(context, Path(temporary), screenshot)
                    print(f"PASS {check.__name__}", flush=True)
            finally:
                context.close()
                browser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
