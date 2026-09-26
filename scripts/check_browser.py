"""Exercise the shipped widget JavaScript in Chromium with a live Python explorer.

Install scripts/requirements-browser.txt and run playwright install chromium.
The harness supplies the small anywidget model interface while browser events
cross into the real Python message handler. Jupyter host checks are separate.
"""

import json
import sys
from importlib.resources import files
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "rainbow_tensor").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from playwright.sync_api import expect, sync_playwright  # noqa: E402

import rainbow_tensor as rt  # noqa: E402

MOUNT = """async ({state, source}) => {
  window.latestRevision = state.revision
  const listeners = new Map()
  const model = {
    get: (key) => state[key],
    on: (event, callback) => listeners.set(event, callback),
    off: (event) => listeners.delete(event),
    send: async (message) => {
      state = await window.dispatchFocus(message)
      window.latestRevision = state.revision
      listeners.get('change:value')?.()
      listeners.get('change:description')?.()
    },
  }
  const url = URL.createObjectURL(new Blob([source], {type: 'text/javascript'}))
  const widget = (await import(url)).default
  URL.revokeObjectURL(url)
  window.disposeFigure = widget.render({model, el: document.getElementById('figure')})
  window.listenerCount = () => listeners.size
}"""


def state(explorer):
    """Transfer only the public synchronized figure traits into the browser harness."""
    return {
        key: getattr(explorer.figure, key) for key in ("value", "revision", "description", "label")
    }


def cell(page, coordinate):
    """Locate the exact decimal coordinate without JavaScript number conversion."""
    payload = json.dumps(coordinate, separators=(",", ":"))
    return page.locator(f"[data-rt-coordinate='{payload}']")


def mount(browser, explorer):
    """Mount the actual module and route its custom messages to an actual explorer."""
    page = browser.new_page()
    page.set_content(
        '<button id="before">Before</button><div id="figure"></div>'
        '<button id="after">After</button>'
    )

    def dispatch(message):
        """Perform the same Python callback used by the widget communication channel."""
        explorer.figure._handle_custom_msg(message, [])
        return state(explorer)

    page.expose_function("dispatchFocus", dispatch)
    source = files("rainbow_tensor").joinpath("_widgets/explorer.js").read_text("utf-8")
    page.evaluate(MOUNT, {"state": state(explorer), "source": source})
    return page


def selected(page, coordinate):
    """Wait for the complete Python redraw and verify its selected coordinate."""
    target = cell(page, coordinate)
    expect(target).to_have_attribute("aria-pressed", "true")
    expect(page.locator('[data-rt-coordinate][tabindex="0"]')).to_have_count(1)
    expect(target).to_be_focused()


def click_cell(page, coordinate):
    """Wait for a same-coordinate click to finish its Python round trip too."""
    revision = page.evaluate("window.latestRevision")
    cell(page, coordinate).click()
    page.wait_for_function("expected => window.latestRevision > expected", arg=revision)
    selected(page, coordinate)


def check_keyboard(browser):
    """Verify tab order, spatial navigation, activation, announcements, and disposal."""
    explorer = rt.explore(rt.transpose, np.arange(6).reshape(2, 3))
    page = mount(browser, explorer)
    try:
        page.locator("#before").focus()
        page.keyboard.press("Tab")
        expect(cell(page, (0, 0))).to_be_focused()
        page.keyboard.press("ArrowRight")
        selected(page, (0, 1))
        page.keyboard.press("ArrowDown")
        selected(page, (1, 1))
        assert explorer.focus == (1, 1)
        assert explorer.visual.trace.terms[0][0].coordinate == (1, 1)
        for key in ("Enter", "Space"):
            revision = explorer.figure.revision
            page.keyboard.press(key)
            page.wait_for_function(
                "() => document.querySelector('[aria-live]').textContent.length > 0"
            )
            expect(cell(page, (1, 1))).to_be_focused()
            # An identical selection still travels through Python and redraws.
            page.wait_for_function(
                "expected => window.latestRevision >= expected", arg=revision + 1
            )
        expect(page.locator('[aria-live="polite"]')).to_contain_text("(1, 1)")
        page.keyboard.press("Tab")
        expect(page.locator("#after")).to_be_focused()
        page.evaluate("window.disposeFigure()")
        assert page.evaluate("window.listenerCount()") == 0
        expect(page.locator("#figure")).to_be_empty()
    finally:
        page.close()
        explorer.close()


def check_occurrences(browser):
    """A browser click must retrace the repeated source paths through a recorded chain."""
    flow = rt.Flow()
    source = flow.input(np.arange(1, 7).reshape(2, 3), name="X")
    selected_input = flow.index(source, ([1, 0, 1], slice(None, None, -1)))
    result = flow.sum(flow.transpose(selected_input), axis=1)
    explorer = rt.explore(result)
    page = mount(browser, explorer)
    try:
        click_cell(page, (2,))
        assert explorer.focus == (2,)
        assert {
            root.reference.coordinate: root.count for root in explorer.visual.provenance.roots
        } == {
            (1, 0): 2,
            (0, 0): 1,
        }
        assert result.value((2,)) == 9
    finally:
        page.close()
        explorer.close()


def check_scalar_and_empty(browser):
    """Scalar results remain one button while empty results expose no false selection."""
    for shape, count in [((), 1), ((0,), 0)]:
        explorer = rt.explore(rt.reshape, shape, shape)
        page = mount(browser, explorer)
        try:
            expect(page.locator("[data-rt-coordinate]")).to_have_count(count)
            if count:
                click_cell(page, ())
            else:
                expect(page.locator('[aria-live="polite"]')).to_contain_text("empty")
        finally:
            page.close()
            explorer.close()


def check_large_coordinates(browser):
    """Arrow navigation preserves integers larger than JavaScript's exact Number range."""
    size = 2**54 + 5
    explorer = rt.explore(rt.reshape, (size,), (size,), focus=(size - 1,))
    page = mount(browser, explorer)
    try:
        click_cell(page, (size - 1,))
        page.keyboard.press("ArrowLeft")
        selected(page, (size - 2,))
        assert explorer.focus == (size - 2,)
    finally:
        page.close()
        explorer.close()


def main():
    """Run browser checks headlessly and release browser and widget resources."""
    rt.set_language("en")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for check in (
                check_keyboard,
                check_occurrences,
                check_scalar_and_empty,
                check_large_coordinates,
            ):
                check(browser)
                print(f"PASS {check.__name__}", flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
