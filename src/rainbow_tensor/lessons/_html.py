"""Render inert captured SVG images and a self-contained finite-state player."""

import base64
import hashlib
import json
from html import escape
from importlib.resources import files

_STYLE = """
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #f3f5f8; color: #172033; }
main { max-width: 1080px; margin: auto; padding: 28px 20px 60px; }
h1 { font-size: clamp(1.5rem, 4vw, 2.2rem); margin: 0 0 12px; }
p { line-height: 1.6; }
.scope, .keyboard { color: #536079; max-width: 80ch; }
nav { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin: 24px 0; }
button { font: inherit; border: 1px solid #a5b1c6; border-radius: 8px; padding: 9px 16px;
         background: #fff; color: inherit; cursor: pointer; }
button:hover:enabled { border-color: #315acb; }
button:disabled { opacity: .45; cursor: default; }
:focus-visible { outline: 3px solid #6d8cf0; outline-offset: 3px; }
.card { background: #fff; border: 1px solid #d6deeb; border-radius: 12px;
        padding: 18px; margin: 16px 0; }
.figure { overflow: auto; }
img { display: block; max-width: 100%; height: auto; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.6; margin-bottom: 0; }
code { overflow-wrap: anywhere; }
dl { display: grid; grid-template-columns: minmax(120px, auto) 1fr; gap: 8px 18px; }
dt { color: #536079; } dd { margin: 0; }
summary { cursor: pointer; }
[hidden] { display: none !important; }
@media (prefers-color-scheme: dark) {
  body { background: #111621; color: #e9eef8; }
  .card, button { background: #1b2434; border-color: #3d4a61; }
  .scope, .keyboard, dt { color: #acb8cc; }
}
@media (max-width: 550px) { main { padding: 18px 12px; } dl { grid-template-columns: 1fr; } }
"""


def _image(svg):
    """Use the browser's inert SVG image context instead of inserting SVG markup."""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def render_html(recording):
    """Embed a validated recording behind a restrictive offline content policy.

    Explanation text and imported metadata reach the DOM only through
    ``textContent``. SVG is an image source, so its scripts and external document
    access cannot execute. The CSP permits only the exact shipped player script,
    inline presentation styles, and embedded image data. No network is required.
    """
    document = recording.to_dict()
    for state in document["states"]:
        state["image"] = _image(state.pop("svg"))
        if state["prediction"] is not None:
            prediction = state["prediction"]
            prediction["source_image"] = _image(prediction.pop("source_svg"))
    payload = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    player = files("rainbow_tensor").joinpath("_widgets/lesson_player.js").read_text(
        encoding="utf-8",
    )
    script_hash = base64.b64encode(hashlib.sha256(player.encode("utf-8")).digest()).decode("ascii")
    policy = (
        "default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
        f"script-src 'sha256-{script_hash}'; base-uri 'none'; form-action 'none'; "
        "connect-src 'none'; object-src 'none'"
    )
    labels = document["labels"]
    return f'''<!doctype html>
<html lang="{escape(document["language"], quote=True)}">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{escape(policy, quote=True)}">
<title>{escape(document["title"])}</title><style>{_STYLE}</style></head>
<body><main id="lesson-player" tabindex="0" aria-label="{escape(document["title"], quote=True)}">
<h1>{escape(document["title"])}</h1><p class="scope">{escape(labels["scope"])}</p>
<nav aria-label="{escape(labels["state"], quote=True)}">
<button id="previous" type="button">{escape(labels["previous"])}</button>
<strong id="state-number" aria-live="polite" aria-atomic="true"></strong>
<button id="next" type="button">{escape(labels["next"])}</button></nav>
<p class="keyboard">{escape(labels["keyboard"])}</p>
<section id="prediction" class="card" hidden>
<h2>{escape(labels["prediction"])}</h2><p id="expression"></p>
<div class="figure"><img id="source-image" alt="{escape(labels["source"], quote=True)}"></div>
<p id="prediction-prompt"></p><pre id="prediction-answer"></pre>
<button id="reveal" type="button">{escape(labels["reveal"])}</button></section>
<section id="result" class="card" aria-label="{escape(labels["result"], quote=True)}">
<p id="focus"></p>
<div class="figure"><img id="result-image" alt="{escape(labels["result"], quote=True)}"></div>
<pre id="explanation"></pre><dl id="completeness"></dl>
<details><summary>{escape(labels["details"])}</summary><pre id="metadata"></pre></details>
</section></main><script id="lesson-data" type="application/json">{payload}</script>
<script>{player}</script></body></html>'''
