# Share a finite interactive lesson

`capture_lesson` freezes explicitly chosen visual states into a `LessonRecording`.
Save it as a standalone HTML file to share the figures, source explanations,
focus coordinates, and available contribution metadata. The recipient needs
a browser. A running notebook kernel and installed Python package are unnecessary.

```{literalinclude} ../../examples/lessons/portable_recording.py
:language: python
```

The HTML opens at the first captured state. Previous and Next buttons move
between states. Left and Right arrow keys do the same, while Home and End
select the first and last states. Light and dark presentation follow the
viewer's system preference when the original figures use the automatic theme.

The example writes `row-sums.html` and `row-sums.json` in the current directory.
`recording.save(path)` always writes HTML. Use `recording.to_json()` to save the
machine-readable representation, or `LessonRecording.from_json(text)` to load it.
`to_dict()` returns an independent copy for inspection.

## Capture only the states you intend to teach

The input is an iterable of `TensorVisual`, `FocusExplorer`, `Walkthrough`, or
`ReductionPlayground` objects. A generator may change the same controller and
yield it repeatedly. Each state is copied when yielded, so later changes do not
alter earlier captures. Capturing reads the already prepared visual and metadata.
It does not trigger another tensor evaluation or enumerate output coordinates.

Walkthrough captures retain the selected occurrence and contribution, their
explanation, and available numerical details. Playground captures retain the
source figure, expression, prediction, diagnosis, and whether the result was
revealed. The offline player can hide or reveal that captured answer.
Changing axes or computing an uncaptured output still requires the live notebook.

## Keep size and completeness explicit

The default limits are 64 states and 5,000,000 UTF-8 JSON bytes. Both must be
positive integers. A recording must have at least one state. Capturing stops
with an error if a limit is exceeded. At most `max_states + 1` entries are
requested from an iterable, including the extra entry needed to detect overflow.

The byte limit covers the serialized recording. HTML adds its player and image
encoding overhead, so its file can be larger. Shapes and coordinates are stored
as decimal strings to preserve integers larger than a browser's exact number range.
The recording format has its own version, independent of the package version.

Immediate-source tracing, operation ancestry, and numerical evaluation each
retain a separate completeness status. Unknown information stays unknown.
Missing paths are not presented as zero contributions. The player explicitly
states that it contains captured states rather than a live tensor explorer.

Figure text and captured values are included in the exported file. Select the
lesson inputs and states you intend to share. The player uses embedded image
data and its shipped script, with no network requests or Python execution.

## Verify offline behavior

```bash
python scripts/check_lessons.py portable-recording
python -m pip install -r scripts/requirements-browser.txt
python -m playwright install chromium
python scripts/check_lesson_html.py
```

The browser check exercises state navigation, prediction reveal, scalar and
empty coordinates, large coordinates, theme changes, and hostile embedded
text. It checks that playback makes no network requests.
