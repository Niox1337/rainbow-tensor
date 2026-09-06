# Themes and configuration

Every view shares one theme system and reads values through a tiny adapter, so
the look and the input source are both flexible.

## Themes

The default theme is `rt.AUTO`, also available as `theme="auto"`. Its SVG follows
the browser's `prefers-color-scheme` setting, including changes while the figure
is open. A saved SVG keeps this behaviour in viewers that support it. Other SVG
viewers receive literal light colours as a fallback.

Pass `theme="light"` or `theme="dark"` for a fixed palette, or set a default for
later calls. A theme passed to one call takes priority over that default.

```python
import numpy as np
import rainbow_tensor as rt

x = np.arange(8).reshape(2, 2, 2)
rt.shape(x, theme="dark")

rt.set_default_theme("dark")
rt.index(x, (0, slice(None), 1))
rt.shape(x, theme="light")  # override the default for this figure
rt.set_default_theme("auto")
```

A theme bundles the colours, fonts, cell size, stroke width, the per axis
truncation limit, and the total visible cell budget. Derive a tweaked copy with
`variant` and pass it directly.

```python
roomy = rt.AUTO.variant(cell_w=64, max_cells=8, max_visible_cells=160)
rt.shape(np.arange(8).reshape(2, 4), theme=roomy)
```

Starting from `rt.AUTO` keeps its adaptive palette. Start from `rt.LIGHT` or
`rt.DARK` to keep a custom theme fixed. Explicit colour values in a variant stay
as supplied, so choose colours that work against both backgrounds when changing
an adaptive theme.

## Global axis colours

To recolour the axis frames everywhere without building a whole theme, set an
axis ramp once with `set_default_axis_colors`. Each colour maps to one axis
depth, and the ramp wraps for deeper tensors. It applies to every later call
that does not pass its own `theme`, so a per call `theme` still wins.

```python
rt.set_default_axis_colors(["#2563eb", "#db2777", "#16a34a"])
rt.shape((2, 2, 2))
rt.shape((2, 2, 2), theme="dark")   # the per call theme keeps its own ramp
rt.set_default_axis_colors(None)    # clear it and fall back to the theme ramp
```

`get_default_axis_colors` returns the current ramp, or `None` when none is set.

## Explanation language

The default language setting is `"auto"`. It reads preferences from the Python
kernel's environment and operating system. Theme detection happens in the
browser, so a remote notebook can have a local dark theme and a language chosen
by the remote kernel.

Set a language explicitly to override automatic detection for later visuals.
English (`"en"`) and Simplified Chinese (`"zh"`) are bundled.

```python
rt.set_language("zh-CN")
rt.get_language()           # -> "zh-CN", the requested setting
rt.get_resolved_language()  # -> "zh", the available catalog
rt.available_languages()   # -> ("en", "zh") before loading other catalogs
rt.set_language("en")      # fixed English
rt.set_language("auto")    # return to automatic selection
```

Automatic selection checks `LANGUAGE`, `LC_ALL`, `LC_MESSAGES`, and `LANG` in
that order. It then tries the Windows UI language on Windows, or Python's
current locale when the UI language is unavailable, and finally English.
Colon-separated preferences are tried in order. Regional tags try their parent
tags, and `C` or `POSIX` selects English. An explicit `set_language(...)` setting
skips these environment and system preferences.

`get_language()` returns the requested setting, including `"auto"`.
`get_resolved_language()` returns the catalog selected for the current
environment. Missing messages fall back through parent catalogs to English.
This selection does not change Python's process locale. Existing visual objects
keep the text created with them, so create a new visual to see a language change.

API operation names and validation exceptions remain in English. See
[Translations](translations) to add a language or load a custom catalog.

## Backend arrays

rainbow-tensor reads arrays through duck typing. If an object exposes a `.shape`
attribute and supports coordinate reads, the visualiser can draw its values
without importing that backend in the core package. NumPy, PyTorch, JAX, and
TensorFlow style arrays all work.

```python
import torch
import jax.numpy as jnp

rt.shape(torch.arange(6).reshape(2, 3))
rt.shape(jnp.arange(6).reshape(2, 3))
```

The CPU contract suite compares index, reshape, transpose, sum, mean, matmul,
and einsum values with NumPy for integer, floating-point, and transposed
inputs. Separate CI jobs install PyTorch, JAX, and TensorFlow explicitly.
A missing required framework fails its job rather than skipping it.
Local runs may skip frameworks that are not installed. GPU transfers, compiled
tracers, and framework-specific accumulation dtypes are outside this contract.

## Renderers

SVG is the default renderer. A custom renderer can be registered for other
output formats, and it receives the same shape, panels, value functions, theme,
and precision that the SVG renderer uses.

```python
class TextRenderer:
    name = "text"
    mime_type = "text/plain"

    def render_tensor(self, **kwargs):
        return str(kwargs["shape"])

    def render_panels(self, **kwargs):
        return str(kwargs["panels"][-1]["shape"])


rt.register_renderer(TextRenderer())
rt.shape((2, 3), renderer="text")
```

Use `set_default_renderer` to choose a renderer for later calls, or pass
`renderer=` on one call for a local override.

Panel shapes are logical shapes. A scalar panel has shape `()` and its one
value is read with `value_fn(())`. Older versions used a `(1,)` display panel
for some scalar results, so custom renderers should remove that workaround.
If any dimension is zero, there are no cell coordinates and the renderer must
not call `value_fn`. A math view with an empty result has `visual.trace=None`.
