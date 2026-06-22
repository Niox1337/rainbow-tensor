# Themes and configuration

Every view shares one theme system and reads values through a tiny adapter, so
the look and the input source are both flexible.

## Themes

Pass `theme="light"` or `theme="dark"` to any call, or set a module default that
every later call follows.

```python
import numpy as np
import rainbow_tensor as rt

x = np.arange(8).reshape(2, 2, 2)
rt.shape(x, theme="dark")

rt.set_default_theme("dark")
rt.index(x, (0, slice(None), 1))
```

A theme bundles the colours, fonts, cell size, stroke width, the per axis
truncation limit, and the total visible cell budget. Derive a tweaked copy with
`variant` and pass it directly.

```python
roomy = rt.LIGHT.variant(cell_w=64, max_cells=8, max_visible_cells=160)
rt.shape(np.arange(8).reshape(2, 4), theme=roomy)
```

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

Every caption line under a visual comes from a keyed template in
`rainbow_tensor.explanations`. The active language is global. Set it once with
`set_language`, and `get_language` returns the current one.

```python
rt.set_language("de")   # affects every later visual
rt.get_language()       # -> "de"
rt.set_language("en")   # back to the default
```

Only English (`"en"`) ships today. A translation adds one table to
`MESSAGES` keyed by its language code, reusing the English keys. English is the
fallback at two levels, so an unknown language or a key a translation has not
filled yet still renders the English line rather than failing.

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
