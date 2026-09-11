# Following groups by colour

A reduction gives each result coordinate a group identity. The source values
that contribute to that coordinate share its fill and border. The focused group
uses the theme's selection highlight, so it remains easy to find among the other
groups.

```python
import numpy as np
from IPython.display import display

import rainbow_tensor as rt

x = np.arange(36).reshape(3, 12)
display(rt.sum(x, axis=0, focus=(0,), theme="light"))
display(rt.sum(x, axis=0, focus=(6,), theme="light"))
```

Each column contributes three values to one output. Moving focus changes the
highlighted column. The other groups retain their colours because their
identities come from result coordinates, independently of the visible preview.
The same rule applies to `mean`, multiple reduction axes and `keepdims=True`.

## Beyond a short palette

Group colours are generated from the logical group ID in OkLCh colour space.
Hue, lightness and chroma vary at different rates instead of restarting a fixed
palette after a few groups. Colours outside sRGB are brought into range by
reducing chroma while retaining hue and lightness. The conversion follows the
[published Oklab equations](https://bottosson.github.io/posts/oklab/).

Only requested groups are calculated. A cache retains at most 1,024 entries,
so a large logical tensor does not require a palette with one entry per output.
Concatenation, stacking, gathering and repetition use the same generator for
their operand identities. Axis frames and einsum label roles keep their own
colour rules.

The first 240 generated group fills are distinct in the tested light and dark
palettes. That does not make hundreds of groups visually distinguishable.
Similar shades become harder to tell apart in a crowded figure. Use the cell
coordinates, focus and contribution trace to identify an exact group.

## Light, dark and automatic figures

```python
display(rt.sum(x, axis=0, focus=(6,), theme="dark"))
display(rt.sum(x, axis=0, focus=(6,), theme="auto"))
```

Each generated group has paired light and dark paints. An automatic SVG switches
between them with the viewer's colour preference, alongside the rest of the
theme. Its literal light colours remain available to viewers without CSS
support. A saved automatic SVG keeps this behaviour without reading the tensor
again.

Fixed custom themes choose the group palette that offers higher contrast with
their actual text colour. When that text uses a CSS value the package cannot
interpret, background brightness and then the preset name provide a fallback.
Check your own text colours against the generated fills when making a custom
theme.

The [group colour notebook](../../examples/13_group_colours.ipynb) walks through
focus changes and a multi-axis reduction. The [architecture guide](architecture)
describes the rendering modules and preview limits.
