# rainbow-tensor

rainbow-tensor visualises tensor shape, indexing, and slicing as SVG inside
IPython and Jupyter notebooks. It is meant for anyone learning how a tensor is
structured and how an indexing or a shape changing expression selects and moves
elements.

Every non-leaf axis draws a frame in its own rainbow colour, leaf elements sit
in rounded cells, and a legend names each axis with its size in the matching
colour. Adjacent axes use clearly separated colours, so a deep tensor stays
readable. Explanations print as plain notebook output below the figure. The same
figure renders in a light or a dark theme without any change to your code, and
`set_default_axis_colors` recolours the axis frames everywhere at once.

## Install

```bash
pip install rainbow-tensor
```

The distribution name is `rainbow-tensor` and the import name is
`rainbow_tensor`.

## Quick start

Import the package as `rt`, then call one of the visualisation functions in a
notebook cell so the SVG is displayed.

```python
import numpy as np
import rainbow_tensor as rt

x = np.arange(8).reshape(2, 2, 2)

rt.shape(x)                          # draw the structure of a shape
rt.index(x, (0, slice(None), 1))     # show how an index selects elements
rt.reshape((2, 3), (3, 2))           # carry values into a new layout
rt.broadcast((3, 1), (1, 4))         # stretch a smaller operand to match
```

Each function returns a small result object. Its `svg` attribute holds the SVG
string, and its `text` attribute holds the explanation printed under the
figure, so the package stays inspectable and testable outside a notebook.

## What it covers

```{rubric} Shape and indexing
```

- Tensors of any rank, with frames nested to arbitrary depth
- Big tensor previews with a total visible cell budget
- Integer indexing, basic slicing, `Ellipsis`, and `None` for a new axis
- Boolean masks and integer array indexing, matching NumPy result shapes

```{rubric} Shape changing and combining
```

- Reshape with row major order and one inferred `-1` axis
- Transpose and permute with axis colours following the move
- Sum and mean reductions over a chosen axis
- Concatenate along an existing axis and stack onto a brand new axis
- Broadcasting two tensors to a common shape, marking every stretched axis

```{toctree}
:maxdepth: 2
:caption: Contents

api
```
