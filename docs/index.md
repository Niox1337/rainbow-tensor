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

The user guide walks through each feature group in order, mirroring the runnable
notebooks in the `examples` folder.

- [Shapes](guide/shapes) draws a tensor and explains the colour scheme, float
  precision, saving, and big tensor previews
- [Indexing](guide/indexing) covers integers, slices, ellipsis, new axes,
  boolean masks, and fancy integer arrays
- [Reshaping and moving axes](guide/reshaping) covers reshape, transpose,
  swapaxes, moveaxis, squeeze, and expand_dims
- [Reductions and math](guide/reductions-and-math) covers sum, mean, matmul,
  and einsum
- [Combining tensors](guide/combining) covers concatenate, stack, broadcast,
  repeat, and take
- [Themes and configuration](guide/themes-and-configuration) covers themes,
  global axis colours, backend arrays, and renderers

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

guide/shapes
guide/indexing
guide/reshaping
guide/reductions-and-math
guide/combining
guide/themes-and-configuration
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api
```
