# rainbow-tensor

Colourful SVG visuals for tensor shapes, indexing, and operations, built for Jupyter notebooks and teaching.

[![PyPI version](https://img.shields.io/pypi/v/rainbow-tensor.svg?color=8b5cf6)](https://pypi.org/project/rainbow-tensor/)
[![Python versions](https://img.shields.io/pypi/pyversions/rainbow-tensor.svg)](https://pypi.org/project/rainbow-tensor/)
[![License](https://img.shields.io/pypi/l/rainbow-tensor.svg?color=blue)](https://github.com/Niox1337/rainbow-tensor/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-rainbow--tensor-8b5cf6.svg)](https://rainbow-tensor.zhixiangfeng.com/)

```python
rt.shape(np.arange(8).reshape(2, 2, 2))
```

![Shape (2, 2, 2)](examples/images/shape_2x2x2.svg)

## Why rainbow-tensor

Tensor shapes are hard to hold in your head, and an index like `(0, slice(None), 1)` gives no hint of what it selects until you run it. A printed array is just a wall of numbers. rainbow-tensor draws the tensor as nested coloured frames and highlights exactly which elements an operation touches, so the structure and the result are clear at a glance.

Every axis keeps one colour through every view, so you can follow an axis as it moves, folds, or stretches. That makes it a fast way to learn how reshapes and reductions work, to teach shape transformations, and to debug a confusing indexing or broadcasting bug. The core imports no deep learning framework, so it stays light and works with plain NumPy or any array that exposes a shape.

## Install

```bash
pip install rainbow-tensor
```

The distribution name is `rainbow-tensor` and the import name is `rainbow_tensor`.

## Quick start

Run inside a Jupyter notebook or an IPython shell so the SVG is displayed. The convention is to import the package as `rt`.

```python
import numpy as np
import rainbow_tensor as rt

x = np.arange(8).reshape(2, 2, 2)

rt.shape(x)                          # draw the structure
rt.index(x, (0, slice(None), 1))     # highlight what an index selects
```

![Index (0, :, 1)](examples/images/index_0_all_1.svg)

Each call returns a small result object. Its `svg` attribute holds the SVG string, its `text` attribute holds the explanation printed under the figure, and `save` writes the SVG to a file.

## What it can show

Shape changing, combining, and broadcasting views draw the source and the result side by side, so the mapping between them is easy to follow.

- **Shapes and indexing** with `shape` and `index`, covering integers, slices, ellipsis, new axes, boolean masks, and fancy integer arrays
- **Reshaping and moving axes** with `reshape`, `transpose`, `swapaxes`, `moveaxis`, `squeeze`, and `expand_dims`
- **Reductions and math** with `sum`, `mean`, `matmul`, and `einsum`, including multiple reduction axes and `keepdims`
- **Combining** with `concatenate`, `stack`, `broadcast`, `repeat`, and `take`
- **Output explanations** with `focus=` on sums, means, matmul, and einsum
- **Memory layout** with `memory`, including byte strides and data ownership

```python
rt.sum(np.arange(12).reshape(3, 4), 0)
```

![Sum over axis 0](examples/images/sum_axis0.svg)

Reduce several axes together and keep their positions for later broadcasting:

```python
x = np.arange(24).reshape(2, 3, 4)
rt.mean(x, axis=(0, 2), keepdims=True, focus=(0, 1, 0))  # (2, 3, 4) -> (1, 3, 1)
```

The focused mean combines eight source values and divides by eight. Omit `axis`
to reduce every axis, or pass `axis=()` to preserve every element. The
[reduction guide](docs/guide/reductions-and-math.md) explains the shape rules,
and [`10_reduction_axes.ipynb`](examples/10_reduction_axes.ipynb) works through
row normalisation with `keepdims` and broadcasting.

Scalars and empty tensors keep their actual shapes. `rt.shape(np.array(7))`
shows one cell at `()`, while `rt.shape((2, 0, 3))` shows **No elements**.
Reducing the zero-length axis gives six zeros for `sum` or six NaNs for `mean`.
Reducing a different axis can leave an empty output with no focus coordinate.
Try [`11_scalars_and_empty_tensors.ipynb`](examples/11_scalars_and_empty_tensors.ipynb)
to compare these cases step by step.

```python
rt.einsum("ij,jk->ik", np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4))
```

![Einsum ij and jk to ik](examples/images/einsum_ij_jk_ik.svg)

## Follow one output

Choose an output coordinate to highlight its contributing source cells and
read the corresponding formula.

```python
a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(3, 4)

visual = rt.matmul(a, b, focus=(1, 2))
visual                             # highlights 3 * 2 + 4 * 6 + 5 * 10 = 80
visual.trace.terms                 # ordered, inspectable source references
rt.mean(a, axis=1, focus=(-1,))     # follow the last row
rt.memory(a.T)                     # explain strides and storage ownership
```

Start with [the learning guide](docs/guide/learning-path.md) or run
[`07_explaining_outputs.ipynb`](examples/07_explaining_outputs.ipynb).

## Compare an index with its result

```python
x = np.array([10, 20, 30])
rt.index(x, ([2, 0, 2],), show_result=True)  # source -> [30, 10, 30]
```

The comparison keeps source colours and highlights the result. Repeated picks
stay repeated, and reverse slices retain their order. See the
[indexing guide](docs/guide/indexing.md) for output-to-source coordinate lookup.

For optional notebook controls, install `rainbow-tensor[interactive]` and use
`rt.explore(rt.matmul, a, b, focus=(1, 2))`. The
[interactive guide](docs/guide/interactive.md) covers coordinate updates and export.

## Keep previews small

Basic slice selections store ranges instead of expanding every selected
coordinate. Reductions compute only the output groups a renderer requests.
Math views accept `max_terms`, defaulting to 10,000 terms per output cell, and
`max_total_terms`, defaulting to 100,000 across the visible outputs. When either
limit is exceeded, output values show `?` with an explanation instead of partial
results. Set both limits to `None` to allow full evaluation of the preview.

Numerical previews use Python scalar arithmetic. Accumulation dtype, rounding,
and overflow can differ from a framework's native kernels. Each math view
explains that model and identifies any generated placeholder operands.

## Theme and language

The default `auto` theme follows the viewer's light or dark preference, including
saved SVG files. Language selection follows the Python environment and system
locale. English and Simplified Chinese are included. Both settings can be
overridden for reproducible examples.

```python
rt.shape(x, theme="dark")         # override this figure
rt.set_default_theme("auto")     # follow the viewer again
rt.set_language("zh-CN")         # use the Chinese catalog
rt.get_resolved_language()        # "zh"
rt.set_language("auto")          # follow the kernel's system settings
```

A new language needs only a JSON catalog with translated messages. Missing
entries fall back to English, and `rt.load_translations("translations")` loads
additional catalogs from a directory. See the [translation guide](docs/guide/translations.md)
and [`12_theme_and_language.ipynb`](examples/12_theme_and_language.ipynb).
Explicit light and dark themes, custom `Theme.variant` settings, and
`set_default_axis_colors` remain available.

## Documentation

The full guide and API reference live at [rainbow-tensor.zhixiangfeng.com](https://rainbow-tensor.zhixiangfeng.com/).

Runnable notebooks for every feature group live in [`examples`](examples), and more sample images live in [`examples/images`](examples/images).

## Development

```bash
pip install -e ".[dev,interactive]"
pytest
ruff check .
python -m build
python scripts/check_distribution.py --interactive
```

The distribution check expects one wheel and one source archive in `dist`.
Use `--dist-dir PATH` when building into another directory. It creates a fresh
temporary environment, installs dependencies, checks both installed packages,
and runs the source archive's tests with its included SVG fixtures.

## License

MIT
