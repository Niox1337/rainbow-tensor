# rainbow-tensor

**See how tensor operations move, reuse and combine elements.**

[![PyPI](https://img.shields.io/pypi/v/rainbow-tensor.svg)](https://pypi.org/project/rainbow-tensor/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/Niox1337/rainbow-tensor/blob/main/pyproject.toml)
[![Tests](https://github.com/Niox1337/rainbow-tensor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Niox1337/rainbow-tensor/actions/workflows/ci.yml)
[![Backend contracts](https://github.com/Niox1337/rainbow-tensor/actions/workflows/backends.yml/badge.svg?branch=main)](https://github.com/Niox1337/rainbow-tensor/actions/workflows/backends.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Niox1337/rainbow-tensor/blob/main/LICENSE)

[Documentation](https://rainbow-tensor.zhixiangfeng.com/) ·
[Notebook examples](https://github.com/Niox1337/rainbow-tensor/tree/main/examples) ·
[LLM visualization guide](https://rainbow-tensor.zhixiangfeng.com/guide/llm-prompts.html) ·
[Releases](https://github.com/Niox1337/rainbow-tensor/releases)

rainbow-tensor turns tensor shapes and operations into SVG figures for Jupyter
notebooks, lessons and technical explanations. Choose an output element to see
its source coordinates and the calculation behind it. Record several operations
to follow that element all the way back to the original inputs.

[![Four tensor panels showing repeated reverse indexing, transpose and sum, with the sources of the first output highlighted](https://raw.githubusercontent.com/Niox1337/rainbow-tensor/main/examples/images/operation_origins.svg)](https://raw.githubusercontent.com/Niox1337/rainbow-tensor/main/examples/images/operation_origins.svg)

*One result, three contributions, two original positions. The first output is
`6 + 3 + 6 = 15`, because the same source element was sampled twice.*

## Installation

Python **3.10 or newer** is required.

```sh
python -m pip install rainbow-tensor
```

For clickable result cells and notebook controls:

```sh
python -m pip install "rainbow-tensor[interactive]"
```

Install into the Python environment used by your notebook kernel. NumPy and
IPython are included as dependencies. PyTorch, JAX and TensorFlow are optional
and installed separately. Static SVG rendering does not require widget packages.

## Your first visualization

Paste this into a notebook cell:

```python
import numpy as np
import rainbow_tensor as rt
from IPython.display import display

x = np.arange(1, 7).reshape(2, 3)
# [[1, 2, 3],
#  [4, 5, 6]]

visual = rt.sum(x, axis=1, focus=(1,))
display(visual)
```

The result has shape `(2,)`. Focusing output `(1,)` highlights the second row
and explains `4 + 5 + 6 = 15`. Change the focus to `(0,)` to follow the first row.
The explanation appears below the figure and is also available as `visual.text`.

To inspect an indexing expression instead:

```python
selection = ([1, 0, 1], slice(None, None, -1))
display(rt.index(x, selection, focus=(2, 0)))
```

The result is `[[6, 5, 4], [3, 2, 1], [6, 5, 4]]`. Its position `(2, 0)` reads
source `(1, 2)`. Repeated picks keep their separate positions in the result.

## Follow a complete operation chain

`Flow` records the steps that produced a result. Each step retains its shape
and coordinate mapping without allocating a complete intermediate tensor.

```python
flow = rt.Flow()
source = flow.input(x, name="X")
selected = flow.index(source, selection, name="S")
transposed = flow.transpose(selected, name="T")
y = flow.sum(transposed, axis=1, name="Y")

assert y.shape == (3,)
assert [y.value((i,)) for i in range(3)] == [15, 12, 9]
display(y.visualize(focus=(0,)))
```

For `Y[0]`, the trace follows `T[0, 0]`, `T[0, 1]` and `T[0, 2]` through the
transpose and index back to `X[1, 2]`, `X[0, 2]` and `X[1, 2]`. The duplicate
contribution remains visible.

```python
trace = y.trace((0,))
assert trace.complete
counts = {item.reference.coordinate: item.count for item in trace.roots}
assert counts == {(1, 2): 2, (0, 2): 1}
```

A structural trace reads no array values. Input occurrence counts describe
paths through the recorded expression, not coefficients or derivatives.
Record operations explicitly through `flow` to retain their history. Input
values are read afresh on each value query or visualization.

Read the [provenance guide](https://rainbow-tensor.zhixiangfeng.com/guide/provenance.html)
or run [the worked notebook](https://github.com/Niox1337/rainbow-tensor/blob/main/examples/15_operation_origins.ipynb).

## Explore by clicking

With the interactive extra installed, pass a tracked result or an operation
and its arguments to `explore`:

```python
explorer = rt.explore(y)
display(explorer)

# A single operation can be explored without recording a Flow.
index_explorer = rt.explore(rt.index, x, selection)
display(index_explorer)
```

Click a visible result cell to update the source highlights and explanation.
Enter and Space activate a focused cell. Coordinate fields also reach positions
hidden by a large preview.

```python
explorer.set_focus((2,))
explorer.visual.save("focused-result.svg")

# Run these when you have finished with the controls.
explorer.close()
index_explorer.close()
```

Live controls need a running notebook kernel and a host with widget support.
`y.visualize(focus=(2,))` produces the equivalent static figure.

## Supported operations

| What you want to understand | Public views |
| --- | --- |
| Tensor structure | `shape` |
| Slices, masks and repeated gathers | `index`, `take`, `repeat` |
| Reshaping and axis movement | `reshape`, `transpose`, `swapaxes`, `moveaxis`, `squeeze`, `expand_dims` |
| Joining and broadcasting | `concatenate`, `stack`, `broadcast` |
| Reductions and contractions | `sum`, `mean`, `matmul`, `einsum` |
| Storage metadata | `memory` |

Operation views support `focus=` to explain one result element. The same
operations are available as `Flow` methods. `shape` and `memory` are standalone
inspection views. `flow.broadcast` returns one tracked output per input, each
with its own values and output port.

Indexing supports integers, slices, ellipsis, new axes, boolean masks and
advanced integer arrays. Reductions support multiple axes, negative axes,
`axis=None`, `axis=()` and `keepdims=True`. Scalars use shape and coordinate
`()`. Empty tensors retain their zero-length dimensions and have no selectable
output element.

NumPy arrays and CPU tensors from **PyTorch, JAX and TensorFlow** are covered by
the backend contract suite. Other array-like inputs need a shape and scalar
coordinate access. Shape tuples such as `rt.shape((2, 3, 4))` use generated
row-major placeholder values.

## Figures, explanations and limits

Every view returns a `TensorVisual` with an SVG, a plain-text explanation and
operation metadata. Its `trace` describes a focused output. Visuals from a
recorded Flow also expose a bounded `provenance` tree.

```python
visual.save("row-sum.svg")
print(visual.text)
```

`save` writes the figure only. Keep `visual.text` alongside it when sharing the
explanation outside a notebook. SVG figures remain usable without a running
Python process.

Large tensors use bounded previews. Repeat mappings and basic slice selections
remain compact instead of enumerating every output position. Math views default
to 10,000 terms per output and 100,000 across the preview. Flow evaluation also
counts recursive factor references and input reads. Work that exceeds a budget
is shown as `?`, and incomplete structural traces identify the limit reached.

Numerical previews use **Python scalar arithmetic**. Backend accumulation dtype,
rounding and overflow may differ. The figures explain logical operations and
coordinate origins, while `memory` reports available storage metadata.

## Theme and language

The default `auto` theme follows the viewer's light or dark preference, including
in saved SVG files. Language follows the Python environment and system locale.
English and Simplified Chinese are included.

```python
rt.set_default_theme("auto")
rt.set_language("en")
display(rt.shape(x, theme="dark"))  # Override the theme for one figure.
```

Add a language by supplying a JSON translation catalog. Missing entries fall
back to English, and `rt.load_translations("translations")` loads extra catalogs
without changing application code. See
[themes](https://rainbow-tensor.zhixiangfeng.com/guide/themes-and-configuration.html)
and [translations](https://rainbow-tensor.zhixiangfeng.com/guide/translations.html).

## Learn and teach

| Start here | What you will find |
| --- | --- |
| [Learning path](https://rainbow-tensor.zhixiangfeng.com/guide/learning-path.html) | Small exercises that connect shapes, output coordinates and source values |
| [Notebook collection](https://github.com/Niox1337/rainbow-tensor/tree/main/examples) | Runnable examples from basic shapes through cross-operation tracing |
| [API reference](https://rainbow-tensor.zhixiangfeng.com/api.html) | Function signatures, parameters and result objects |
| [LLM visualization guide](https://rainbow-tensor.zhixiangfeng.com/guide/llm-prompts.html) | A reusable prompt and verified patterns for generating beginner explanations |
| [Architecture](https://rainbow-tensor.zhixiangfeng.com/guide/architecture.html) | Coordinate mappings, rendering boundaries and performance decisions |

## Contributing

Bug reports, clearer teaching examples and translation catalogs are welcome.
For a bug report, include a small reproducing input, the operation, the expected
result and your package, Python and backend versions. Use
[GitHub Issues](https://github.com/Niox1337/rainbow-tensor/issues) to report a
problem or discuss a new operation.

To work on the package:

```sh
git clone https://github.com/Niox1337/rainbow-tensor.git
cd rainbow-tensor
python -m pip install -e ".[dev,interactive]"
python -m pytest
python -m ruff check .
```

Optional backend tests skip when their framework is absent from that Python
environment. CI separately requires PyTorch, JAX and TensorFlow, as well as
running the main suite on Python 3.10 and 3.12. SVG golden tests protect existing
figures, and distribution checks exercise installed wheels and source packages.

```sh
python -m build
python scripts/check_distribution.py --interactive
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

The distribution check expects one wheel and one source archive in `dist`.
Use `--dist-dir PATH` for a different build directory. Keep changes focused and
add a regression example when correcting an operation's behavior.

## License

[MIT](https://github.com/Niox1337/rainbow-tensor/blob/main/LICENSE),
Copyright 2026 Zhixiang Feng.
