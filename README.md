# rainbow-tensor

Visualise tensor shape, indexing, and slicing as SVG inside IPython and Jupyter notebooks.

rainbow-tensor is made for people who are learning how a tensor is structured and how an indexing expression selects elements. It draws the tensor as nested blocks, rows, and cells, then highlights exactly which elements an index picks out.

## Features

- Static SVG output that stays sharp at any zoom level in a notebook
- Shape visualisation for 1D, 2D, and 3D tensors
- Index visualisation with highlighted selections and a plain text explanation
- Works with shape tuples and with array-like objects that expose a `.shape` attribute, such as NumPy arrays
- No tensor library is imported by the core, so the package stays lightweight

## Installation

Install from source for development.

```bash
git clone https://github.com/Niox1337/rainbow-tensor.git
cd rainbow-tensor
pip install -e .
```

Install with the development tools (pytest, ruff, build).

```bash
pip install -e ".[dev]"
```

## Usage

Run the examples in a Jupyter notebook or an IPython shell so the SVG is displayed.

Visualise a shape.

```python
from rainbow_tensor import show_shape

show_shape((2, 2, 2))
```

Visualise how an index selects elements.

```python
from rainbow_tensor import show_index

show_index((2, 2, 2), (0, slice(None), 1))
```

For the shape `(2, 2, 2)` the index `(0, slice(None), 1)` selects the values `1` and `3`, the selected coordinates are `(0, 0, 1)` and `(0, 1, 1)`, and the result shape is `(2,)`.

Use a real NumPy array to display its actual values.

```python
import numpy as np
from rainbow_tensor import show_shape, show_index

x = np.arange(8).reshape(2, 2, 2)
show_shape(x)
show_index(x, (0, slice(None), 1))
```

Each function returns a small result object. Its `svg` attribute holds the SVG string, so the package can be inspected and tested outside a notebook.

## Supported

- 1D, 2D, and 3D tensors
- Shape tuples and array-like objects with a `.shape` attribute
- Integer indexing
- Basic slicing with `slice(None)`, `slice(start, stop)`, and `slice(start, stop, step)`

## Unsupported in the first version

- Advanced NumPy indexing
- Boolean masks
- `None` and `newaxis`
- Ellipsis
- 4D or higher tensors
- Interactive controls and animation

## Development

```bash
pytest
ruff check .
python -m build
```

## License

MIT
