# rainbow-tensor

A Python package designed specifically for IPython/Jupyter notebook environments with numpy support.

## Features

- Can only be used in IPython environments (Jupyter notebooks, IPython shell)
- Automatically includes numpy as a dependency
- Simple and lightweight

## Installation

Install the package using pip:

```bash
pip install rainbow-tensor
```

Or install from source:

```bash
git clone https://github.com/Niox1337/rainbow-tensor.git
cd rainbow-tensor
pip install .
```

## Usage

This package can **only** be used in IPython/Jupyter notebook environments. If you try to import it in a regular Python script, it will raise an error.

**In a Jupyter notebook or IPython shell:**

```python
import rainbow_tensor
# numpy is available through the package
print(rainbow_tensor.np.array([1, 2, 3]))
```

**In a regular Python script (this will fail):**

```python
import rainbow_tensor  # RuntimeError: rainbow-tensor can only be used in IPython environments
```

## Requirements

- Python >= 3.6
- numpy >= 1.19.0
- ipython >= 7.0.0

## License

MIT