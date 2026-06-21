# Reshaping and moving axes

These views rearrange a tensor without changing its values. The source and the
result sit in one figure with a connector, and each axis keeps its colour so you
can trace where it lands.

```python
import numpy as np
import rainbow_tensor as rt

x = np.arange(12).reshape(3, 4)
```

## Reshape

Reshape keeps the row major order, so element k stays element k while the layout
changes. A single `-1` lets one axis be inferred from the rest.

```python
rt.reshape(x, (2, 6))
rt.reshape(np.arange(24).reshape(2, 3, 4), (-1, 4))
```

## Transpose and swapaxes

`transpose` reverses or permutes the axes, colouring each result axis by the
source axis it came from. `swapaxes` does the same for exactly two axes, with
negative axes resolved like NumPy.

```python
rt.transpose(x)
rt.transpose(x, (1, 0))
rt.swapaxes(np.arange(24).reshape(2, 3, 4), 0, 2)
```

## Moveaxis

`moveaxis` moves one or several axes to chosen destinations while the rest keep
their order, and each moved axis keeps its source colour.

```python
rt.moveaxis(np.arange(24).reshape(2, 3, 4), 0, 2)
```

## Squeeze and expand_dims

`squeeze` removes size one axes, either every one of them or a chosen axis or
tuple, and marks the removed axes while each surviving axis keeps its colour.
`expand_dims` inserts a size one axis at a positive or negative position and
marks the inserted axis in the accent colour.

```python
rt.squeeze(np.ones((1, 3, 1)))
rt.squeeze(np.ones((1, 3, 1)), 0)
rt.expand_dims(x, 1)
```
