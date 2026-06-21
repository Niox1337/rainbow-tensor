# Combining tensors

These views join several tensors into one or stretch one to a new shape. Each
operand is drawn in its own tint, so the seam between operands or the new axis
stays clear.

```python
import numpy as np
import rainbow_tensor as rt

a = np.arange(6).reshape(2, 3)
b = np.arange(100, 106).reshape(2, 3)
```

## Concatenate and stack

`concatenate` joins operands along an existing axis, and the sizes may differ on
that axis. `stack` places the operands onto a brand new axis, so every operand
keeps its full shape. A mismatch raises a clear error rather than drawing a
wrong figure.

```python
rt.concatenate([a, b], 0)
rt.concatenate([a, b], 1)
rt.stack([a, b], 0)
```

## Broadcast

`broadcast` stretches a smaller operand to match a larger one. Each operand is
drawn in its own shape and again stretched to the common shape, so the repeated
values along a stretched axis are visible and every stretched axis is marked.
Axes line up from the right, and on each axis the sizes must be equal or one of
them must be `1`.

```python
rt.broadcast((3, 1), (1, 4))
rt.broadcast((2, 3, 4), (4,))
```

## Repeat

`repeat` copies elements along an axis. A single count repeats every element, or
one count per element repeats each by its own amount, matching `numpy.repeat`.
Each source element and the run of copies it produces share one tint, so the
result reads as real materialised copies. This is the contrast with `broadcast`,
which only stretches a size one axis and copies nothing.

```python
x = np.arange(12).reshape(3, 4)
rt.repeat(x, 2, axis=0)
rt.repeat(x, [1, 2, 0, 1], axis=1)
```

## Take

`take` gathers values along one axis from a list of indices, matching
`numpy.take`. Each gathered source slice and the result slice it feeds share one
tint, so a repeated index repeats a tint and a reordered index reorders them.
Negative axes and negative indices both work.

```python
rt.take(x, [0, 2, 2, 1], axis=1)
rt.take(x, [2, 0], axis=0)
```
