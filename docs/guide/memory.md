# Memory layout

Two arrays can have the same shape but walk through their stored values in
different ways. `memory` draws the logical array and explains the storage
metadata exposed by the input.

```python
import numpy as np
import rainbow_tensor as rt

x = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
rt.memory(x)
rt.memory(x.T)
```

For `x`, the byte strides are `(12, 4)). Moving one row advances 12 bytes,
while moving one column advances 4 bytes. Transposing gives strides `(4, 12)`:
the logical axes change places while the underlying values remain in place.
The transposed array is backed by another object and does not own its data.

## Follow slices through storage

```python
rt.memory(x[:, ::-1])
rt.memory(x[:, ::2])
rt.memory(x.T.copy())
```

A negative stride walks backwards through storage. A larger stride skips
values. A zero stride, as in some broadcast views, reuses a stored value along
an axis. C-contiguous and F-contiguous flags describe whether the values form
a contiguous layout in row-major or column-major order. An array can satisfy
both conditions when its dimensions allow it.

The diagram always shows logical indices. It does not draw physical addresses.
The ownership and base fields describe the current array, but cannot identify
which earlier operation created it or establish whether two arbitrary arrays
share memory.

## Inspect metadata in Python

```python
visual = rt.memory(x.T)
visual.metadata["strides"]       # (4, 12)
visual.metadata["owns_data"]     # False
visual.metadata["has_base"]      # True
print(visual.text)
```

The metadata dictionary also contains `shape`, `dtype`, `itemsize`,
`c_contiguous`, and `f_contiguous`. Missing fields are `None`.
Only NumPy-style byte-stride and flag attributes are read. Backend-specific
methods are not invoked to guess equivalent metadata.

A shape tuple carries no storage information:

```python
rt.memory((2, 3))                # storage fields are reported as unknown
```

Object dtype item sizes describe the array's element slots, not the total
memory used by the Python objects stored in those slots.
