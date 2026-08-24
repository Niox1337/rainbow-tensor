# Explain one tensor operation at a time

Start with small arrays whose values you can check by hand. Use one question
per figure: what is the shape, which cells are selected, or how is one output
computed?

```python
import numpy as np
from IPython.display import display

import rainbow_tensor as rt

a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(3, 4)
rt.shape(a)
```

## Predict a result, then inspect it

The entry at row 1, column 2 of `a @ b` is a dot product. Predict its value,
then ask the visualisation to focus on that output:

```python
visual = rt.matmul(a, b, focus=(1, 2))
visual
```

The source highlights identify the contributing row and column. The output
highlight identifies the answer. Change `focus` to another coordinate to
follow a different calculation.

```python
assert (a @ b)[1, 2] == 80
visual.trace.output_coord
visual.trace.terms
```

Each trace term contains references to source elements. References in one
term multiply together, and the terms add together. A mean also divides by
the number of elements in its group. Traces keep a bounded sample of terms
and report the full term count, so a long sum does not fill the notebook.

## Compare a sum with a mean

```python
rt.sum(a, axis=1, focus=(1,))
rt.mean(a, axis=1, focus=(1,))
```

Both focus on the same row. Its sum is `3 + 4 + 5 = 12`, while its mean is
`12 / 3 = 4`. A one-dimensional output needs a one-element coordinate tuple,
including its trailing comma. Negative coordinates count from the end.
A scalar output uses `focus=()`.

## Keep an axis to normalize each row

Suppose each row contains three scores. To turn them into fractions of that
row's total, each row needs its own divisor:

```python
scores = np.array([[2., 4., 6.], [3., 6., 9.]])
display(rt.sum(scores, axis=-1, keepdims=True, focus=(1, 0)))

row_totals = scores.sum(axis=1, keepdims=True)  # shape (2, 1)
display(rt.broadcast(scores, row_totals))
normalized = scores / row_totals
np.testing.assert_allclose(normalized.sum(axis=1), [1., 1.])
rt.shape(normalized)
```

`keepdims=True` leaves one column for the totals, 12 and 18. Broadcasting
stretches each total across its own row. Without that retained axis, the totals
have shape `(2,)`, which cannot line up with the three columns.

The visual explains the reduction and broadcasting. NumPy computes the array
used for division. To combine several axes, pass a tuple such as `axis=(0, 2)`.
Omitting `axis` reduces all axes, while `axis=()` reduces none. See the
[reduction guide](reductions-and-math.md) for their result shapes and focus tuples.

## Read the same calculation as einsum

```python
rt.einsum("ij,jk->ik", a, b, focus=(1, 2))
```

Here `i` chooses the output row, `k` chooses the output column, and `j`
is summed out. Matching label colours let you follow those roles across
the operands.

## Separate shape from storage

```python
rt.transpose(a)
rt.memory(a.T)
rt.memory(a.T.copy())
```

The first view explains where values move in the logical grid. The other
two report byte strides and ownership. A shape diagram alone cannot establish
whether an array owns its storage.

## Use previews deliberately

```python
selection = rt.index((1_000_000, 1_000_000), (Ellipsis,))
selection.selected.count         # 1_000_000_000_000
selection.selected[0]            # (0, 0)
```

A shape tuple generates row-major placeholder values. It does not allocate
the corresponding tensor. Basic slice selections are compact and expose
`.count`, membership checks, and lazy iteration. Calling `list` explicitly
materializes all selected coordinates, so use it only for small selections.

The display budget and the arithmetic budget control different costs.
`max_visible_cells` limits the drawn cells in each panel. For sums, means,
matmul, and einsum, `max_terms` limits arithmetic terms per output element,
and `max_total_terms` limits their total across the visible output panel.
For a multi-axis reduction, one output uses the product of the reduced axis sizes.
When either limit is exceeded, outputs show `?` and the explanation says that
values were not evaluated. A partial sum is never presented as a full result.

```python
rt.sum((2, 100_000), axis=1)  # show the relation without the long sum
rt.sum(a, axis=1, max_terms=None, max_total_terms=None)  # remove both limits
```

For runnable walkthroughs, open `examples/07_explaining_outputs.ipynb` and
`examples/10_reduction_axes.ipynb` in Jupyter.
