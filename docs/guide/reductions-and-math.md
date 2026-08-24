# Reductions and math

These views collapse or contract axes. The source and the result sit side by
side so the reduced or contracted axis is easy to see.

```python
import numpy as np
from IPython.display import display

import rainbow_tensor as rt
```

## Sum and mean

Choose which axes to collapse with `axis`. It accepts one integer, a tuple of
integers, or `None` to reduce every axis. Leaving it out also reduces every axis.
Negative axes count from the end, so `axis=-1` selects the last axis. Each axis
may appear only once in a tuple.

```python
x = np.arange(24).reshape(2, 3, 4)
display(rt.sum(x, axis=1))  # result shape (2, 4)
display(rt.mean(x, axis=(0, 2)))  # result shape (3,)
display(rt.sum(x))  # scalar result, shape ()
```

The source values that fold into one result element share its background colour.
The focused group stays highlighted, and surviving axes keep their source colours.
`mean` divides each group by the number of contributing values. Reducing axes
`(0, 2)` of `x` uses `2 * 4 = 8` values per output, so its divisor is 8.

### Keep reduced axes for broadcasting

Pass `keepdims=True` as a keyword to leave reduced axes at length one. For `x`,
the shapes are:

| `axis` | Default result | With `keepdims=True` |
| --- | --- | --- |
| `None` | `()` | `(1, 1, 1)` |
| `1` | `(2, 4)` | `(2, 1, 4)` |
| `-1` | `(2, 3)` | `(2, 3, 1)` |
| `(0, 2)` | `(3,)` | `(1, 3, 1)` |
| `()` | `(2, 3, 4)` | `(2, 3, 4)` |

The retained reduced axes use the accent colour. Keeping those size-one axes
helps a reduction result line up with its original array. For example, divide
each row by its own total to make that row sum to one:

```python
scores = np.array([[2., 4., 6.], [3., 6., 9.]])
display(rt.sum(scores, axis=1, keepdims=True, focus=(1, 0)))

row_totals = scores.sum(axis=1, keepdims=True)  # shape (2, 1), values [[12.], [18.]]
display(rt.broadcast(scores, row_totals))     # (2, 3) and (2, 1) align by row
normalized = scores / row_totals
np.testing.assert_allclose(normalized.sum(axis=1), [1., 1.])
```

Without `keepdims`, the totals have shape `(2,)`. Broadcasting compares axes
from the right, and its size 2 cannot align with the 3 columns. The extra
length-one axis makes each total stretch across the columns of its own row.
Rainbow Tensor draws the calculation. NumPy performs the division above.

### Reduce no axes

An empty axis tuple means that no axes are reduced. `rt.sum(x, axis=())` and
`rt.mean(x, axis=())` preserve the shape and each element's value. Each output
has one source contribution, and the mean divisor is 1. This says nothing about
preserving a backend's dtype, since previews use the numerical model below.

`axis=()` does not mean an empty input. Sources still need at least one axis
and a positive size on every axis. Scalar and zero-sized source arrays are not
supported, although reducing all axes of a supported source gives a scalar result.

Reduction details are available without parsing the explanation:

```python
visual = rt.mean(x, axis=(-3, -1), keepdims=True)
visual.metadata["reduction"]
# {"axes": (0, 2), "keepdims": True, "term_count": 8}
```

## Matmul

`matmul` draws `a @ b` as the two operands and the output. The shared inner axis
is marked in the accent colour, and the row and column that combine into the
first output element are highlighted, so the contraction reads straight off the
figure. Vector, matrix, and batched products all work, and the batch axes
broadcast like `numpy.matmul`.

```python
a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(3, 4)

rt.matmul(a, b)
rt.matmul(np.arange(3), b)
rt.matmul(np.ones((5, 2, 3)), np.ones((5, 3, 4)))
```

## Einsum

`einsum` turns a subscript expression into labelled operand panels and an output
panel. Every label is coloured by its role, so free, shared, and contracted
labels read as three distinct groups, and a label keeps that colour across every
operand and the output. An ellipsis stands for the broadcast axes a subscript
leaves unnamed.

```python
rt.einsum("ij,jk->ik", a, b)
rt.einsum("abc,cde,ef->abdf", (2, 2, 2), (2, 2, 2), (2, 2))
rt.einsum("...ij,...jk->...ik", (2, 2, 3), (2, 3, 4))
```

The output shape is derived from the free labels and checked with the same size
rules as NumPy. A dedicated `matmul` view exists for the common matrix multiply
that `einsum("ik,kj->ij", a, b)` also expresses.

## Focus on one output

Pass `focus=` to `sum`, `mean`, `matmul`, or `einsum` to choose the output
whose contributing source cells are highlighted.

```python
visual = rt.matmul(a, b, focus=(1, 2))
visual
visual.trace.output_coord       # (1, 2)
visual.trace.term_count         # 3
visual.trace.terms              # ordered references to the three products

rt.sum(a, 1, focus=(-1,))        # follow the last row's sum
rt.einsum("ij,jk->ik", a, b, focus=(1, 2))
```

Coordinates are tuples in the output shape. Negative entries count from the
end, and a scalar output uses `focus=()`. Invalid coordinates fail before
reading array values.

For reductions, `keepdims` also changes the focus tuple. Reducing axes `(0, 2)`
of `x` gives shape `(3,)`, so use `focus=(1,)`. Keeping the axes gives shape
`(1, 3, 1)`, so the same group is `focus=(0, 1, 0)`.

```python
visual = rt.mean(x, axis=(2, 0), focus=(1,))
visual.trace.term_count         # 8
visual.trace.divisor            # 8
# Source values: 4, 5, 6, 7, 16, 17, 18, 19. Their mean is 11.5.
```

Reduction traces follow source row-major coordinate order, with the last reduced
source axis changing fastest. Reordering the axis tuple does not reorder the
trace. Here `(2, 0)`, `(0, 2)`, and `(-3, -1)` all describe the same groups.

Every math visual exposes an immutable `OutputTrace`. Its `terms` contain
`OperandRef` objects naming the operand number and source coordinate.
Multiply references in each term, add the terms, then divide by `divisor`
for a mean. The trace preserves repeated factors caused by broadcasting.
It stores at most eight terms and reports `complete=False` when more exist.
This bounded coordinate description does not evaluate the numeric result.

## Understand the numerical model

These figures explain coordinates and arithmetic using Python scalar values.
They do not call a framework's native reduction or matrix multiplication kernel.
Backend accumulation dtype, rounding, and overflow can therefore differ.

```python
x = np.array([1e8, 1, -1e8], dtype=np.float32)
visual = rt.sum(x, axis=0)
visual                           # Python-scalar preview: 1.0
np.sum(x)                        # NumPy float32 accumulation: 0.0
visual.metadata["numeric_semantics"]
```

The metadata reports `arithmetic="python_scalar"` and one `operand_sources`
entry per operand. `array_values` means values read from the input array.
`generated_values` means row-major placeholders generated from a shape tuple.
The figure states this model even when a computation is skipped. Use the original
framework operation when you need its native numerical result.

## Separate display and computation limits

Reductions evaluate requested output groups on demand and reuse their values
within one render. Hidden output groups do not trigger reads. Matmul and einsum
also compute values on demand.

All four functions accept `max_terms=10_000`, the maximum number of source
contributions or products to sum for each output cell, and
`max_total_terms=100_000`, the maximum across the visible output panel.
For a multi-axis sum or mean, the per-output term count is the product of the
reduced axis sizes. `keepdims` does not change that count. Reducing no axes
with `axis=()` costs one term per output.
The layout is planned before reading values. If either limit is exceeded, every
output value displays `?`. No partial arithmetic result is used, and changing
the order in which cells are rendered cannot change the budget decision.
Set either limit to `None` to remove that limit, or set both to `None` to remove
both limits.

```python
visual = rt.einsum("ij,jk->ik", (2, 1_000_000), (1_000_000, 2), focus=(1, 1))
visual.metadata["value_evaluation"]["status"]  # "skipped"
visual.trace.term_count                       # 1_000_000

visual = rt.sum((20, 10_000), axis=1)
visual.metadata["value_evaluation"]["reason"]  # "max_total_terms"
```

When a contraction is skipped, source highlights may show only the bounded
trace sample. The explanation states that sampling was used. Source cells that
are actually visible are still read to draw the input panels.

The metadata records `output_count`, the number of planned visible outputs,
and `total_terms`, their combined calculation cost. Its `scope` is
`per_output_cell_and_preview`, and `reason` identifies the exceeded limit or is
`None` when values can be evaluated. These counts describe the plan, even if a
custom renderer reads fewer cells. Custom renderers may choose different output
coordinates within the planned cell count. Repeated requests are cached. Extra
distinct coordinates return `?` and cannot add unbudgeted work.

These are calculation limits, not memory or elapsed-time guarantees. A term in
a multi-operand einsum may read several operands. Reading displayed input cells
is separate from calculating outputs. The theme option `max_visible_cells`
limits the number of displayed cells in each panel.

For an executable walkthrough of axis choices and row normalization, open
`examples/10_reduction_axes.ipynb` in Jupyter.
