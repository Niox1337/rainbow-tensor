# Reductions and math

These views collapse or contract axes. The source and the result sit side by
side so the reduced or contracted axis is easy to see.

```python
import numpy as np
import rainbow_tensor as rt
```

## Sum and mean

`sum` and `mean` collapse one axis. The source values that fold into the same
result element share one background colour with that result element, the first
group stays highlighted, and every surviving axis keeps its original colour.
`mean` divides each group by its count, so the result holds floats.

```python
rt.sum(np.arange(24).reshape(2, 3, 4), 0)
rt.mean(np.arange(24).reshape(2, 3, 4), 1)
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
