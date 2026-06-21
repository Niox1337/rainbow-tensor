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
