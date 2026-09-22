# Follow an element through several operations

A final tensor value can hide several steps. An index chooses a source,
transpose changes its coordinate, and sum combines it with other values.
`Flow` records those steps so a selected result can be followed back to its
original inputs.

## Record a short chain

This input has two rows:

```python
import numpy as np
import rainbow_tensor as rt

x = np.arange(1, 7).reshape(2, 3)
# [[1, 2, 3],
#  [4, 5, 6]]

flow = rt.Flow()
source = flow.input(x, name="X")
selected = flow.index(source, ([1, 0, 1], slice(None, None, -1)), name="S")
transposed = flow.transpose(selected, name="T")
y = flow.sum(transposed, axis=1, name="Y")
```

The index picks the second row, then the first row, then the second row again.
The reverse slice reads each selected row from right to left.

```text
S = [[6, 5, 4],     T = [[6, 3, 6],     Y = [15, 12, 9]
     [3, 2, 1],          [5, 2, 5],
     [6, 5, 4]]         [4, 1, 4]]
```

Before opening a figure, predict which original positions contribute to `Y[0]`.
Then inspect the selected output:

```python
assert y.shape == (3,)
assert y.value((0,)) == 15
visual = y.visualize(focus=(0,))
visual
```

The final sum has three terms:

```text
Y[0] = T[0, 0] + T[0, 1] + T[0, 2]
     = X[1, 2] + X[0, 2] + X[1, 2]
     = 6 + 3 + 6
```

There are three contributions but only two distinct original positions.
`X[1, 2]` participates twice. A trace keeps both paths through the repeated
gather, even when a panel draws that source cell only once.

`y` is a `TrackedTensor` with a logical `shape`. `visual` is the usual
`TensorVisual`, with `svg`, `text`, `save`, and `result_shape`. Its `trace`
describes the final operation. Its `provenance` contains the multi-step trace.

## Choose another result

Install `rainbow-tensor[interactive]` in the notebook kernel environment for
click and keyboard controls:

```python
explorer = rt.explore(y)
explorer
```

Click a final result cell, or use Enter or Space on it. The coordinate fields
also reach positions omitted by a large preview. `explorer.set_focus((1,))`
selects `Y[1]`, whose value is `5 + 2 + 5 = 12`. `Y[2]` is `4 + 1 + 4 = 9`.
Call `explorer.close()` when finished.

Without widgets, use `y.visualize(focus=(1,))`. Static SVG files retain their
highlights without a running kernel. Save `visual.text` separately to share the
plain-text explanation alongside the figure.

For a single operation, the same focus controls cover indexing, shape
transformations, combining, reductions, matmul and einsum:

```python
rt.reshape(x, (3, 2), focus=(2, 1))
rt.explore(rt.repeat, x, [1, 0, 2], axis=1)
```

Shapes and memory views continue to describe structure and storage. A logical
origin does not claim that a backend copied or shared the underlying memory.

## Inspect the coordinate history

Structural tracing never reads array values:

```python
trace = y.trace((0,))
assert trace.complete
counts = {root.reference.coordinate: root.count for root in trace.roots}
assert counts == {(1, 2): 2, (0, 2): 1}
```

`trace.steps` is an immutable tuple of occurrences. The first step describes
the selected output. Each step's `terms` contains ordered tuples of child
positions in `trace.steps`. Factors inside a term multiply together, and terms
are summed at that step. Each `ElementRef` identifies a node, an output port
and a coordinate. Input identity is part of the reference, so equal coordinates
in two different inputs remain distinct.

`trace.roots` counts reached paths to original inputs. These are participation
counts, not coefficients or derivatives. In a matrix product, a source value
also depends on the other factor in its term. In nested means, each division
belongs to its own operation. The trace retains those groups instead of
flattening them into one unweighted sum.

## Which operations can be recorded

| Operation family | Flow methods |
| --- | --- |
| Selection | `index`, `take`, `repeat` |
| Shape changes | `reshape`, `transpose`, `swapaxes`, `moveaxis`, `squeeze`, `expand_dims` |
| Multiple inputs | `concatenate`, `stack`, `broadcast` |
| Arithmetic | `sum`, `mean`, `matmul`, `einsum` |

Pass tracked tensors from the same Flow as operands. Names are optional but
must be unique within that Flow when provided. Ordinary arrays enter through
`flow.input`. An existing array's earlier history cannot be reconstructed from
its values, so record each step that should appear in the explanation.

Broadcast returns one output for each input, stretched to their common shape:

```python
left = flow.input(np.array([[2], [5]]), name="A")
right = flow.input(np.array([10, 20, 30]), name="B")
left_stretched, right_stretched = flow.broadcast(left, right, name="C")
assert left_stretched.shape == right_stretched.shape == (2, 3)
assert left_stretched.value((1, 2)) == 5
assert right_stretched.value((1, 2)) == 30
```

Both outputs share a recorded operation ID and have separate output ports.
They can be used independently in later steps.

Shape tuples are also accepted, such as `flow.input((2, 3), name="P")`.
Their values are generated row-major positions from zero, matching static
shape-only examples.

## Fixed recipes and live input values

Each step captures normalized shape, axis and selection parameters. Advanced
index arrays are copied. Changing an index list later does not change the
recorded recipe. Create a new step when the intended operation changes.

Input values remain live:

```python
x[1, 2] = 60
assert y.value((0,)) == 123  # 60 + 3 + 60
x[1, 2] = 6
```

Every value query and visual refresh reads the required input values again.
Shared intermediate elements are cached only within that request. Input shapes
must remain fixed, and a numerical request rejects a shape changed after
recording.

Evaluation uses Python scalar arithmetic, preserving the sum, multiplication
and mean division at each operation. It does not execute a backend kernel or
materialise intermediate tensors. Accumulation dtype, rounding and overflow
can differ from NumPy, PyTorch, JAX or TensorFlow kernels.

## Keep exploration bounded

Structural and numerical limits answer different questions:

| Setting | Default | What it limits |
| --- | --- | --- |
| `max_depth` | 6 | Operation edges followed from the selected output |
| `max_nodes` | 80 | Occurrences stored in the trace, including repeated paths |
| `max_edges` | 120 | References between those occurrences |
| `max_panels` | 8 | Tensors drawn in one visual |
| `max_terms` | 10,000 | Terms in any recursively required element |
| `max_total_terms` | 100,000 | Combined operation terms, factor references and input reads |

Structural queries report truncation explicitly:

```python
partial = y.trace((0,), max_depth=1)
assert not partial.complete
assert "max_depth" in partial.truncated_reasons
```

On an incomplete trace, root counts describe only reached paths. Missing paths
are not treated as absent contributions. `max_panels` limits the drawing
without changing the structural trace. Scalar results use `()`. Empty outputs
have no selectable coordinate and return an empty trace.

Numerical work is planned before any input values are read. If planning exceeds
a budget, `value` raises `rt.ValueBudgetExceeded` and a visual shows `?` for
values across all its panels. It never displays a partially computed result.

```python
try:
    y.value((0,), max_terms=2)
except rt.ValueBudgetExceeded as error:
    assert error.reason == "max_terms"
```

Passing `None` disables the corresponding numerical term budget. A separate
guard still limits numerical recursion to 64 operation edges. Preview cell
limits and structural trace limits remain independent. Index arrays and masks
must still be inspected to define their result shapes, and variable repeat
counts require storage proportional to the supplied counts.

The complete worked notebook is
[15_operation_origins.ipynb](../../examples/15_operation_origins.ipynb).
