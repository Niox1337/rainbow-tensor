# Explore one output in a notebook

Notebook controls are part of the standard installation. Install the package
in the same environment as your notebook kernel:

```text
python -m pip install rainbow-tensor
```

The required `ipywidgets` and `anywidget` packages are installed automatically.
Live controls also need a running kernel and a notebook host with widget support.

Pass an operation and its usual inputs to `explore`:

```python
import numpy as np
import rainbow_tensor as rt

a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(3, 4)
explorer = rt.explore(rt.matmul, a, b, focus=(1, 2))
explorer
```

Click a visible output cell to update its source highlights and coordinate
formula. A cell can also receive keyboard focus and be activated with Enter or
Space. Coordinate fields and **Update focus** reach positions hidden by the
preview. Coordinates start at zero. A scalar output has one clickable cell and
no coordinate fields.

`index`, `sum`, `mean`, `einsum`, shape transformations and combining operations
use the same controls. For example:

```python
reshape_explorer = rt.explore(rt.reshape, a, (3, 2))
repeat_explorer = rt.explore(rt.repeat, a, [1, 0, 2], axis=1)
broadcast_explorer = rt.explore(rt.broadcast, (2, 1), (1, 3), focus_operand=1)
```

For broadcast, `focus_operand=0` or `1` chooses which stretched output can be
selected. Shapes and memory views describe structure and storage without an
output operation to trace, so they remain static.

For mathematical operations, the controls reuse the numerical model and
`max_terms` and `max_total_terms` budgets. Each update plans the new visible
outputs under the same limits. Selecting another output never raises either
limit. If the new plan exceeds a limit, output values appear as question marks.

## Follow an indexed output back to its source

Repeated indices create separate output positions that can share one source
element. A reverse slice also changes the order. Try both together:

```python
x = np.arange(12).reshape(3, 4)
selection = ([2, 0, 2], slice(None, None, -2))
index_explorer = rt.explore(rt.index, x, selection)
index_explorer
```

The result is `[[11, 9], [3, 1], [11, 9]]`. The explorer starts at output
`(0, 0)`, which reads source `(2, 3)`. Output `(2, 0)` reads that same source
element because row 2 was picked twice. Click either result cell, or type its
coordinate into the fields and press **Update focus**, to follow the connection.

You can also change focus from Python:

```python
index_explorer.set_focus((1, 1))
index_explorer.visual.trace.terms[0][0].coordinate  # (0, 1)
index_explorer.visual.index_mapping.source_coord((2, 0))  # (2, 3)
```

The trace describes the current output. The index mapping can answer other
output coordinates without changing the focus. Use `index_explorer.close()`
when finished with the controls.

For a script or a host without live widget support, use the same focused figure directly:

```python
rt.index(x, selection, focus=(1, 1))
```

Passing `focus` shows the source and result side by side and highlights their
connection. A normal `rt.index(x, selection)` call keeps its original static
source-selection view. The complete walkthrough in
[14_index_explorer.ipynb](../../examples/14_index_explorer.ipynb) includes a
static result alongside the notebook controls.

## Explore a reduction that keeps its axes

The coordinate fields follow the result shape, including any size-one axes
retained by `keepdims=True`:

```python
x = np.arange(24).reshape(2, 3, 4)
reduction_explorer = rt.explore(
    rt.mean, x, axis=(0, 2), keepdims=True, focus=(0, 1, 0)
)
reduction_explorer
```

The result shape is `(1, 3, 1)`. Only the middle coordinate can change, while
the other two stay at 0. Each output averages `2 * 4 = 8` source values.

```python
reduction_explorer.set_focus((0, 2, 0))
reduction_explorer.visual.trace.divisor  # 8
```

Without `keepdims`, the same operation has shape `(3,)` and uses `focus=(2,)`.
With `axis=None`, the default, all axes reduce to a scalar unless they are kept.
With `axis=()`, no axes are reduced and the fields follow the original shape.
Call `reduction_explorer.close()` when finished with this example.

## Scalar outputs and empty outputs

A scalar output has one addressable value at `()`. An empty output has none.
For example, the source below has shape `(2, 0, 3)`, so reducing its last axis
leaves an empty result of shape `(2, 0)`:

```python
empty_source = np.empty((2, 0, 3))
empty_explorer = rt.explore(rt.sum, empty_source, axis=2)
assert empty_explorer.focus is None
assert empty_explorer.visual.trace is None
empty_explorer
```

There are no coordinate fields, and **Update focus** is disabled. Calling
`set_focus` raises `IndexError` because no coordinate exists. The empty figure
is still available through `empty_explorer.visual` and can be saved as SVG.
Close the controls with `empty_explorer.close()` when finished.

Reducing axis 1 instead gives shape `(2, 3)`. Those six outputs can be selected
normally, even though each combines zero source values. They contain 0 for
`sum` and NaN for `mean`.

## Keep a static result

The latest successful result is a normal `TensorVisual`:

```python
explorer.set_focus((-1, -1))
explorer.visual.trace
explorer.visual.save("focused-output.svg")
```

Python updates accept negative coordinates, just like `focus`. Invalid updates
preserve the last successful figure. Button errors appear beside the controls.
Inputs are read again on each update, so edits to array values are reflected.
Keep their shape fixed while an explorer is open, or create a new explorer.

Call `explorer.close()` when finished to release its widget communications.
Static SVG export works without a running kernel. Live controls require an active
kernel and a notebook host that supports Jupyter widgets. The required widget
packages are loaded only when `explore` is called. If either package is missing,
repair the installation in the notebook kernel's environment.

## Explore a recorded chain

Pass a tracked tensor directly to keep its earlier operations available:

```python
flow = rt.Flow()
source = flow.input(np.arange(1, 7).reshape(2, 3), name="X")
selected = flow.index(source, ([1, 0, 1], slice(None, None, -1)), name="S")
y = flow.sum(flow.transpose(selected, name="T"), axis=1, name="Y")
chain_explorer = rt.explore(y, focus=(0,))
chain_explorer
```

Click one of the final `Y` cells to follow it through the index, transpose and
sum. `Y[0]` is `6 + 3 + 6 = 15`. The same original position participates twice.
The figure keeps local operation equations and reports repeated input paths.

```python
chain_explorer.set_focus((1,))
chain_explorer.visual.provenance.complete  # True for this small example
chain_explorer.visual.save("operation-origins.svg")
```

The latest figure remains a `TensorVisual`. Its `trace` describes the final
operation and its `provenance` describes the bounded chain. A static equivalent
is `y.visualize(focus=(1,))`.

Flow captures index and axis parameters when each step is recorded. Updating
the explorer rereads input array values while keeping those recipes fixed.
Record a new step to use changed selection parameters. This differs from
`rt.explore(rt.index, array, selection)`, which rebuilds its direct index mapping
on each update. In both cases, input shapes must stay fixed.

The default SVG renderer supplies clickable cell metadata. A custom SVG
renderer without that metadata can still use the coordinate controls.
Static figures remain available for export or hosts without live widget support.
See [cross-operation origins](provenance.md) for structural and recursive value
limits. Close the example with
`chain_explorer.close()` when finished.
