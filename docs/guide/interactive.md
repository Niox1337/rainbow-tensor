# Explore one output in a notebook

Install the optional controls in the same environment as your notebook kernel:

```text
pip install "rainbow-tensor[interactive]"
```

Pass a mathematical operation and its usual inputs to `explore`:

```python
import numpy as np
import rainbow_tensor as rt

a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(3, 4)
explorer = rt.explore(rt.matmul, a, b, focus=(1, 2))
explorer
```

Edit an output coordinate with the keyboard and press **Update focus**.
The source highlights and coordinate formula follow that output. Coordinates
start at zero. `sum`, `mean`, and `einsum` work the same way. A scalar output
has no coordinate fields, only the update button.

The controls reuse the static operations, including their numerical model and
`max_terms` and `max_total_terms` budgets. Each update plans the new visible
outputs under the same limits. Selecting another output never raises either
limit. If the new plan exceeds a limit, output values appear as question marks.

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
kernel and a notebook host that supports Jupyter widgets. The optional dependency
is imported only when `explore` is called.
