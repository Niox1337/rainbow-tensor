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
`max_terms` budget. Selecting another output never raises that budget. A value
that was skipped remains a question mark until you explicitly request a larger
budget when creating the explorer.

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
