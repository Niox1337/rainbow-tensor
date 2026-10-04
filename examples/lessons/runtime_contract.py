"""Check the installed teaching surface before choosing a visual explanation."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

contract = rt.capabilities()
assert contract["package"]["version"] == rt.__version__
assert contract["operations"]["divide"]["explore"]
assert contract["operations"]["divide"]["flow_signature"] is not None
assert "softmax" not in contract["operations"]
assert contract["semantics"]["structural_trace_reads_values"] is False

values = np.array([[1, 2, 3], [4, 5, 6]])
result = values.sum(axis=1)
explorer = rt.explore(rt.sum, values, axis=1, focus=(1,))
assert explorer.visual.result_shape == result.shape == (2,)
assert explorer.visual.trace.output_coord == (1,)
assert explorer.visual.provenance is None
display(explorer)

# A static view of a computed array cannot recover its earlier operation history.
flow = rt.Flow()
source = flow.input(values, name="Values")
selected = flow.index(source, ([1, 0, 1], 2), name="Selected")
total = flow.sum(selected, name="Total")
trace = total.trace(())
assert trace.complete
assert total.value(()) == values[[1, 0, 1], 2].sum() == 15
assert {root.reference.coordinate: root.count for root in trace.roots} == {
    (1, 2): 2, (0, 2): 1,
}
lesson = rt.walkthrough(total)
display(lesson)
