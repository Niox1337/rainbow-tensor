"""Keep duplicate contributions while following a chain back to its input."""

from pathlib import Path

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

x = np.arange(1, 7).reshape(2, 3)
selection = ([1, 0, 1], slice(None, None, -1))
expected = x[selection].T.sum(axis=1)

flow = rt.Flow()
source = flow.input(x, name="X")
selected = flow.index(source, selection, name="S")
transposed = flow.transpose(selected, name="T")
y = flow.sum(transposed, axis=1, name="Y")
assert y.shape == expected.shape == (3,)
assert [y.value((i,)) for i in range(3)] == expected.tolist() == [15, 12, 9]

trace = y.trace((0,))
assert trace.complete
assert {root.reference.coordinate: root.count for root in trace.roots} == {
    (1, 2): 2, (0, 2): 1,
}
assert y.value((0,)) == x[1, 2] + x[0, 2] + x[1, 2] == 6 + 3 + 6
explorer = rt.explore(y, focus=(0,))
display(explorer)

explorer.set_focus((1,))
assert explorer.visual.trace.output_coord == (1,)
assert explorer.visual.provenance.complete
assert y.value((1,)) == 5 + 2 + 5 == 12
assert {root.reference.coordinate: root.count for root in explorer.visual.provenance.roots} == {
    (1, 1): 2, (0, 1): 1,
}
display(explorer)

# Export the current figure and its accompanying explanation separately.
visual = explorer.visual
visual.save("operation-origins.svg")
Path("operation-origins.txt").write_text(visual.text, encoding="utf-8")
assert Path("operation-origins.svg").read_text(encoding="utf-8") == visual.svg
assert Path("operation-origins.txt").read_text(encoding="utf-8") == visual.text
