"""Predict a row sum, then select the other row and check its sources."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

x = np.arange(1, 7).reshape(2, 3)
result = x.sum(axis=1)
assert result.tolist() == [6, 15]

explorer = rt.explore(rt.sum, x, axis=1, focus=(1,))
assert explorer.visual.result_shape == result.shape == (2,)
assert [term[0].coordinate for term in explorer.visual.trace.terms] == [
    (1, 0), (1, 1), (1, 2),
]
assert result[1] == x[1, 0] + x[1, 1] + x[1, 2] == 15
display(explorer)

# Predict which row will be highlighted before changing the output coordinate.
explorer.set_focus((0,))
assert explorer.focus == (0,)
assert [term[0].coordinate for term in explorer.visual.trace.terms] == [
    (0, 0), (0, 1), (0, 2),
]
assert result[0] == 1 + 2 + 3 == 6
display(explorer)

