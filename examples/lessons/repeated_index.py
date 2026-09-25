"""Follow distinct result positions that read the same reversed source cell."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

x = np.arange(1, 7).reshape(2, 3)
selection = ([1, 0, 1], slice(None, None, -1))
result = x[selection]
assert result.tolist() == [[6, 5, 4], [3, 2, 1], [6, 5, 4]]

explorer = rt.explore(rt.index, x, selection, focus=(0, 0))
assert explorer.visual.result_shape == result.shape
assert explorer.visual.trace.terms[0][0].coordinate == (1, 2)
display(explorer)

# The second copy changes the selected output, but keeps the original source.
explorer.set_focus((2, 0))
assert explorer.focus == (2, 0)
assert explorer.visual.trace.terms[0][0].coordinate == (1, 2)
assert result[0, 0] == result[2, 0] == x[1, 2] == 6
display(explorer)

explorer.set_focus((1, 0))
assert explorer.visual.trace.terms[0][0].coordinate == (0, 2)
assert result[1, 0] == x[0, 2] == 3
display(explorer)

