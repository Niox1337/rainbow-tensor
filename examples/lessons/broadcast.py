"""Locate both expanded operands before checking a NumPy addition."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

a = np.array([[10], [20]])
b = np.array([[1, 2, 3]])
result = a + b
assert result.tolist() == [[11, 12, 13], [21, 22, 23]]

left = rt.explore(rt.broadcast, a, b, focus=(1, 2), focus_operand=0)
right = rt.explore(rt.broadcast, a, b, focus=(1, 2), focus_operand=1)
assert left.result_shape == right.result_shape == result.shape == (2, 3)
assert left.visual.trace.terms[0][0].operand == 0
assert left.visual.trace.terms[0][0].coordinate == (1, 0)
assert right.visual.trace.terms[0][0].operand == 1
assert right.visual.trace.terms[0][0].coordinate == (0, 2)
assert result[1, 2] == a[1, 0] + b[0, 2] == 23
display(left)
display(right)

# A new result row changes only the left operand's source row.
left.set_focus((0, 2))
right.set_focus((0, 2))
assert left.visual.trace.terms[0][0].coordinate == (0, 0)
assert right.visual.trace.terms[0][0].coordinate == (0, 2)
assert result[0, 2] == 10 + 3 == 13
display(left)
display(right)

# The broadcast views explain expansion. NumPy performs the addition above.
display(rt.shape(result))
