"""Distinguish logical Python scalar arithmetic from NumPy storage arithmetic."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

small = np.array([250], dtype=np.uint8)
increment = np.array([10], dtype=np.uint8)
stored = np.add(small, increment)
flow = rt.Flow()
left = flow.input(small, name="StoredByte")
right = flow.input(increment, name="Increment")
logical = flow.add(left, right, name="LogicalSum")
assert stored.dtype == np.uint8
assert int(stored[0]) == 4
assert logical.value((0,)) == 260
assert rt.capabilities()["semantics"]["arithmetic"] == "python_scalars"
display(logical.visualize(focus=(0,)))

# A preview deliberately does not inherit NumPy's floating divide-by-zero policy.
with np.errstate(divide="ignore", invalid="ignore"):
    numpy_zero = np.divide(np.array([1.0, 0.0]), 0.0)
assert np.isinf(numpy_zero[0]) and np.isnan(numpy_zero[1])
try:
    rt.divide(np.array([1.0]), 0.0)
except ZeroDivisionError:
    pass
else:
    raise AssertionError("A Python scalar preview must reject a zero denominator")

safe = rt.explore(rt.divide, np.array([1.0, 0.0]), 2.0, focus=(0,))
assert safe.visual.trace.expression.operator == "divide"
display(safe)
