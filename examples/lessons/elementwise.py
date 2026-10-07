"""Match broadcast source coordinates while keeping binary operand order."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

a = np.array([[10], [20]])
b = np.array([[1, 2, 3]])
expected = a + b
addition = rt.explore(rt.add, a, b, focus=(1, 2))
expression = addition.visual.trace.expression
assert addition.result_shape == expected.shape == (2, 3)
assert expression.operator == "add"
assert [(ref.operand, ref.coordinate) for ref in expression.operands] == [
    (0, (1, 0)), (1, (0, 2)),
]
assert addition.visual.trace.terms == ()
assert expected[1, 2] == a[1, 0] + b[0, 2] == 23
display(addition.visual)

addition.set_focus((0, 2))
assert addition.visual.trace.expression.operands[0].coordinate == (0, 0)
assert expected[0, 2] == 13
display(addition)

# A literal is a scalar value. A tuple instead describes generated tensor data.
halved = rt.divide(a, 2, focus=(1, 0))
assert halved.trace.expression.operator == "divide"
assert halved.trace.expression.operands[1].coordinate == ()
display(halved)

flow = rt.Flow()
left = flow.input(a, name="A")
right = flow.input(b, name="B")
two = flow.input(2, name="Two")
assert flow.add(left, right).value((1, 2)) == (a + b)[1, 2] == 23
assert flow.subtract(left, right).value((1, 2)) == (a - b)[1, 2] == 17
assert flow.multiply(left, right).value((1, 2)) == (a * b)[1, 2] == 60
assert flow.divide(left, two).value((1, 0)) == (a / 2)[1, 0] == 10
