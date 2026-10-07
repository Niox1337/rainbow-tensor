"""Separate a mean's subtotal from its divisor, then inspect a dot product."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

x = np.arange(1, 7).reshape(2, 3)
mean_lesson = rt.walkthrough(rt.mean, x, axis=1, focus=(1,))
assert mean_lesson.snapshot.numeric_complete
assert mean_lesson.snapshot.factor_values == (4,)
assert mean_lesson.snapshot.subtotal == 4
assert mean_lesson.snapshot.output_value == x.mean(axis=1)[1] == 5
display(mean_lesson.visual)

mean_lesson.next_term()
assert mean_lesson.snapshot.term == 1
assert mean_lesson.snapshot.factor_values == (5,)
assert mean_lesson.snapshot.subtotal == 9
mean_lesson.select_term(2)
assert mean_lesson.snapshot.subtotal == 15
assert mean_lesson.snapshot.step.divisor == 3
assert mean_lesson.snapshot.output_value == 15 / 3
display(mean_lesson.visual)

# A new output starts again at its first contribution.
mean_lesson.set_focus((0,))
assert mean_lesson.snapshot.term == 0
assert mean_lesson.snapshot.factor_values == (1,)
assert mean_lesson.snapshot.output_value == 2
display(mean_lesson)

a = np.array([[1, 2]])
b = np.array([[3], [4]])
product_lesson = rt.walkthrough(rt.matmul, a, b, focus=(0, 0))
assert product_lesson.snapshot.factor_values == (1, 3)
assert product_lesson.snapshot.term_value == 3
product_lesson.next_term()
assert product_lesson.snapshot.factor_values == (2, 4)
assert product_lesson.snapshot.term_value == 8
assert product_lesson.snapshot.subtotal == (a @ b)[0, 0] == 11
display(product_lesson)
