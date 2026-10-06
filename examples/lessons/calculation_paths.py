"""Select a denominator path and follow its own ordered row contributions."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

scores = np.array([[2., 4., 6.], [3., 6., 9.]])
flow = rt.Flow()
source = flow.input(scores, name="Scores")
totals = flow.sum(source, axis=1, keepdims=True, name="Totals")
normalized = flow.divide(source, totals, name="Normalized")
lesson = rt.walkthrough(normalized, focus=(1, 2))
assert normalized.value((1, 2)) == (scores / scores.sum(axis=1, keepdims=True))[1, 2]
assert lesson.snapshot.factor_values == (9., 18.)
assert lesson.snapshot.output_value == 0.5
display(lesson)

denominator = next(path for path in lesson.paths if path.role == "denominator")
lesson.select_path(denominator.occurrence)
assert lesson.snapshot.step.operation == "sum"
assert lesson.snapshot.output_value == 18
lesson.select_term(2)
assert lesson.snapshot.subtotal == 3 + 6 + 9
display(lesson)

# One original value occurs as numerator and again inside the denominator.
trace = lesson.snapshot.trace
assert next(root.count for root in trace.roots if root.reference == source.ref((1, 2))) == 2
lesson.set_focus((0, 0))
assert lesson.snapshot.occurrence == 0
assert lesson.snapshot.output_value == 2 / 12
display(lesson)
