"""Follow a numerator and its row total through an explicitly recorded division."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

scores = np.array([[2., 4., 6.], [3., 6., 9.]])
expected = scores / scores.sum(axis=1, keepdims=True)

flow = rt.Flow()
source = flow.input(scores, name="Scores")
totals = flow.sum(source, axis=1, keepdims=True, name="Totals")
normalized = flow.divide(source, totals, name="Normalized")
assert totals.shape == (2, 1)
assert normalized.shape == expected.shape == (2, 3)
assert normalized.value((1, 2)) == expected[1, 2] == 9 / 18 == 0.5
np.testing.assert_allclose(expected.sum(axis=1), [1., 1.])

trace = normalized.trace((1, 2))
assert trace.complete
assert trace.steps[0].binary_operator == "divide"
assert {root.reference.coordinate: root.count for root in trace.roots} == {
    (1, 0): 1, (1, 1): 1, (1, 2): 2,
}
normalization_lesson = rt.walkthrough(normalized, focus=(1, 2))
assert normalization_lesson.snapshot.factor_values == (9., 18.)
assert normalization_lesson.snapshot.binary_operator == "divide"
assert normalization_lesson.snapshot.term_value == 0.5
display(normalization_lesson.visual)

# Inspect the denominator's own sum without flattening the surrounding division.
denominator = next(
    index for index, step in enumerate(normalization_lesson.snapshot.trace.steps)
    if step.operation == "sum"
)
normalization_lesson.select_occurrence(denominator)
normalization_lesson.select_term(2)
assert normalization_lesson.snapshot.subtotal == 3 + 6 + 9 == 18
assert normalization_lesson.snapshot.output_value == 18
display(normalization_lesson.visual)

normalization_lesson.set_focus((0, 0))
assert normalization_lesson.snapshot.occurrence == 0
assert normalization_lesson.snapshot.binary_operator == "divide"
assert normalized.value((0, 0)) == 2 / 12
display(normalization_lesson)

