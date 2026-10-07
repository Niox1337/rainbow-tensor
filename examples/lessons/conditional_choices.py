"""Separate comparison inputs and branch candidates from a chosen output source."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

values = np.array([[1, 4, 2], [8, 3, 7]])
expected = np.where(values > 3, values, 0)
flow = rt.Flow()
source = flow.input(values, name="Values")
threshold = flow.input(3, name="Threshold")
zero = flow.input(0, name="Zero")
condition = flow.greater(source, threshold, name="AboveThreshold")
chosen = flow.where(condition, source, zero, name="Chosen")
total = flow.sum(chosen, name="Total")
assert chosen.shape == expected.shape
assert total.value(()) == expected.sum() == 19

lesson = rt.walkthrough(chosen, focus=(1, 0))
assert lesson.snapshot.step.selection_operator == "where"
assert lesson.snapshot.step.terms == ()
assert lesson.snapshot.selected_source.source == source.ref((1, 0))
assert lesson.snapshot.selected_source.reason == "condition_true"
assert lesson.snapshot.output_value == expected[1, 0] == 8
display(lesson)

lesson.set_focus((1, 1))
assert lesson.snapshot.selected_source.source == zero.ref(())
assert lesson.snapshot.selected_source.reason == "condition_false"
assert lesson.snapshot.output_value == expected[1, 1] == 0
assert {path.role for path in lesson.paths} >= {"condition", "when_true", "when_false"}
display(lesson)

# The structural tree contains alternatives. It is not a list of chosen values.
assert len(chosen.trace((1, 1)).steps[0].candidates) == 3
display(total.visualize())
