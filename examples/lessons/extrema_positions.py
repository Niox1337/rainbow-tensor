"""Distinguish a largest value from the position that supplies it, including ties."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

values = np.array([[2, 8, 8], [7, 7, 3]])
flow = rt.Flow()
source = flow.input(values, name="Values")
largest = flow.max(source, axis=1, name="Largest")
positions = flow.argmax(source, axis=1, name="Positions")
assert [largest.value((i,)) for i in range(2)] == values.max(axis=1).tolist() == [8, 7]
assert [positions.value((i,)) for i in range(2)] == values.argmax(axis=1).tolist() == [1, 0]
lesson = rt.walkthrough(positions, focus=(0,))
assert lesson.snapshot.selected_source.source == source.ref((0, 1))
assert lesson.snapshot.selected_source.reason == "first_max"
assert lesson.snapshot.output_value == 1
display(lesson)

lesson.set_focus((1,))
assert lesson.snapshot.selected_source.source == source.ref((1, 0))
assert lesson.snapshot.output_value == 0
display(lesson)

with_nan = flow.input(np.array([1., np.nan, np.nan]), name="WithNaN")
nan_position = flow.argmax(with_nan, name="FirstNaN")
assert nan_position.value(()) == np.argmax(np.array([1., np.nan, np.nan])) == 1
nan_lesson = rt.walkthrough(nan_position)
assert nan_lesson.snapshot.selected_source.reason == "first_nan"
assert nan_lesson.snapshot.selected_source.source == with_nan.ref((1,))
display(nan_lesson)
