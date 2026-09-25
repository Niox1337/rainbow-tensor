"""Distinguish one scalar, no output elements, and empty reduction groups."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

scalar = rt.explore(rt.sum, np.array(7), focus=())
assert scalar.result_shape == scalar.focus == ()
assert scalar.visual.trace.terms[0][0].coordinate == ()
assert not scalar.update_button.disabled
display(scalar)

empty = np.empty((2, 0, 3))
no_outputs = rt.explore(rt.sum, empty, axis=2)
assert no_outputs.result_shape == (2, 0)
assert no_outputs.focus is None
assert no_outputs.visual.trace is None
assert no_outputs.update_button.disabled
display(no_outputs)

empty_sums = rt.explore(rt.sum, empty, axis=1, focus=(1, 2))
assert empty_sums.result_shape == (2, 3)
assert empty_sums.visual.trace.term_count == 0
assert np.sum(empty, axis=1)[1, 2] == 0
display(empty_sums)

flow = rt.Flow()
source = flow.input(empty, name="Empty")
means = flow.mean(source, axis=1, name="Means")
assert means.shape == (2, 3)
assert np.isnan(means.value((1, 2)))
assert means.trace((1, 2)).steps[0].term_count == 0
display(means.visualize(focus=(1, 2)))

