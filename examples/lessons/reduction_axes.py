"""Predict how a tuple of reduction axes and keepdims change output coordinates."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

x = np.arange(24).reshape(2, 3, 4)
playground = rt.reduction_playground(rt.sum, x, axis=(0, 2), focus=(1,))
assert not playground.revealed
playground.prediction.value = "(3,)"
playground.reveal()
assert playground.visual.result_shape == x.sum(axis=(0, 2)).shape == (3,)
assert playground.visual.trace.output_coord == (1,)
assert playground.visual.trace.term_count == 8
assert x.sum(axis=(0, 2))[1] == 4 + 5 + 6 + 7 + 16 + 17 + 18 + 19 == 92
display(playground.visual)

playground.set_parameters(keepdims=True)
assert not playground.revealed
playground.prediction.value = "(1, 3, 1)"
playground.reveal()
playground.explorer.set_focus((0, 1, 0))
assert playground.visual.result_shape == (1, 3, 1)
assert playground.visual.trace.output_coord == (0, 1, 0)
assert playground.visual.trace.term_count == 8
display(playground.visual)

# An empty axis tuple reduces nothing, preserving every source coordinate.
playground.set_parameters(axis=(), keepdims=False)
assert playground.visual.result_shape == x.shape
assert playground.explorer.focus == (0, 1, 0)
assert playground.visual.trace.term_count == 1
playground.reveal()
display(playground.visual)

# None reduces all axes. Try predicting this scalar shape before revealing it.
playground.set_parameters(axis=None)
assert not playground.revealed
playground.prediction.value = "()"
playground.reveal()
assert playground.visual.result_shape == ()
assert playground.visual.trace.output_coord == ()
assert playground.visual.trace.term_count == 24
assert x.sum() == 276
display(playground)
