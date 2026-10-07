"""Explain which source axes survive a prediction and which become length one."""

import numpy as np
from IPython.display import display

import rainbow_tensor as rt

cube = np.arange(24).reshape(2, 3, 4)
playground = rt.reduction_playground(rt.sum, cube, axis=(0, 2), predict=True)
playground.prediction.value = "(2, 1)"
playground.reveal()
assert playground.feedback.code == "rank"
assert playground.feedback.expected == cube.sum(axis=(0, 2)).shape == (3,)
assert playground.feedback.reduced_axes == (0, 2)
display(playground)

# Retaining reduced axes creates length-one slots in their original positions.
playground.set_parameters(axis=(0, 2), keepdims=True)
playground.prediction.value = "(3,)"
playground.reveal()
assert playground.feedback.code == "keepdims"
assert playground.feedback.expected == (1, 3, 1)
display(playground)

playground.prediction.value = "(1, 3, 1)"
playground.reveal()
assert playground.feedback.code == "matches"
assert playground.feedback.predicted == playground.visual.result_shape
display(playground)
