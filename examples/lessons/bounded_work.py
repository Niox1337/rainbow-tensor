"""Explain a partial trace and an unevaluated value without inventing an answer."""

from IPython.display import display

import rainbow_tensor as rt

flow = rt.Flow()
source = flow.input((2, 20), name="Generated")
total = flow.sum(source, axis=1, name="Total")

partial = total.trace((0,), max_nodes=3)
assert not partial.complete
assert "max_nodes" in partial.truncated_reasons

explorer = rt.explore(total, focus=(0,), max_nodes=3, max_terms=4)
assert not explorer.visual.provenance.complete
assert explorer.visual.metadata["value_evaluation"]["status"] == "skipped"
assert "?" in explorer.visual.svg
display(explorer)

# Choosing another output does not lift the arithmetic limit.
explorer.set_focus((1,))
assert explorer.focus == (1,)
assert explorer.visual.metadata["value_evaluation"]["status"] == "skipped"
assert "?" in explorer.visual.svg
display(explorer)

try:
    total.value((1,), max_terms=4)
except rt.ValueBudgetExceeded as error:
    assert error.reason == "max_terms"
else:
    raise AssertionError("The value query must preserve its requested budget")
