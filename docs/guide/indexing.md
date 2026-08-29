# Indexing

`index` shows how an indexing expression selects elements. The selected cells
fill green and the selected frames keep their colour, while the rest of the
tensor dims so the selected path stands out.

```python
import numpy as np
import rainbow_tensor as rt

x = np.arange(8).reshape(2, 2, 2)
rt.index(x, (0, slice(None), 1))
```

For this array the index `(0, slice(None), 1)` selects the values `1` and `3`,
the selected coordinates are `(0, 0, 1)` and `(0, 1, 1)`, and the result shape
is `(2,)`.

## Compare the source with the result

Use `show_result=True` to keep the original array visible beside the indexed
result. An arrow connects the panels. The source keeps its usual colours and
the result cells are highlighted.

```python
x = np.array([10, 20, 30])
visual = rt.index(x, ([2, 0, 2],), show_result=True)
visual                              # result values: 30, 10, 30
visual.index_mapping.source_coord((0,))  # (2,)
visual.index_mapping.result_count    # 3, including the repeated pick
```

Result order matters. The source has only two selected positions in this
example, but the output has three elements. `visual.selected` describes source
highlights and must not be used to reconstruct output order. Use
`visual.index_mapping.source_coord(output_coordinate)` for that purpose.
Scalar results occupy one display cell. Empty results contain no value cells.

## Integers and slices

An integer drops an axis. A slice keeps it. Slice bounds and steps may be
negative, so `slice(4, 1, -1)` walks backwards.
NumPy integer scalars and other objects implementing `__index__` are accepted.
Trailing axes can be omitted and are kept as full slices.

```python
rt.index(np.arange(5), (slice(4, 1, -1),))
```

## Ellipsis and a new axis

`Ellipsis` fills the axes you leave out, so `(0, ..., 1)` keeps the middle axis.
`None` inserts a size one axis that shows up in the result shape and label.

```python
rt.index(np.arange(12).reshape(2, 3, 2), (0, ..., 1))
rt.index(np.arange(6).reshape(2, 3), (None, 1, slice(None)))
rt.index(np.arange(12).reshape(3, 4), (None, [2, 0]), show_result=True)
```

## Boolean masks

A boolean mask of matching shape highlights every True position. A boolean
array on a single axis acts like its nonzero integer array and mixes with
slices and integer indices.

```python
x = np.arange(20).reshape(4, 5)
rt.index(x, x % 3 == 0)
```

## Fancy indexing

Integer arrays gather coordinates. Paired arrays pick one element each, and
multi-dimensional index arrays broadcast together and may mix with slices, with
the gathered block placed where NumPy puts it.

```python
x = np.arange(20).reshape(4, 5)
rt.index(x, ([0, 2, 3], [1, 4, 0]))
```

Integer scalars participate in the gathered block. If a slice, new axis, or ellipsis
separates them from integer arrays, the gathered axes move to the front.
Even an ellipsis that fills zero axes can affect this ordering.
Floating-point index arrays are rejected rather than rounded or truncated.

## Large selections

Basic indexing stores source-axis ranges, so a large selection does not need
a list of every coordinate before drawing its preview.

```python
visual = rt.index((1_000_000, 1_000_000), (Ellipsis,))
visual.selected.count            # 1_000_000_000_000
visual.selected[0]               # (0, 0)
(10, 20) in visual.selected      # True
```

For basic indexing, `visual.selected` supports lazy iteration, integer lookup,
and explicit slices.
Use `.count` for selections larger than Python's `len` limit. Converting the
selection to a list still materializes it, so reserve that for small examples.
The lower-level `selected_coordinates` helper continues to return a list.

Advanced indexing also keeps sliced axes as ranges. Index arrays are normalized
once, and result coordinates are mapped only when needed. For example,
`rt.index((1_000_000, 1_000_000), ([0], slice(None)), show_result=True)` can
preview a million outputs without building a million selected coordinate tuples.

Advanced source highlights support membership and explicit iteration over unique
positions. They do not expose a constant-time unique count. Use
`visual.index_mapping.result_count` for the exact output size, including repeats.
Explicitly iterating every highlight can still require substantial work.
Boolean masks must be scanned, and their true coordinates are retained, so their
cost depends on the supplied mask. The lower-level `advanced_index` helper still
materializes its source-coordinate list for compatibility.

Ragged nested index lists and floating-point indices are rejected. Empty boolean
masks retain their boolean type and dimensionality. Boolean scalar indices
remain unsupported. Scalar sources use coordinate `()`, so `rt.index(np.array(7),
(), show_result=True)` selects their one value. Empty sources and empty index
results show **No elements** without reading values. Scalar integer indices into a
zero-length axis raise `IndexError`.

## Clear errors

An out of range index points at the offending axis instead of failing deep
inside NumPy.

```python
x = np.arange(8).reshape(2, 2, 2)
try:
    rt.index(x, (2, slice(None), 1))
except IndexError as error:
    print(error)
```
