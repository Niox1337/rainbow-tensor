# Changelog

## Unreleased

### Added

- Shape changing views. Reshape draws the same values flowing from the old layout into the new one, transpose reorders the axes with each axis keeping its colour, and sum and mean collapse a chosen axis and mark the elements that fold into each result. The source and the result sit side by side in one figure through a new multi panel renderer
- Render tensors of any rank. Frames nest to arbitrary depth, alternating across and down so a 4D or 5D tensor stays compact, with per axis truncation at every level
- Advanced indexing. A full-shape boolean mask highlights every True position, and integer index arrays highlight the gathered coordinates. The result shape matches NumPy for mixed basic and advanced indexing, including moving the gathered axis to the front when a slice separates the advanced axes

### Changed

- Renamed `show_shape` to `shape` and `show_index` to `index`. The first argument is an array-like object whose own values are rendered, or a shape tuple as before

## 0.2.1

### Fixed

- show_shape and show_index no longer render twice in a notebook cell. The result renders once through _repr_svg_ and is still returned for inspection
- The SVG width now grows to fit the label and explanation lines, so longer text is no longer clipped at the edge of the frame

## 0.2.0

### Changed

- Redesigned the visualisation so each axis has its own colour, with axis 0 frames red, axis 1 frames orange, and selected leaf values green
- The shape label numbers and the index label tokens are now coloured to match the frames and the selected values
- In an index view the unselected frames and values are drawn in a neutral dark tone instead of being faded

## 0.1.0

First version.

### Added

- Static SVG visualisation of tensor shape for 1D, 2D, and 3D tensors through `show_shape`
- Static SVG visualisation of basic indexing through `show_index`, with highlighted selections and de-emphasised context
- A plain text explanation of the result shape and of which axes are kept or removed
- Support for shape tuples and array-like objects with a `.shape` attribute, including NumPy arrays
- Integer indexing and basic slicing with `slice(None)`, `slice(start, stop)`, and `slice(start, stop, step)`

### Known limitations

- Interactive widgets are not included in this version
- Advanced indexing, boolean masks, `None`, `newaxis`, and ellipsis are not supported
- Tensors of 4D or higher are not supported

### Planned next steps

- Interactive controls in a notebook widget
- Higher dimensional layouts
- Optional support for more tensor libraries through shape duck typing
