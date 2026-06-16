# Changelog

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
