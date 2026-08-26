"""Storage metadata is reported without guessing aliasing or copying history."""

import numpy as np
import pytest

from rainbow_tensor import TensorVisual, memory


def test_memory_reports_owned_c_contiguous_array():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    visual = memory(array)

    assert isinstance(visual, TensorVisual)
    assert visual.metadata == {
        "shape": (2, 3),
        "dtype": "int32",
        "itemsize": 4,
        "strides": (12, 4),
        "c_contiguous": True,
        "f_contiguous": False,
        "owns_data": True,
        "has_base": False,
    }
    assert ">6<" in visual.svg
    assert "Byte strides: (12, 4)." in visual.text
    assert "C-contiguous: yes. F-contiguous: no." in visual.text


def test_memory_reports_transposed_view_without_naming_an_operation():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32).T

    visual = memory(array)

    assert visual.shape == (3, 2)
    assert visual.metadata["strides"] == (4, 12)
    assert visual.metadata["c_contiguous"] is False
    assert visual.metadata["f_contiguous"] is True
    assert visual.metadata["owns_data"] is False
    assert visual.metadata["has_base"] is True
    assert "view backed by another object" in visual.text
    assert "do not identify which earlier operation" in visual.text


@pytest.mark.parametrize("step, expected_stride", [(2, 16), (-1, -8)])
def test_memory_reports_stepped_and_reversed_slices(step, expected_stride):
    array = np.arange(8, dtype=np.int64)[::step]

    visual = memory(array)

    assert visual.metadata["strides"] == (expected_stride,)
    assert visual.metadata["c_contiguous"] is False
    assert visual.metadata["f_contiguous"] is False
    assert visual.metadata["owns_data"] is False
    if step < 0:
        assert "negative stride walks backwards" in visual.text


def test_memory_reports_copy_ownership_without_claiming_an_operation():
    array = np.arange(8, dtype=np.int64)[::2].copy()

    visual = memory(array)

    assert visual.metadata["strides"] == (8,)
    assert visual.metadata["c_contiguous"] is True
    assert visual.metadata["owns_data"] is True
    assert visual.metadata["has_base"] is False
    assert "view backed by another object" not in visual.text


def test_memory_explains_zero_stride_broadcast_view():
    array = np.broadcast_to(np.array([7], dtype=np.int16), (3,))

    visual = memory(array)

    assert visual.metadata["strides"] == (0,)
    assert visual.metadata["owns_data"] is False
    assert "zero stride reuses the same storage" in visual.text


def test_shape_tuple_has_explicitly_unknown_storage_metadata():
    visual = memory((2, 3))

    assert visual.shape == (2, 3)
    assert visual.metadata["shape"] == (2, 3)
    assert all(value is None for key, value in visual.metadata.items() if key != "shape")
    assert "Dtype: unknown." in visual.text
    assert "Byte strides: unknown." in visual.text
    assert "Owns data: unknown. Base object present: unknown." in visual.text


def test_array_without_optional_metadata_keeps_values_and_unknown_fields():
    class MinimalArray:
        shape = (2,)

        def __getitem__(self, coord):
            return (19, 23)[coord[0]]

    visual = memory(MinimalArray())

    assert ">23<" in visual.svg
    assert all(value is None for key, value in visual.metadata.items() if key != "shape")


def test_absent_base_does_not_establish_ownership():
    class Array:
        shape = (2,)
        base = None

        def __getitem__(self, coord):
            return coord[0]

    visual = memory(Array())

    assert visual.metadata["has_base"] is False
    assert visual.metadata["owns_data"] is None


def test_mapping_flags_and_non_numpy_metadata_are_supported():
    class Array:
        shape = (2,)
        strides = (8,)
        itemsize = 8
        dtype = "float64"
        flags = {"C_CONTIGUOUS": True, "F_CONTIGUOUS": True, "OWNDATA": False}

        def __getitem__(self, coord):
            return coord[0] + 0.125

    visual = memory(Array(), theme="dark", precision=3)

    assert visual.metadata["strides"] == (8,)
    assert visual.metadata["c_contiguous"] is True
    assert visual.metadata["owns_data"] is False
    assert visual.metadata["has_base"] is None
    assert ">1.125<" in visual.svg


def test_methods_and_unsupported_properties_are_not_interpreted_as_metadata():
    class Array:
        shape = (2,)

        def strides(self):
            raise AssertionError("backend methods must not be called")

        @property
        def itemsize(self):
            raise NotImplementedError("not exposed")

        def __getitem__(self, coord):
            return coord[0]

    visual = memory(Array())

    assert visual.metadata["strides"] is None
    assert visual.metadata["itemsize"] is None


@pytest.mark.parametrize("invalid_shape", [(-1,), (2, -1)])
def test_memory_preserves_shape_validation(invalid_shape):
    with pytest.raises(ValueError):
        memory(invalid_shape)


@pytest.mark.parametrize("strides, itemsize", [((4, 8), 4), ((1.5,), 4), ((4,), -1)])
def test_invalid_optional_metadata_is_reported_as_unknown(strides, itemsize):
    class Array:
        shape = (2,)

        def __getitem__(self, coord):
            return coord[0]

    array = Array()
    array.strides = strides
    array.itemsize = itemsize
    visual = memory(array)

    if strides != (4,):
        assert visual.metadata["strides"] is None
    if itemsize < 0:
        assert visual.metadata["itemsize"] is None


def test_zero_byte_dtype_has_a_known_item_size():
    """A valid zero-byte dtype is distinct from missing storage metadata."""
    visual = memory(np.zeros((2,), dtype="V0"))

    assert visual.metadata["itemsize"] == 0
    assert visual.metadata["strides"] == (0,)
    assert "Item size: 0 bytes." in visual.text


def test_custom_renderer_preserves_metadata_and_explanation():
    class TextRenderer:
        name = "memory-test-text"
        mime_type = "text/plain"

        def render_tensor(self, **kwargs):
            return str(kwargs["shape"])

        def render_panels(self, **kwargs):
            return "unused"

    visual = memory(np.array([1, 2], dtype=np.int16), renderer=TextRenderer())

    assert visual.content == "(2,)"
    assert visual.mime_type == "text/plain"
    assert visual._repr_svg_() is None
    assert visual.metadata["strides"] == (2,)
    assert "Item size: 2 bytes." in visual.text
