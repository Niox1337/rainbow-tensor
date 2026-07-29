"""Preview work should depend on displayed results, not hidden groups."""

import pytest

import rainbow_tensor as rt


class CountingArray:
    """Represent a constant array without allocating its underlying tensor."""

    def __init__(self, shape, value=1):
        self.shape = shape
        self.value = value
        self.reads = 0

    def __getitem__(self, coordinate):
        assert len(coordinate) == len(self.shape)
        assert all(0 <= index < size for index, size in zip(coordinate, self.shape))
        self.reads += 1
        return self.value


@pytest.mark.parametrize("operation", [rt.sum, rt.mean])
def test_hidden_reduction_groups_do_not_increase_backend_reads(operation):
    theme = rt.LIGHT.variant(max_visible_cells=6)
    small = CountingArray((100, 100))
    large = CountingArray((10_000, 100))

    operation(small, 1, theme=theme)
    operation(large, 1, theme=theme)

    assert 0 < small.reads < 1_000
    assert large.reads == small.reads


@pytest.mark.parametrize("operation, expected", [(rt.sum, 300), (rt.mean, 3)])
def test_reduction_reuses_requested_values_within_one_render(operation, expected):
    class RepeatedOutputRenderer:
        """Request the same output twice, as a layout resize can do."""

        name = "repeated-output"
        mime_type = "text/plain"

        def render_tensor(self, **kwargs):
            raise AssertionError("A reduction must render its input and output panels")

        def render_panels(self, panels, **kwargs):
            value = panels[-1]["value_fn"]
            assert value((7,)) == expected
            assert value((7,)) == expected
            return str(expected)

    array = CountingArray((100, 100), value=3)
    operation(array, 1, renderer=RepeatedOutputRenderer())
    assert array.reads == 100


def test_reduction_cache_does_not_outlive_a_visual():
    array = CountingArray((2, 3), value=1)
    first = rt.sum(array, 1)
    array.value = 5
    second = rt.sum(array, 1)

    assert "value 3" in first.svg
    assert "value 15" in second.svg
