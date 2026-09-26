"""Large index labels stay bounded without changing the exact selected result."""

from xml.etree import ElementTree

import numpy as np
import pytest

import rainbow_tensor as rt
from rainbow_tensor.indexing import format_index, format_token


class CountingIndex:
    """Expose an index tree while rejecting full traversal and slice copies."""

    def __init__(self, shape, reads=None, prefix=()):
        self.shape = shape
        self.reads = [] if reads is None else reads
        self.prefix = prefix

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        assert isinstance(index, int), "labels must not copy an index slice"
        assert 0 <= index < self.shape[0]
        coordinate = (*self.prefix, index)
        self.reads.append(coordinate)
        if len(self.shape) > 1:
            return CountingIndex(self.shape[1:], self.reads, coordinate)
        return sum(position * 100**axis for axis, position in enumerate(reversed(coordinate)))

    def __iter__(self):
        raise AssertionError("labels must not enumerate the complete index array")

    def __str__(self):
        raise AssertionError("labels must not stringify the complete index array")


@pytest.mark.parametrize("index,expected", [
    ([], "[]"),
    ([0, 1, -1], "[0, 1, -1]"),
    ([[0, 1], [2, 3]], "[[0, 1], [2, 3]]"),
    ([[0], [1], [2]], "[[0], [1], [2]]"),
    ([[], []], "[[], []]"),
    ([True, False, True], "[True, False, True]"),
    (np.array([[0, 1], [2, 3]]), "[[0, 1], [2, 3]]"),
    (np.array([True, False]), "[True, False]"),
])
def test_small_index_arrays_keep_their_exact_existing_notation(index, expected):
    assert format_token(index) == expected


def test_special_index_tokens_keep_their_exact_notation():
    assert format_index((slice(None, None, -1), None, Ellipsis, -1)) == "::-1, None, ..., -1"


def test_flat_label_reads_only_head_and_tail_and_reports_original_length():
    index = CountingIndex((40_000,))
    label = format_token(index)
    assert index.reads == [(0,), (1,), (2,), (39_997,), (39_998,), (39_999,)]
    assert label == "[0, 1, 2, ... (39994 of 40000 entries omitted), 39997, 39998, 39999]"
    assert len(label) < 100


def test_nested_label_shares_one_read_budget_across_every_branch():
    index = CountingIndex((19, 19))
    label = format_token(index)
    assert len(index.reads) <= 32
    assert (0, 0) in index.reads
    assert (18, 18) in index.reads
    assert "1818" in label
    assert "of 19 entries omitted" in label
    assert len(label) < 1000


def test_nested_branching_does_not_multiply_the_global_entry_budget():
    index = CountingIndex((100,) * 20)
    label = format_token(index)
    assert len(index.reads) <= 32
    assert max(map(len, index.reads)) <= 4
    assert len(label) < 2000
    assert "of 100 entries omitted" in label


def test_deep_singleton_lists_stop_before_python_recursion_limits():
    index = 9
    for _ in range(1000):
        index = [index]
    label = format_token(index)
    assert label == "[[[[[... (1 of 1 entries omitted)]]]]]"
    assert len(label) < 100


def test_large_boolean_labels_preserve_their_kind_at_both_ends():
    label = format_token([True] * 50 + [False] * 50)
    assert label.startswith("[True, True, True,")
    assert label.endswith("False, False, False]")
    assert "94 of 100 entries omitted" in label


def test_empty_nested_arrays_do_not_claim_missing_entries():
    label = format_token(np.empty((3, 0), dtype=int))
    assert label == "[[], [], []]"


def test_omission_text_uses_the_selected_language():
    previous = rt.get_language()
    try:
        rt.set_language("zh")
        label = format_token(list(range(10)))
        assert "省略 4 项" in label
        assert "共 10 项" in label
    finally:
        rt.set_language(previous)


def test_forty_thousand_gathers_keep_exact_mapping_with_a_small_svg():
    class Source:
        shape = (2, 2)

        def __init__(self):
            self.reads = []

        def __getitem__(self, coordinate):
            self.reads.append(coordinate)
            return 1

    source = Source()
    selection = (np.repeat([0, 1], 20_000), np.repeat([1, 0], 20_000))
    visual = rt.index(source, selection, theme="light")
    mapping = visual.index_mapping
    assert mapping.result_shape == (40_000,)
    assert mapping.result_count == 40_000
    assert mapping.source_coord((0,)) == (0, 1)
    assert mapping.source_coord((19_999,)) == (0, 1)
    assert mapping.source_coord((20_000,)) == (1, 0)
    assert mapping.source_coord((39_999,)) == (1, 0)
    assert len(source.reads) == 4
    assert len(visual.svg.encode("utf-8")) < 10_000
    assert "39994 of 40000 entries omitted" in visual.svg
    assert ElementTree.fromstring(visual.svg).tag.endswith("svg")
