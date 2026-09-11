"""Measure duplicate-gather previews and their compact highlight queries.

Run after installing the checkout in editable mode. Compare the same command
on two revisions with ``hyperfine --warmup 2 --runs 7``. The default workload
uses two index arrays of length 40,000 and only four source cells, so source
reads stay bounded while candidate-query costs become measurable. The preview
phase includes parsing, labels, layout and SVG output. The query phase reuses
one mapping and measures the actual layout membership and prefix queries.
"""

import argparse
import json
from time import perf_counter

import numpy as np

import rainbow_tensor as rt
from rainbow_tensor.index_mapping import IndexMapping
from rainbow_tensor.layout import build_layout


class CountingSource:
    """Expose four values and count reads made by the complete preview."""

    shape = (2, 2)

    def __init__(self):
        self.reads = 0

    def __getitem__(self, coordinate):
        if len(coordinate) != 2 or any(value not in (0, 1) for value in coordinate):
            raise IndexError(coordinate)
        self.reads += 1
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preview", "query"), default="preview")
    parser.add_argument("--group-size", type=int, default=20_000)
    parser.add_argument("--loops", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.group_size < 1 or arguments.loops < 1:
        parser.error("--group-size and --loops must be positive")
    index = (
        np.repeat([0, 1], arguments.group_size),
        np.repeat([1, 0], arguments.group_size),
    )
    rt.set_language("en")
    rt.set_default_theme("light")
    mapping = IndexMapping(CountingSource.shape, index) if arguments.phase == "query" else None
    reads = 0
    svg_bytes = None
    started = perf_counter()
    for _ in range(arguments.loops):
        if mapping is None:
            source = CountingSource()
            visual = rt.index(source, index)
            if source.reads != 4:
                raise AssertionError(f"preview read {source.reads} values instead of four")
            if visual.index_mapping.result_count != 2 * arguments.group_size:
                raise AssertionError("preview lost repeated gather positions")
            reads += source.reads
            svg_bytes = len(visual.svg.encode("utf-8"))
        else:
            layout = build_layout(CountingSource.shape, selected=mapping.selection, theme=rt.LIGHT)
            selected = {cell.coord for cell in layout.cells if cell.selected}
            if selected != {(0, 1), (1, 0)}:
                raise AssertionError(f"incorrect source highlights: {selected}")
    elapsed = perf_counter() - started
    print(json.dumps({
        "phase": arguments.phase,
        "index_length": 2 * arguments.group_size,
        "loops": arguments.loops,
        "elapsed_seconds": elapsed,
        "milliseconds_per_operation": 1000 * elapsed / arguments.loops,
        "backend_reads": reads,
        "svg_bytes": svg_bytes,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
