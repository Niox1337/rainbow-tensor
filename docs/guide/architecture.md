# Architecture and performance

A rainbow-tensor visual explains an operation through shapes, coordinates and a
bounded sample of values. It does not need to construct the complete result
tensor. Keeping those responsibilities separate makes large logical tensors
useful teaching inputs without turning a notebook preview into a large compute
job.

## From an operation to a visual

| Layer | Responsibility |
| --- | --- |
| [Public views](../../src/rainbow_tensor/views/__init__.py) | Assemble an operation's panels, explanations, value callbacks and coordinate trace. Operation families have separate modules for shapes, reshaping, combining, reductions and einsum. |
| [Shape and coordinate operations](../../src/rainbow_tensor/ops/__init__.py) | Calculate result shapes and source-coordinate mappings using Python shapes and integers. This layer does not import a tensor backend. |
| [Index mapping](../../src/rainbow_tensor/index_mapping.py) | Preserve result order and repeated gathers while mapping individual output coordinates back to the source. Compact source-selection queries answer membership, prefix and axis-pin requests without expanding a selected slice product. |
| [Basic selection](../../src/rainbow_tensor/selection.py) | Represent basic indexing with ranges, including reverse slices and very large dimensions. Enumerating the selection is an explicit choice. |
| [Layout](../../src/rainbow_tensor/layout.py) | Choose visible positions and place cells and frames within the preview limits. Scalars have one real cell at coordinate `()`. Empty shapes have an empty marker and no value reads. |
| [Evaluation planning](../../src/rainbow_tensor/evaluation.py) | Check per-output and total-preview term budgets before numerical value callbacks run. Cache completed outputs and bound additional requests from custom renderers. |
| [Recorded flows](../../src/rainbow_tensor/provenance/flow.py) | Capture explicit operation recipes and assign stable input, operation and output-port identities without materialising intermediate arrays. |
| [Provenance queries](../../src/rainbow_tensor/provenance/query.py) | Traverse a bounded occurrence tree and plan shared recursive value work before reading an input. Preserve repeated paths and each operation's arithmetic grouping. |
| [Flow presentation](../../src/rainbow_tensor/provenance/view.py) | Render reached tensors, local equations and input contribution counts as a regular result object. |
| [Notebook controllers](../../src/rainbow_tensor/interactive/__init__.py) | Keep focus, calculation walkthroughs, and reduction prediction controls in separate modules. |
| [Portable lessons](../../src/rainbow_tensor/lessons/__init__.py) | Capture bounded prepared states as immutable JSON data and a self-contained offline HTML player. |
| [SVG elements](../../src/rainbow_tensor/_svg/elements.py) | Escape text, format values, measure text, produce paint attributes and wrap a complete SVG document. Adaptive colours carry literal light fallbacks. |
| [Tensor drawing](../../src/rainbow_tensor/_svg/tensor.py) | Draw one tensor body, its frames, cells, empty marker and legend from the layout. |
| [SVG facade](../../src/rainbow_tensor/render_svg.py) | Compose single-tensor and multi-panel figures, captions and connectors. Historical rendering entry points and helper imports remain available here. |
| [Group colours](../../src/rainbow_tensor/colors.py) | Derive stable group and operand paints from their logical IDs, with paired light and dark colours and a bounded cache. |

The [result object](../../src/rainbow_tensor/visual.py) exposes the SVG, plain
explanation, metadata and optional provenance. A renderer consumes coordinate
callbacks rather than requiring a NumPy array. That boundary also supports
shape-only examples and other array-like backends.

The internal `_svg` package separates reusable paint and text rules from tensor
drawing. Panel composition stays in `render_svg.py`, so callers do not need to
follow the internal split. New operation semantics belong in `ops` and their
teaching presentation belongs in `views`.

The `provenance` package adds an explicit layer above coordinate operations.
Its `model` contains immutable tracked tensors and element references, `flow`
builds recipes, `query` handles structural and numerical requests, and `view`
assembles teaching panels. A tracked tensor describes a logical result and has
its own `shape`. Rendering it creates a separate `TensorVisual`, so operation
history does not depend on a particular SVG or notebook widget.

The `interactive` package owns focus controls, walkthroughs, and reduction
playgrounds in separate modules. Public factories remain available directly
from `rainbow_tensor`. The historical `walkthrough` and `playground` modules
re-export the same controller objects for import compatibility. Keep numerical
and coordinate rules below this layer so static figures and notebook controls
share their meaning. Shape prediction diagnostics operate only on shapes.

The `lessons` package captures visuals and controller snapshots that are
already prepared. It does not evaluate a tensor again or add uncaptured states
to the recording. Its HTML player navigates those saved states without Python
or network access.

The capability reference reads the public view registry and callable signatures.
The explorer uses the same operation inventory. Documentation exports that
contract at build time, so contributors do not maintain a second API list.

Every graph reference includes an operation ID, an output port and a coordinate.
Broadcast outputs can therefore share one recorded operation while retaining
their separate values. Traversal records occurrences rather than collapsing all
repeated references into one branch. A repeated gather contributes twice even
when its original source cell is drawn once.

Group colours are generated on demand instead of cycling through a short fixed
palette. Only requested IDs are calculated and the cache holds at most 1,024
entries. Changing focus does not assign a new identity to a group. Colours can
still become difficult to distinguish in a crowded figure, so coordinates and
the focused trace remain the precise way to identify contributions.

## What bounded rendering does and does not cover

Source-value reads follow the visible layout. Numerical previews have separate
limits for terms per output cell and across the whole preview. An operation that
exceeds its evaluation budget still displays its shape and a bounded source
sample, with unknown output values explained in the text and metadata.

Index arrays must still be validated and stored. Boolean masks require a scan
to determine which positions are selected. These costs depend on the supplied
index data, even when only a few source cells are drawn.

For equally shaped gather arrays, candidate positions share one coordinate
space. Their compatibility can therefore be decided by set intersection,
starting with the smallest candidate set. An empty intermediate intersection
stops the query. Differently shaped arrays retain the general broadcast-aware
join, including its treatment of singleton and private axes.

Index labels have their own limits. The
[index formatter](../../src/rainbow_tensor/indexing.py) shares a budget of 32
indexed entries across a label's nested arrays and stops after four recursive
levels. It retains head and tail entries where the budget permits, and localized
omission markers report the containing array's original length. Small labels
keep their complete notation. These limits affect captions only, so the index
mapping still retains every result position.

A size check on 23 September 2026 used the preview command below with
`--group-size 20000 --loops 1`. Both versions retained 40,000 gather positions
and read four source cells. Only the index label formatter changed:

| Label formatter | Complete SVG size |
| --- | --- |
| Full index arrays | 482,687 bytes |
| Bounded head and tail labels | 2,902 bytes |

Repeat follows the same principle. Uniform counts use a constant-size quotient
lookup. Per-element counts use cumulative boundaries and binary search, with
storage proportional to the input counts rather than the repeated output.
The lookup's `count` remains exact beyond Python's platform-sized length limit.

## Recorded flow boundaries

Structural queries have independent limits for depth, occurrence nodes and
edges. Truncation is explicit, and input counts on an incomplete trace describe
only reached paths. Panel limits control how many tensors are drawn without
changing the structural trace.

Numerical queries plan recursive dependencies before any input values are read.
The total budget counts operation terms, factor references and input reads.
Shared intermediate elements are evaluated once within that request. Each
operation retains its own arithmetic, including division at each mean, so
flattening a trace cannot silently change the result. A separate depth guard
limits numerical recursion to 64 operation edges.

Value caches last for one request. Updating an input array's values is reflected
on the next request, while a changed input shape is rejected. Operation
parameters, including copied index arrays, stay fixed after recording.
Numerical work uses Python scalars and does not dispatch tensor backend kernels.
See [the provenance guide](provenance.md) for the public contract and defaults.

Conditional and extrema traces list structural candidates separately from the
source chosen by numerical evaluation. A winner can be found even when its
candidate path is outside the bounded trace. The visual highlights that source
without adding invented edges to the trace.

## Reproducing the index workload

Install the checkout in editable mode, then run the
[benchmark script](../../scripts/benchmark_index_queries.py) from the repository
root:

```sh
python scripts/benchmark_index_queries.py --phase preview --group-size 20000 --loops 5
python scripts/benchmark_index_queries.py --phase query --group-size 20000 --loops 10
```

The source shape is `(2, 2)`. The index arrays repeat `[0, 1]` and `[1, 0]`,
20,000 times per value. The result retains 40,000 positions but selects only two
distinct source cells. Each complete preview reads four source values, including
the unselected context cells. The query phase constructs one mapping before its
timed loop and then runs actual layout membership and prefix queries without
reading source values.

For repeated process measurements:

```sh
hyperfine --warmup 2 --runs 7 "python scripts/benchmark_index_queries.py --phase preview --group-size 20000 --loops 5"
hyperfine --warmup 2 --runs 7 "python scripts/benchmark_index_queries.py --phase query --group-size 20000 --loops 10"
```

Compare the same script and inputs in each implementation. The JSON printed by
the script measures its operation loop. Hyperfine measures the complete process,
including imports, input construction and the query phase's initial mapping.
Those are different timing boundaries.

## Measurements and the Rust decision

Measurements were taken on 22 September 2026 using Windows 11 x64, an AMD Ryzen
7 7700X, Python 3.12.12, NumPy 2.1.3 and Hyperfine 1.20.0. Each Hyperfine command
had two warmups and seven measured runs. Times below are process means with
standard deviations, not a promise for every machine or indexing pattern.

The original join was preserved from the `d419ed5` baseline before the rendering
split in `116af13`. The final before-and-after harness swapped that saved join
and the new equal-shape path while keeping the workload and surrounding package
the same.

| Final before-and-after experiment | Original join | Equal-shape path | Speedup |
| --- | --- | --- | --- |
| Five complete previews | 2.009 ± 0.024 s | 1.220 ± 0.002 s | 1.65× |
| Ten query-layout operations, including process setup | 1.737 ± 0.017 s | 0.305 ± 0.003 s | 5.69× |

An earlier, separate prototype experiment compared a Python set path with a
small Rust sorted-intersection routine compiled using Rust 1.96.0. It included
the cost of converting Python candidate coordinates into native buffers.

| Earlier prototype experiment, five complete previews | Process time |
| --- | --- |
| Original join | 2.249 ± 0.018 s |
| Python set prototype | 1.346 ± 0.008 s |
| Rust prototype with conversion on each query | 1.423 ± 0.006 s |
| Rust prototype with converted candidate buffers cached | 1.374 ± 0.007 s |

The prototype and final tables are separate runs and should not be combined to
derive another speedup. The practical result was consistent: changing the
Python algorithm improved the complete preview, while adding a native boundary
did not beat it. The shipped path needs no Rust extension or persistent query
cache.

Rust remains an option for a future measured hotspot. A useful boundary would
accept a batch of already normalized, contiguous data and do enough work to
amortize conversion and call costs. That decision should include complete
preview timings, buffer ownership and lifetime, wheel support, Python fallback
behaviour and the existing coordinate and numerical contracts. Moving a small
loop while keeping repeated Python callbacks around it is not sufficient
evidence. Value-read limits and compact handling of huge logical shapes must
remain intact whichever implementation performs the work.
