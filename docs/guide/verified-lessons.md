# Verified tensor lessons

Each lesson pairs a small NumPy example with assertions about its shape,
sources, and selected output. The same Python file shown here runs in CI.
Start with one lesson, predict its answer, then follow the coordinates in
the figure.

Copy a lesson into a notebook with widget support, or run its source with
IPython's `%run`. The scripts change focus once or twice to check the next
answer. Their controls remain open for further exploration. Close them when
finished with `explorer.close()`, or the corresponding named controls.
For a host without widget support, display a control's `.visual` instead.

## Sum one row

Which source cells produce output `(1,)`? The second row contains 4, 5,
and 6, giving 15. Reducing axis 1 removes the column axis. Selecting output
`(0,)` should move the source highlights to the first row and give 6.

```{literalinclude} ../../examples/lessons/reduction.py
:language: python
```

Try keeping the reduced axis. The result shape becomes `(2, 1)`, so the
first row's output coordinate becomes `(0, 0)`.

## Follow two copies of one element

Predict the source of outputs `(0, 0)` and `(2, 0)` after a repeated row
selection and a reversed column slice. Both read input `(1, 2)`, whose
value is 6. They remain separate positions in the result.

```{literalinclude} ../../examples/lessons/repeated_index.py
:language: python
```

The last selection, `(1, 0)`, instead reads input `(0, 2)`. Its value is 3.
A list of highlighted source cells alone cannot describe output ordering or
how many times a source was selected.

## Locate both broadcast operands

Before calculating `a + b`, ask which value each operand supplies at
`(1, 2)`. The answers are `a[1, 0] = 20` and `b[0, 2] = 3`, giving 23.
Changing to output `(0, 2)` changes the left source row but keeps the right
source coordinate.

```{literalinclude} ../../examples/lessons/broadcast.py
:language: python
```

The two explorers explain expansion independently. NumPy performs the addition
in this lesson. Close both controls with `left.close()` and `right.close()`.

## Preserve repeated contributions through a chain

The selected output of this chain has three contributions but only two
distinct original cells:

```text
Y[0] = T[0, 0] + T[0, 1] + T[0, 2]
     = X[1, 2] + X[0, 2] + X[1, 2]
     = 6 + 3 + 6
     = 15
```

The original value 6 is used twice. Predict how those source coordinates
change when selecting `Y[1]`, whose value is `5 + 2 + 5 = 12`.

```{literalinclude} ../../examples/lessons/operation_origins.py
:language: python
```

The script also exports the current SVG and its text explanation. In a
notebook those files are written to the current working directory. A saved SVG
keeps its highlights, but its controls require a live notebook.

## Distinguish a scalar from an empty result

A scalar result has one element at `()`. An empty output has no element to
select. An empty reduction group is different again. It can produce an output
cell whose sum is 0 or whose mean is undefined.

```{literalinclude} ../../examples/lessons/scalars_and_empties.py
:language: python
```

For input shape `(2, 0, 3)`, reducing axis 2 gives shape `(2, 0)`, with no
outputs. Reducing axis 1 gives shape `(2, 3)`, with six empty groups.
The three live controls here are `scalar`, `no_outputs`, and `empty_sums`.
Each has a `.close()` method.

## Read limits without inventing a result

A shape-only input generates example values without allocating the full
logical tensor. Here the trace and arithmetic budgets deliberately stop work.
Predict whether selecting the second row will remove those limits. It does
not.

```{literalinclude} ../../examples/lessons/bounded_work.py
:language: python
```

A question mark means that a value was not evaluated. A partial trace shows
only the paths it reached. Neither is evidence that omitted contributions are
zero.

## Run the checks

From a source checkout with the project installed, run all registered lessons:

```bash
python scripts/check_lessons.py
```

Use names for a focused check or list the available cases:

```bash
python scripts/check_lessons.py --list
python scripts/check_lessons.py reduction operation-origins
python scripts/check_lessons.py --timeout 45
```

The checker starts a fresh Python process and temporary working directory for
each lesson. It checks the actual assertions, captures displayed figures,
validates SVG and text export, and closes controls even after a failed
assertion. A failure reports the source filename and line. Temporary output
is removed when the lesson finishes or times out.

The checker runs explicitly registered repository code. It is not a sandbox
for arbitrary generated Python. These checks exercise Python-side focus
changes. Browser clicks and keyboard events need their own browser tests.
The package's distribution checks also run these lessons against installed
wheel and source distributions.

For the teaching rationale, read the [learning path](learning-path.md).
Continue with [guided arithmetic and reductions](guided-lessons.md) for
ordered binary expressions, term-by-term explanations and axis prediction.
For instructions on generating a lesson from a NumPy question, use the
[agent reference](llm-prompts.md).
