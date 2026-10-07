/* Replay explicitly captured states without executing tensor operations. */
(() => {
  "use strict";
  const lesson = JSON.parse(document.getElementById("lesson-data").textContent);
  const labels = lesson.labels;
  const root = document.getElementById("lesson-player");
  const previous = document.getElementById("previous");
  const next = document.getElementById("next");
  const reveal = document.getElementById("reveal");
  const result = document.getElementById("result");
  let position = 0;
  let revealed = true;

  function text(id, value) {
    document.getElementById(id).textContent = value;
  }

  function showResult() {
    result.hidden = !revealed;
    reveal.textContent = revealed ? labels.hide : labels.reveal;
    reveal.setAttribute("aria-expanded", String(revealed));
    reveal.setAttribute("aria-controls", "result");
  }

  function render() {
    const active = document.activeElement;
    const state = lesson.states[position];
    text("state-number", `${labels.state} ${position + 1} / ${lesson.states.length}`);
    previous.disabled = position === 0;
    next.disabled = position === lesson.states.length - 1;
    const coordinate = state.focus === null ? labels.empty :
      `(${state.focus.join(", ")}${state.focus.length === 1 ? "," : ""})`;
    text("focus", `${labels.focus}: ${coordinate}`);
    document.getElementById("result-image").src = state.image;
    text("explanation", state.text);
    const completeness = document.getElementById("completeness");
    completeness.replaceChildren();
    for (const key of ["trace", "ancestry", "values"]) {
      const name = document.createElement("dt");
      const value = document.createElement("dd");
      name.textContent = labels[key];
      value.textContent = labels[state.completeness[key]];
      completeness.append(name, value);
    }
    text("metadata", JSON.stringify({metadata: state.metadata, lesson: state.lesson}, null, 2));
    const prediction = state.prediction;
    document.getElementById("prediction").hidden = prediction === null;
    revealed = prediction === null || prediction.revealed;
    if (prediction !== null) {
      text("expression", `${labels.expression}: ${prediction.expression}`);
      text("prediction-prompt", prediction.prompt);
      text("prediction-answer", prediction.answer);
      document.getElementById("source-image").src = prediction.source_image;
    }
    showResult();
    if ((active === previous && previous.disabled) || (active === next && next.disabled)) {
      root.focus({preventScroll: true});
    }
  }

  function move(target) {
    const nextPosition = Math.max(0, Math.min(lesson.states.length - 1, target));
    if (nextPosition === position) return;
    position = nextPosition;
    render();
  }

  previous.addEventListener("click", () => move(position - 1));
  next.addEventListener("click", () => move(position + 1));
  reveal.addEventListener("click", () => { revealed = !revealed; showResult(); });
  root.addEventListener("keydown", event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const target = {
      ArrowLeft: position - 1, ArrowRight: position + 1,
      Home: 0, End: lesson.states.length - 1,
    }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    move(target);
  });
  render();
})();
