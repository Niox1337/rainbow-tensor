/** Connect visible result cells to a live kernel without changing saved SVGs. */
function render({ model, el }) {
  el.classList.add("rt-focus-explorer")
  const style = document.createElement("style")
  style.textContent = `
    .rt-focus-explorer [data-rt-coordinate],
    .rt-focus-explorer [data-rt-coordinate] * { cursor: pointer !important }
    .rt-focus-explorer [data-rt-coordinate]:focus-visible {
      outline: 2px solid currentColor
    }
  `
  const figure = document.createElement("div")
  figure.style.overflowX = "auto"
  el.append(style, figure)

  function paint() {
    const active = figure.contains(document.activeElement)
      ? document.activeElement.getAttribute("data-rt-coordinate") : null
    figure.innerHTML = model.get("value")
    if (active !== null) {
      const cell = [...figure.querySelectorAll("[data-rt-coordinate]")]
        .find((item) => item.getAttribute("data-rt-coordinate") === active)
      cell?.focus({ preventScroll: true })
    }
  }

  function activate(event) {
    if (event.type === "keydown" && !["Enter", " "].includes(event.key)) return
    const cell = event.target instanceof Element
      ? event.target.closest("[data-rt-coordinate]") : null
    if (!cell || !figure.contains(cell)) return
    event.preventDefault()
    cell.focus({ preventScroll: true })
    model.send({
      type: "focus",
      revision: model.get("revision"),
      coordinate: cell.getAttribute("data-rt-coordinate"),
    })
  }

  paint()
  model.on("change:value", paint)
  figure.addEventListener("click", activate)
  figure.addEventListener("keydown", activate)
  return () => {
    model.off("change:value", paint)
    figure.removeEventListener("click", activate)
    figure.removeEventListener("keydown", activate)
    el.replaceChildren()
  }
}

export default { render }
