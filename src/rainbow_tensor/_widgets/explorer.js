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
  figure.setAttribute("role", "group")
  const announcement = document.createElement("div")
  announcement.setAttribute("role", "status")
  announcement.setAttribute("aria-live", "polite")
  announcement.style.cssText = "position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)"
  el.append(style, figure, announcement)

  function cells() {
    return [...figure.querySelectorAll("[data-rt-coordinate]")]
  }

  function announce() {
    figure.setAttribute("aria-label", model.get("label") || "")
    const selected = cells().find((cell) => cell.getAttribute("aria-pressed") === "true")
    announcement.textContent = [selected?.getAttribute("aria-label"), model.get("description")]
      .filter(Boolean).join(". ")
  }

  function focusCell(cell) {
    cells().forEach((item) => item.setAttribute("tabindex", item === cell ? "0" : "-1"))
    cell.focus({ preventScroll: true })
  }

  function paint() {
    const active = figure.contains(document.activeElement)
      ? document.activeElement.getAttribute("data-rt-coordinate") : null
    figure.innerHTML = model.get("value")
    const available = cells()
    const entry = available.find((cell) => cell.getAttribute("data-rt-coordinate") === active)
      || available.find((cell) => cell.getAttribute("aria-pressed") === "true") || available[0]
    available.forEach((cell) => cell.setAttribute("tabindex", cell === entry ? "0" : "-1"))
    if (active !== null) {
      entry?.focus({ preventScroll: true })
    }
    announce()
  }

  // Coordinates arrive as decimal strings so even very large Python integers
  // keep their identity. JSON.parse would round values beyond 2**53.
  function coordinate(cell) {
    const payload = cell.getAttribute("data-rt-coordinate").slice(1, -1).trim()
    return payload ? payload.split(",").map((value) => BigInt(value.trim())) : []
  }

  function neighbor(cell, key) {
    const origin = coordinate(cell)
    if (!origin.length) return null
    const horizontal = ["ArrowLeft", "ArrowRight"].includes(key)
    const axis = horizontal ? origin.length - 1 : Math.max(0, origin.length - 2)
    const forward = ["ArrowRight", "ArrowDown"].includes(key)
    let nearest = null
    let distance = null
    for (const candidate of cells()) {
      const position = coordinate(candidate)
      if (position.length !== origin.length
          || position.some((value, index) => index !== axis && value !== origin[index])) continue
      const delta = forward ? position[axis] - origin[axis] : origin[axis] - position[axis]
      if (delta > 0n && (distance === null || delta < distance)) {
        nearest = candidate
        distance = delta
      }
    }
    return nearest
  }

  function activate(event) {
    const arrows = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]
    if (event.type === "keydown" && !["Enter", " ", ...arrows].includes(event.key)) return
    let cell = event.target instanceof Element
      ? event.target.closest("[data-rt-coordinate]") : null
    if (!cell || !figure.contains(cell)) return
    event.preventDefault()
    event.stopPropagation()
    if (event.type === "keydown" && arrows.includes(event.key)) {
      cell = neighbor(cell, event.key)
      if (!cell) return
    }
    focusCell(cell)
    model.send({
      type: "focus",
      revision: model.get("revision"),
      coordinate: cell.getAttribute("data-rt-coordinate"),
    })
  }

  paint()
  model.on("change:value", paint)
  model.on("change:description", announce)
  model.on("change:label", announce)
  figure.addEventListener("click", activate)
  figure.addEventListener("keydown", activate)
  return () => {
    model.off("change:value", paint)
    model.off("change:description", announce)
    model.off("change:label", announce)
    figure.removeEventListener("click", activate)
    figure.removeEventListener("keydown", activate)
    el.replaceChildren()
  }
}

export default { render }
