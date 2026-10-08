/** Keep native path selection keys inside the notebook lesson. */
function render({ model, el }) {
  el.classList.add("rt-calculation-paths")
  const label = document.createElement("label")
  const caption = document.createElement("span")
  const select = document.createElement("select")
  label.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%"
  select.style.cssText = "width:100%;font:inherit"
  select.size = 6
  label.append(caption, select)
  el.append(label)
  let optionSignature = null

  function paint() {
    const options = model.get("options") || []
    const signature = JSON.stringify(options)
    const focused = document.activeElement === select
    if (signature !== optionSignature) {
      select.replaceChildren(...options.map(([text, value]) => {
        const option = document.createElement("option")
        option.textContent = text
        option.value = String(value)
        return option
      }))
      optionSignature = signature
    }
    caption.textContent = model.get("description") || ""
    select.setAttribute("aria-label", caption.textContent)
    select.disabled = model.get("disabled")
    const value = model.get("value")
    select.value = value === null ? "" : String(value)
    if (focused) select.focus({ preventScroll: true })
  }

  function isolate(event) {
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End", "PageUp",
      "PageDown", "Enter", " "].includes(event.key)) event.stopPropagation()
  }

  function choose() {
    if (!select.disabled && select.selectedIndex >= 0) {
      model.set("value", Number(select.value))
      model.save_changes()
    }
  }

  const events = ["change:options", "change:value", "change:description", "change:disabled"]
  events.forEach((event) => model.on(event, paint))
  select.addEventListener("keydown", isolate)
  select.addEventListener("change", choose)
  paint()
  return () => {
    events.forEach((event) => model.off(event, paint))
    select.removeEventListener("keydown", isolate)
    select.removeEventListener("change", choose)
    el.replaceChildren()
  }
}

export default { render }
