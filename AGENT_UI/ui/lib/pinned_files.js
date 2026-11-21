;(function (root, factory) {
  const api = factory()
  if (typeof module === 'object' && module.exports) module.exports = api
  else root.PinnedFiles = api
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const PIN_MAX = 10

  function _clean(paths) {
    return Array.isArray(paths)
      ? paths.filter(p => typeof p === 'string' && p.trim()).map(p => p.trim())
      : []
  }

  function dedupe(paths) {
    const seen = new Set()
    const out = []
    for (const p of _clean(paths)) {
      if (seen.has(p)) continue
      seen.add(p)
      out.push(p)
    }
    return out
  }

  function mergePinned(current, candidates, max = PIN_MAX) {
    const base = dedupe(current)
    const seen = new Set(base)
    let duplicateCount = 0
    let rejectedCount = 0
    for (const p of _clean(candidates)) {
      if (seen.has(p)) {
        duplicateCount += 1
        continue
      }
      if (base.length >= max) {
        rejectedCount += 1
        continue
      }
      seen.add(p)
      base.push(p)
    }
    return { files: base, duplicateCount, rejectedCount }
  }

  function withoutPinned(paths, pinned) {
    const pins = new Set(dedupe(pinned))
    return dedupe(paths).filter(p => !pins.has(p))
  }

  return { PIN_MAX, dedupe, mergePinned, withoutPinned }
})
