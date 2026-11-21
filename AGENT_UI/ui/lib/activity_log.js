/* Activity log model: turn cards, newest first, results attached to their call.
 * Pure data (no DOM) so ordering and caps are unit-testable; renderer.js draws it. */
;(function (root, factory) {
  const api = factory()
  if (typeof module === 'object' && module.exports) module.exports = api
  else root.ActivityLog = api
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const text = v => String(v ?? '')

  function createActivityModel({ maxTurns = 100, maxEntries = 500 } = {}) {
    let turns = []          // oldest -> newest internally; snapshot() reverses
    let nextTurn = 1

    function currentTurn(ts) {
      if (!turns.length) turns.push({ id: nextTurn++, ts: text(ts), entries: [] })
      return turns[turns.length - 1]
    }

    function enforceCaps() {
      while (turns.length > maxTurns) turns.shift()
      let total = turns.reduce((n, t) => n + t.entries.length, 0)
      while (total > maxEntries && turns.length) {
        const oldest = turns[0]
        oldest.entries.shift()          // entries are stored oldest -> newest
        total--
        if (!oldest.entries.length && turns.length > 1) turns.shift()
        else if (!oldest.entries.length) break
      }
      turns = turns.filter((t, i) => t.entries.length > 0 || i === turns.length - 1)
    }

    return {
      startTurn(ts) {
        turns.push({ id: nextTurn++, ts: text(ts), entries: [] })
        enforceCaps()
      },
      addCall({ callId, name, detail = '', args = '', ts = '' }) {
        const turn = currentTurn(ts)
        turn.entries.push({
          kind: 'call', id: text(callId), name: text(name), detail: text(detail), args: text(args),
          status: 'running', preview: '', truncated: false, ts: text(ts),
        })
        enforceCaps()
      },
      finishCall({ callId, status = 'returned', preview = '', truncated = false }) {
        const id = text(callId)
        for (let t = turns.length - 1; t >= 0; t--) {
          const entry = turns[t].entries.find(e => e.kind === 'call' && e.id === id)
          if (entry) {
            entry.status = text(status) || 'returned'
            entry.preview = text(preview)
            entry.truncated = Boolean(truncated)
            return entry
          }
        }
        return null
      },
      addPlan(steps, ts = '') {
        currentTurn(ts).entries.push({ kind: 'plan', steps: (steps || []).map(text), ts: text(ts) })
        enforceCaps()
      },
      snapshot() {
        return [...turns].reverse().map(t => ({ ...t, entries: [...t.entries].reverse() }))
      },
      clear() { turns = [] },
    }
  }

  // More than three lines or 180 characters is shown collapsed with a disclosure.
  function describeText(value) {
    const full = text(value)
    const lines = full.split('\n')
    const head = lines.slice(0, 3).join('\n')
    const compact = Array.from(head).slice(0, 180).join('')
    const collapsed = lines.length > 3 || Array.from(head).length > 180
    return { collapsed, compact: collapsed ? compact : full, full }
  }

  // Plain words, never an animation frame.
  function toolStatusLabel(runningNames) {
    const names = (runningNames || []).filter(Boolean)
    return names.length ? `กำลังใช้ ${names.join(', ')}` : 'กำลังคิด…'
  }

  return { createActivityModel, describeText, toolStatusLabel }
})
