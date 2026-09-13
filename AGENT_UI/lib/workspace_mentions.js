(function (root, factory) {
  const api = factory()
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  else root.WorkspaceMentions = api
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function _relativePath(file) {
    return String((file && file.relative_path) || '').replace(/\\/g, '/').replace(/^\.\//, '')
  }

  function _name(file) {
    const rel = _relativePath(file)
    return String((file && file.name) || (rel ? rel.split('/').pop() : ''))
  }

  function findMentionContext(value, caret) {
    const text = String(value || '')
    const end = Number.isInteger(caret) ? Math.max(0, Math.min(caret, text.length)) : text.length
    const prefix = text.slice(0, end)
    const match = prefix.match(/(?:^|\s)@([^\s@{}]*)$/u)
    if (!match) return null
    const query = match[1] || ''
    return { start: end - query.length - 1, end, query }
  }

  function _allCandidates(files) {
    const safeFiles = Array.isArray(files) ? files : []
    const nameCounts = new Map()
    for (const file of safeFiles) {
      const name = _name(file)
      if (!name) continue
      const key = name.toLocaleLowerCase()
      nameCounts.set(key, (nameCounts.get(key) || 0) + 1)
    }
    const out = []
    for (const file of safeFiles) {
      const relativePath = _relativePath(file)
      const name = _name(file)
      if (!relativePath || !name) continue
      const duplicateName = (nameCounts.get(name.toLocaleLowerCase()) || 0) > 1
      const reference = duplicateName ? relativePath : name
      const token = /\s/u.test(reference) ? `@{${reference}}` : `@${reference}`
      out.push({
        kind: 'mention',
        name,
        relativePath,
        token,
        desc: duplicateName ? relativePath : (relativePath === name ? 'Workspace' : relativePath),
      })
    }
    return out
  }

  function buildMentionCandidates(files, query, limit = 20) {
    const needle = String(query || '').toLocaleLowerCase()
    const candidates = _allCandidates(files).filter(item => {
      if (!needle) return true
      return item.name.toLocaleLowerCase().includes(needle)
        || item.relativePath.toLocaleLowerCase().includes(needle)
    })
    candidates.sort((a, b) => {
      const aStarts = a.name.toLocaleLowerCase().startsWith(needle) ? 0 : 1
      const bStarts = b.name.toLocaleLowerCase().startsWith(needle) ? 0 : 1
      if (aStarts !== bStarts) return aStarts - bStarts
      return a.relativePath.localeCompare(b.relativePath, undefined, { sensitivity: 'base' })
    })
    return candidates.slice(0, Math.max(1, Number(limit) || 20))
  }

  function _tokenPresent(text, token) {
    let from = 0
    while (true) {
      const index = text.indexOf(token, from)
      if (index < 0) return false
      const before = index === 0 ? '' : text[index - 1]
      const after = text[index + token.length] || ''
      const beforeOk = !before || /\s|[([{]/u.test(before)
      const afterOk = !after || /\s|[.,!?;:)}\]]/u.test(after)
      if (beforeOk && afterOk) return true
      from = index + token.length
    }
  }

  function extractMentionPaths(text, files) {
    const source = String(text || '')
    const seen = new Set()
    const paths = []
    for (const item of _allCandidates(files)) {
      if (_tokenPresent(source, item.token) && !seen.has(item.relativePath)) {
        seen.add(item.relativePath)
        paths.push(item.relativePath)
      }
    }
    return paths
  }

  function insertMention(value, start, end, token) {
    const text = String(value || '')
    const left = text.slice(0, Math.max(0, start))
    const right = text.slice(Math.max(start, end))
    const spacer = !right || !/^\s/u.test(right) ? ' ' : ''
    const nextValue = left + token + spacer + right
    return { value: nextValue, caret: left.length + token.length + spacer.length }
  }

  return { findMentionContext, buildMentionCandidates, extractMentionPaths, insertMention }
})
