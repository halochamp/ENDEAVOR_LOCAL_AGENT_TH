;(function (root, factory) {
  const api = factory()
  if (typeof module === 'object' && module.exports) module.exports = api
  else root.EditAccess = api
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function normalizeState(payload) {
    const value = payload && typeof payload === 'object' ? payload : {}
    const folders = Array.isArray(value.folders)
      ? value.folders.filter(p => typeof p === 'string' && p.trim()).map(p => p.trim())
      : []
    const focus = typeof value.focus_folder === 'string' ? value.focus_folder.trim() : ''
    const max = Number.isInteger(value.max_folders) && value.max_folders > 0
      ? value.max_folders
      : 10
    return {
      folders: [...new Set(folders)],
      focus_folder: focus,
      max_folders: max,
      error: typeof value.error === 'string' ? value.error : '',
    }
  }

  function focusIsPersistent(state) {
    const normalized = normalizeState(state)
    return !!normalized.focus_folder && normalized.folders.includes(normalized.focus_folder)
  }

  function focusStatus(state) {
    const normalized = normalizeState(state)
    if (!normalized.focus_folder) return 'No Active Workspace'
    return focusIsPersistent(normalized) ? 'Approved + focused' : 'Temporary'
  }

  function folderCount(state) {
    const normalized = normalizeState(state)
    return `${normalized.folders.length} / ${normalized.max_folders}`
  }

  return { normalizeState, focusIsPersistent, focusStatus, folderCount }
})
