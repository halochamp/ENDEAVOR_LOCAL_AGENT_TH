const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const { normalizeState, focusIsPersistent, focusStatus, folderCount } = require('../lib/edit_access')

test('Focus state remains separate from persistent Approved Edit Folders', () => {
  const state = normalizeState({
    folders: ['/approved'],
    focus_folder: '/focused',
    max_folders: 10,
  })
  assert.equal(focusIsPersistent(state), false)
  assert.equal(focusStatus(state), 'Temporary')
  assert.equal(folderCount(state), '1 / 10')
  assert.deepEqual(state.folders, ['/approved'])
})

test('Edit Access settings are backend-driven and wired through Electron IPC', () => {
  const root = path.join(__dirname, '..')
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8')
  const renderer = fs.readFileSync(path.join(root, 'renderer.js'), 'utf8')
  const main = fs.readFileSync(path.join(root, 'main.js'), 'utf8')
  const preload = fs.readFileSync(path.join(root, 'preload.js'), 'utf8')

  assert.ok(html.indexOf('lib/edit_access.js') < html.indexOf('renderer.js'))
  assert.match(html, /id="approved-edit-folders-add"/)
  assert.match(html, /id="focus-folder-select"/)
  assert.match(html, /id="btn-focus"/)
  assert.ok(html.indexOf('id="btn-focus"') > html.indexOf('id="btn-workspace"'))
  assert.match(html, /id="left-focus"/)
  assert.match(html, /focus-files-select/)
  assert.match(renderer, /get_approved_edit_folders/)
  assert.match(renderer, /add_approved_edit_folders/)
  assert.match(renderer, /set_focus_folder/)
  assert.match(renderer, /remove_approved_edit_folder/)
  assert.match(html, /Active Workspace.*temporary/i)
  assert.match(main, /ipcMain\.handle\('show-approved-edit-folder-dialog'/)
  assert.match(main, /route: '\/edit-access\/authorize'/)
  assert.match(main, /route: '\/edit-access\/delete'/)
  assert.match(preload, /showApprovedEditFolderDialog: \(\) => ipcRenderer\.invoke\('show-approved-edit-folder-dialog'\)/)
  assert.match(preload, /setRuntimeSettings: \(payload\) => ipcRenderer\.invoke\('set-runtime-settings', payload\)/)
})

test('Focus browsing and mentions use a temporary focus scope', () => {
  const root = path.join(__dirname, '..')
  const renderer = fs.readFileSync(path.join(root, 'renderer.js'), 'utf8')
  const server = fs.readFileSync(path.join(root, '..', 'agent_server.py'), 'utf8')
  assert.match(renderer, /scope: 'focus'/)
  assert.match(renderer, /focus_mentions/)
  assert.match(server, /def _safe_focus_real/)
  assert.match(server, /def _focus_mention_paths/)
  assert.match(server, /Active Workspace โดยตรง/)
})
