const test = require('node:test')
const assert = require('node:assert')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { isInsideWorkspace } = require('../lib/workspace_guard')

function mkTmpDir(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix))
}

test('accepts a file directly inside the workspace', () => {
  const ws = mkTmpDir('ws-')
  const file = path.join(ws, 'note.txt')
  fs.writeFileSync(file, 'hi')
  assert.strictEqual(isInsideWorkspace(file, ws), true)
  fs.rmSync(ws, { recursive: true, force: true })
})

test('accepts a nested file inside the workspace', () => {
  const ws = mkTmpDir('ws-')
  const dir = path.join(ws, 'sub', 'dir')
  fs.mkdirSync(dir, { recursive: true })
  const file = path.join(dir, 'note.txt')
  fs.writeFileSync(file, 'hi')
  assert.strictEqual(isInsideWorkspace(file, ws), true)
  fs.rmSync(ws, { recursive: true, force: true })
})

test('accepts the workspace root itself', () => {
  const ws = mkTmpDir('ws-')
  assert.strictEqual(isInsideWorkspace(ws, ws), true)
  fs.rmSync(ws, { recursive: true, force: true })
})

test('rejects a path traversal escape via ../', () => {
  const root = mkTmpDir('root-')
  const ws = path.join(root, 'workspace')
  fs.mkdirSync(ws)
  const outside = path.join(root, 'secret.txt')
  fs.writeFileSync(outside, 'nope')
  const escaped = path.join(ws, '..', 'secret.txt')
  assert.strictEqual(isInsideWorkspace(escaped, ws), false)
  fs.rmSync(root, { recursive: true, force: true })
})

test('rejects a symlink that points outside the workspace', () => {
  const root = mkTmpDir('root-')
  const ws = path.join(root, 'workspace')
  fs.mkdirSync(ws)
  const outside = path.join(root, 'secret.txt')
  fs.writeFileSync(outside, 'nope')
  const link = path.join(ws, 'link.txt')
  fs.symlinkSync(outside, link)
  assert.strictEqual(isInsideWorkspace(link, ws), false)
  fs.rmSync(root, { recursive: true, force: true })
})

test('rejects a non-existent path', () => {
  const ws = mkTmpDir('ws-')
  assert.strictEqual(isInsideWorkspace(path.join(ws, 'missing.txt'), ws), false)
  fs.rmSync(ws, { recursive: true, force: true })
})
