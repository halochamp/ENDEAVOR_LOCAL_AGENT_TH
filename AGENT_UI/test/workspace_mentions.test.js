const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const {
  findMentionContext,
  buildMentionCandidates,
  extractMentionPaths,
  insertMention,
} = require('../lib/workspace_mentions')

const files = [
  { name: 'world.doc', relative_path: 'world.doc' },
  { name: 'report.pdf', relative_path: 'docs/report.pdf' },
  { name: 'same.txt', relative_path: 'a/same.txt' },
  { name: 'same.txt', relative_path: 'b/same.txt' },
  { name: 'My File.docx', relative_path: 'notes/My File.docx' },
]

test('@ context is detected at the caret inside ordinary prose', () => {
  const active = 'ช่วยสรุป @wor'
  assert.deepEqual(findMentionContext(active, active.length), { start: 9, end: active.length, query: 'wor' })
  const finished = 'ช่วยสรุป @world.doc ต่อ'
  assert.equal(findMentionContext(finished, finished.length), null)
})

test('unique names use short @filename while duplicates use relative paths', () => {
  const all = buildMentionCandidates(files, '', 20)
  assert.equal(all.find(x => x.relativePath === 'world.doc').token, '@world.doc')
  assert.equal(all.find(x => x.relativePath === 'a/same.txt').token, '@a/same.txt')
  assert.equal(all.find(x => x.relativePath === 'b/same.txt').token, '@b/same.txt')
  assert.equal(all.find(x => x.relativePath === 'notes/My File.docx').token, '@{My File.docx}')
})

test('mention search matches filename or nested relative path', () => {
  assert.deepEqual(
    buildMentionCandidates(files, 'repo', 20).map(x => x.relativePath),
    ['docs/report.pdf'],
  )
  assert.deepEqual(
    buildMentionCandidates(files, 'a/s', 20).map(x => x.relativePath),
    ['a/same.txt'],
  )
})

test('selected mention replaces only the active @ token and keeps following prose', () => {
  const result = insertMention('สรุป @wor ให้หน่อย', 5, 9, '@world.doc')
  assert.equal(result.value, 'สรุป @world.doc ให้หน่อย')
  assert.equal(result.caret, 'สรุป @world.doc'.length)
})

test('query extraction resolves only unambiguous visible mention tokens', () => {
  assert.deepEqual(
    extractMentionPaths('@world.doc สรุปไฟล์ให้หน่อย และ @a/same.txt ด้วย', files),
    ['world.doc', 'a/same.txt'],
  )
  assert.deepEqual(extractMentionPaths('@same.txt สรุป', files), [])
  assert.deepEqual(extractMentionPaths('@{My File.docx} สรุป', files), ['notes/My File.docx'])
})

test('Electron sidebar calls the tab Workspace and loads the mention helper before renderer', () => {
  const root = path.join(__dirname, '..')
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8')
  const renderer = fs.readFileSync(path.join(root, 'renderer.js'), 'utf8')
  assert.match(html, /id="btn-workspace"[\s\S]*icon-btn-label">Agent<br>Workspace</)
  assert.ok(html.indexOf('lib/workspace_mentions.js') < html.indexOf('renderer.js'))
  assert.match(renderer, /type: 'get_workspace_mentions'/)
  assert.match(renderer, /workspace_mentions: workspaceMentions/)
})
