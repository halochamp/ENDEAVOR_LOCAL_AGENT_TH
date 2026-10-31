const assert = require('node:assert/strict')
const test = require('node:test')

const { createActivityModel, describeText, toolStatusLabel } = require('../lib/activity_log')

const call = (id, name = 'bash', extra = {}) => ({ callId: id, name, detail: '', args: '', ts: '10:00:00', ...extra })

test('newest turn and newest call come first', () => {
  const m = createActivityModel()
  m.startTurn('10:00:00')
  m.addCall(call('a', 'read_file'))
  m.addCall(call('b', 'bash'))
  m.startTurn('10:05:00')
  m.addCall(call('c', 'web_search'))
  const turns = m.snapshot()
  assert.deepEqual(turns.map(t => t.ts), ['10:05:00', '10:00:00'])
  assert.deepEqual(turns[0].entries.map(e => e.name), ['web_search'])
  assert.deepEqual(turns[1].entries.map(e => e.name), ['bash', 'read_file'])
})

test('a result attaches to its call even after later turns started', () => {
  const m = createActivityModel()
  m.startTurn('t1')
  m.addCall(call('a'))
  m.startTurn('t2')
  const updated = m.finishCall({ callId: 'a', status: 'returned', preview: 'line1\nline2', truncated: true })
  assert.equal(updated.status, 'returned')
  const old = m.snapshot()[1].entries[0]
  assert.equal(old.preview, 'line1\nline2')
  assert.equal(old.truncated, true)
})

test('calls start as running and a failed result is recorded as failed', () => {
  const m = createActivityModel()
  m.addCall(call('a'))
  assert.equal(m.snapshot()[0].entries[0].status, 'running')
  m.finishCall({ callId: 'a', status: 'failed', preview: '[error] boom', truncated: false })
  assert.equal(m.snapshot()[0].entries[0].status, 'failed')
})

test('a call without a started turn creates one implicitly', () => {
  const m = createActivityModel()
  m.addCall(call('a'))
  assert.equal(m.snapshot().length, 1)
})

test('unknown results are ignored without changing anything', () => {
  const m = createActivityModel()
  m.addCall(call('a'))
  assert.equal(m.finishCall({ callId: 'zzz', status: 'returned', preview: 'x' }), null)
  assert.equal(m.snapshot()[0].entries[0].status, 'running')
})

test('plans are entries of the current turn, newest first', () => {
  const m = createActivityModel()
  m.startTurn('t1')
  m.addCall(call('a'))
  m.addPlan(['one', 'two'], '10:01:00')
  const entries = m.snapshot()[0].entries
  assert.equal(entries[0].kind, 'plan')
  assert.deepEqual(entries[0].steps, ['one', 'two'])
  assert.equal(entries[1].kind, 'call')
})

test('caps drop the oldest entries first and never leave an empty turn', () => {
  const m = createActivityModel({ maxTurns: 3, maxEntries: 5 })
  for (let t = 0; t < 5; t++) {
    m.startTurn(`t${t}`)
    m.addCall(call(`c${t}a`))
    m.addCall(call(`c${t}b`))
  }
  const turns = m.snapshot()
  assert.ok(turns.length <= 3)
  const total = turns.reduce((n, t) => n + t.entries.length, 0)
  assert.ok(total <= 5, `total ${total}`)
  assert.ok(turns.every(t => t.entries.length > 0))
  assert.equal(turns[0].ts, 't4')
  assert.equal(turns[0].entries[0].id, 'c4b')
})

test('clear removes everything and a new call starts a fresh turn', () => {
  const m = createActivityModel()
  m.addCall(call('a'))
  m.clear()
  assert.deepEqual(m.snapshot(), [])
  m.addCall(call('b'))
  assert.equal(m.snapshot().length, 1)
})

test('describeText collapses more than three lines or 180 characters', () => {
  assert.deepEqual(describeText('short'), { collapsed: false, compact: 'short', full: 'short' })
  const lines = describeText('1\n2\n3\n4')
  assert.equal(lines.collapsed, true)
  assert.equal(lines.compact, '1\n2\n3')
  assert.equal(lines.full, '1\n2\n3\n4')
  const long = describeText('x'.repeat(400))
  assert.equal(long.collapsed, true)
  assert.equal(long.compact.length, 180)
  assert.equal(describeText('').collapsed, false)
})

test('status text is plain words, not an animation frame', () => {
  assert.equal(toolStatusLabel([]), 'กำลังคิด…')
  assert.equal(toolStatusLabel(['bash']), 'กำลังใช้ bash')
  assert.equal(toolStatusLabel(['bash', 'read_file']), 'กำลังใช้ bash, read_file')
})
