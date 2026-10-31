const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const ROOT = path.join(__dirname, '..')
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8')
const renderer = fs.readFileSync(path.join(ROOT, 'renderer.js'), 'utf8')

test('the activity library loads before the renderer that uses it', () => {
  const lib = html.indexOf('lib/activity_log.js')
  assert.ok(lib > -1)
  assert.ok(lib < html.indexOf('src="renderer.js"'))
})

test('renderer draws the activity log from the model and attaches tool results', () => {
  assert.match(renderer, /ActivityLog\.createActivityModel\(/)
  assert.match(renderer, /case 'tool_result':/)
  assert.match(renderer, /finishActivity\(ev\)/)
  assert.match(renderer, /activityModel\.snapshot\(\)/)
  // call_id and the redacted arguments from the server are what the log shows.
  assert.match(renderer, /ev\.call_id/)
  assert.match(renderer, /ev\.arguments/)
})

test('status while a tool runs is plain text: no spinner frames or timers remain', () => {
  assert.doesNotMatch(renderer, /startSpin|stopSpin|FRAMES|spinTimer/)
  assert.doesNotMatch(renderer, /ts-frame/)
  assert.doesNotMatch(html, /ts-frame/)
  assert.match(renderer, /ActivityLog\.toolStatusLabel\(/)
})

test('activity text is written with textContent, never innerHTML', () => {
  const start = renderer.indexOf('function _activityText(')
  const end = renderer.indexOf('function renderActivity()')
  const block = renderer.slice(start, end)
  assert.doesNotMatch(block, /innerHTML/)
  const draw = renderer.slice(end, renderer.indexOf('// ── Messages ', end))
  assert.doesNotMatch(draw, /innerHTML/)
})
