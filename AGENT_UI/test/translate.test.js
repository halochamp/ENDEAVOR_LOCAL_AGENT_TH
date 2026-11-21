const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const { rust, bridge, rustFn } = require('./shell_sources')
const vm = require('node:vm')

const ROOT = path.join(__dirname, '..', 'ui')
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8')
const renderer = fs.readFileSync(path.join(ROOT, 'renderer.js'), 'utf8')

test('resuming a restored paused job restarts status polling', async () => {
  let polled = 0
  const context = vm.createContext({
    translateState: { jobId: 'restored' },
    window: { electronAPI: { translateResume: async () => ({ ok: true, job: { status: 'queued' } }) } },
    renderTranslateState() {},
    _pollTranslateJob: async () => { polled++; return { status: 'done' } },
    wsSend() {}, workspaceRoot: '',
  })
  vm.runInContext(renderer.slice(renderer.indexOf('async function resumeTranslate()'),
    renderer.indexOf('async function cancelTranslate()')), context)
  await context.resumeTranslate()
  assert.equal(polled, 1)
})

test('multiple callers share one status poll loop', async () => {
  let loops = 0
  let finish
  const context = vm.createContext({ _pollTranslateJobLoop: () => {
    loops++
    return new Promise(resolve => { finish = resolve })
  } })
  vm.runInContext(renderer.slice(renderer.indexOf('let translatePollPromise = null'),
    renderer.indexOf('async function _pollTranslateJobLoop()')), context)
  const first = context._pollTranslateJob()
  assert.equal(first, context._pollTranslateJob())
  assert.equal(loops, 1)
  finish({ status: 'done' })
  await first
})

test('Electron exposes a dedicated Translate to Thai tab', () => {
  assert.match(html, /id="btn-translate"/)
  assert.match(html, /id="left-translate"/)
  assert.match(html, /id="translate-start-btn"/)
  assert.match(html, /id="translate-open-output-btn"/)
})

test('renderer wires Translate controls without routing through chat', () => {
  assert.match(renderer, /on\('#btn-translate', \(\) => togglePanel\('translate'\)\)/)
  assert.match(renderer, /on\('#translate-start-btn', startTranslate\)/)
  assert.match(renderer, /async function startTranslate\(\)/)
  assert.match(renderer, /window\.electronAPI\.translateStart\(\)/)
  assert.match(renderer, /window\.electronAPI\.translateStatus\(translateState\.jobId\)/)
  assert.match(renderer, /function openTranslateOutput\(\)/)
  const start = renderer.slice(renderer.indexOf('async function startTranslate()'), renderer.indexOf('function openTranslateOutput()'))
  assert.doesNotMatch(start, /sendMessage\(/)
  assert.doesNotMatch(start, /wsSend\(\{ type: ['"]chat/)
})

test('the bridge and the Rust shell keep Translate IPC authenticated and file scoped', () => {
  assert.match(bridge, /translateStart: function \(\) \{ return invoke\('translate_start'\) \}/)
  assert.match(bridge, /translatePause: function \(jobId\) \{ return invoke\('agent_job', \{ route: '\/translate\/pause'/)
  assert.match(bridge, /translateCancel: function \(jobId\) \{ return invoke\('agent_job', \{ route: '\/translate\/cancel'/)
  const start = rustFn('translate_start')
  assert.match(start, /"\/translate\/start"/)
  assert.match(start, /"X-Target-Language", "th"/)
  assert.match(start, /Duration::from_secs\(600\)/)
})
