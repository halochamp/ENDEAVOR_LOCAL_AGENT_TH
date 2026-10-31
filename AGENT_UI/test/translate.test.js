const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const vm = require('node:vm')

const ROOT = path.join(__dirname, '..')
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8')
const renderer = fs.readFileSync(path.join(ROOT, 'renderer.js'), 'utf8')
const preload = fs.readFileSync(path.join(ROOT, 'preload.js'), 'utf8')
const main = fs.readFileSync(path.join(ROOT, 'main.js'), 'utf8')

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

test('preload and main keep Translate IPC authenticated and file scoped', () => {
  assert.match(preload, /translateStart: \(\) => ipcRenderer\.invoke\('translate-start'/)
  assert.match(preload, /translatePause: \(jobId\) => ipcRenderer\.invoke\('translate-pause'/)
  assert.match(preload, /translateCancel: \(jobId\) => ipcRenderer\.invoke\('translate-cancel'/)
  assert.match(main, /ipcMain\.handle\('translate-start'/)
  assert.match(main, /route: '\/translate\/start'/)
  assert.match(main, /'X-Target-Language': 'th'/)
  assert.match(main, /timeoutMs: 600_000/)
})
