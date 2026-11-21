const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const { rust, bridge, rustFn } = require('./shell_sources')
const vm = require('node:vm')

const ROOT = path.join(__dirname, '..', 'ui')
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8')
const renderer = fs.readFileSync(process.env.RENDERER_PATH || path.join(ROOT, 'renderer.js'), 'utf8')

const START = '// ── Text to Speech ─'
const END = '// ── end Text to Speech ─'

function harness(api = {}, confirmAnswers = null) {
  const asked = []
  const opened = []
  const els = {}
  const sent = []
  const mkClass = () => ({ s: new Set(), add(c) { this.s.add(c) }, remove(c) { this.s.delete(c) }, contains(c) { return this.s.has(c) }, toggle(c, on) { on ? this.s.add(c) : this.s.delete(c) } })
  const element = id => els[id] || (els[id] = {
    id, textContent: '', value: '', disabled: false, classList: mkClass(), children: [],
    set innerHTML(v) { this.children = [] }, get innerHTML() { return '' },
    appendChild(child) { this.children.push(child); if (!this.value && this.children.length === 1) this.value = child.value },
  })
  ;['tts-voice', 'tts-voices-note', 'tts-speed', 'tts-speed-label', 'tts-start-btn', 'tts-pause-btn', 'tts-resume-btn', 'tts-cancel-btn',
    'tts-status', 'tts-progress', 'tts-result', 'tts-result-stats', 'tts-result-path', 'tts-result-warning'].forEach(element)
  els['tts-speed'].value = '1'
  const code = renderer.slice(renderer.indexOf(START), renderer.indexOf(END))
  assert.ok(code.length > 500, 'the Text to Speech block must exist in renderer.js')
  const ctx = {
    document: { getElementById: element, createElement: () => ({ value: '', textContent: '' }) },
    window: { electronAPI: api },
    askConfirm: () => { asked.push(1); return Promise.resolve(confirmAnswers.shift()) },   // the Rust shell's native yes/no
    togglePanel: name => { opened.push(name) },
    wsSend: msg => { sent.push(msg); return true },
    setTimeout: (fn) => { fn(); return 0 },
    Promise, Math, Number, Array, String, console,
  }
  vm.createContext(ctx)
  vm.runInContext(code, ctx)
  const get = name => vm.runInContext(name, ctx)
  if (!confirmAnswers) vm.runInContext('ttsState.modelConfirmed = true', ctx)   // every flow below starts after the question was answered yes
  return { ctx, els, sent, get, asked, opened }
}

const READY = { ok: true, engine: { ready: true, reason: '' }, voices: [{ id: 'anna', label: 'Anna' }, { id: 'bob', label: 'bob' }], voices_dir: '/x/voices' }

test('the sidebar has a TTS button and a panel with the controls, voice picker and speed slider', () => {
  for (const id of ['btn-tts', 'left-tts', 'tts-voice', 'tts-speed', 'tts-start-btn', 'tts-pause-btn', 'tts-resume-btn', 'tts-cancel-btn',
    'tts-status', 'tts-result', 'tts-open-output-btn', 'tts-refresh-btn']) {
    assert.ok(html.includes(`id="${id}"`), `index.html must have #${id}`)
  }
  assert.match(html, /id="tts-speed"[^>]*min="0.8"[^>]*max="1.2"/)
  assert.match(html, /CC-BY-NC/)
  assert.match(renderer, /on\('#btn-tts', \(\) => togglePanel\('tts'\)\)/)
  assert.match(renderer, /if \(name === 'tts'\) refreshTtsVoices\(\)/)
  for (const name of ['tts-start-btn:startTts', 'tts-pause-btn:pauseTts', 'tts-resume-btn:resumeTts', 'tts-cancel-btn:cancelTts', 'tts-open-output-btn:openTtsOutput', 'tts-refresh-btn:refreshTtsVoices']) {
    const [id, fn] = name.split(':')
    assert.match(renderer, new RegExp(`on\\('#${id}', ${fn}\\)`), `${id} must call ${fn}`)
  }
})

test('the bridge exposes the TTS calls and the Rust shell handles them', () => {
  for (const fn of ['ttsVoices', 'ttsStart', 'ttsStatus', 'ttsPause', 'ttsResume', 'ttsCancel']) assert.ok(bridge.includes(`${fn}:`), `bridge.js must expose ${fn}`)
  assert.match(bridge, /ttsStart: function \(voice, speed\) \{ return invoke\('tts_start'/)
  for (const route of ['/tts/voices', '/tts/status', '/tts/pause', '/tts/resume', '/tts/cancel']) assert.ok(rust.includes(`"${route}"`), `the Rust shell must allow ${route}`)
  const start = rustFn('tts_start')
  assert.match(rust, /"\/tts\/start"/)
  assert.match(start, /"X-Voice", clean_voice\(&voice\)/)
  assert.match(start, /&\["txt", "md", "markdown"\]/)
})

test('voices fill the picker; an engine that is not ready, or no voices, says what to do and keeps Start disabled', () => {
  const h = harness()
  h.ctx.renderTtsVoices(READY)
  assert.deepEqual(h.els['tts-voice'].children.map(o => o.value), ['anna', 'bob'])
  assert.equal(h.els['tts-voices-note'].textContent, '')
  assert.equal(h.els['tts-start-btn'].disabled, false)
  h.ctx.renderTtsVoices({ ok: true, engine: { ready: false, reason: 'The OmniVoice model files were not found' }, voices: [], voices_dir: '/x' })
  assert.match(h.els['tts-voices-note'].textContent, /ยังใช้ไม่ได้: The OmniVoice model files/)
  assert.equal(h.els['tts-start-btn'].disabled, true)
  h.ctx.renderTtsVoices({ ok: true, engine: { ready: true }, voices: [], voices_dir: '/x/voices' })
  assert.match(h.els['tts-voices-note'].textContent, /ref_voice\.wav[\s\S]*ref_voice\.txt[\s\S]*\/x\/voices/)
  assert.equal(h.els['tts-voice'].disabled, true)
})

test('the controls follow the job status', () => {
  const h = harness()
  h.ctx.renderTtsVoices(READY)
  const state = status => {
    h.ctx.renderTtsState({ status, message: status })
    return ['start', 'pause', 'resume', 'cancel'].map(n => !h.els[`tts-${n}-btn`].disabled)
  }
  assert.deepEqual(state('idle'), [true, false, false, false])
  assert.deepEqual(state('running'), [false, true, false, true])
  assert.deepEqual(state('pausing'), [false, false, true, true])
  assert.deepEqual(state('paused'), [false, false, true, true])
  assert.deepEqual(state('cancelling'), [false, false, false, true])
  assert.deepEqual(state('done'), [true, false, false, false])
  assert.deepEqual(state('cancelled'), [true, false, false, false])
})

test('progress shows the file, the piece and the time left; a finished job shows its result and warnings', () => {
  const h = harness()
  h.ctx.renderTtsVoices(READY)
  h.ctx.renderTtsState({ status: 'running', message: 'กำลังอ่านออกเสียง', source_name: 'story.md', done: 2, total: 5, eta_sec: 180 })
  assert.match(h.els['tts-progress'].textContent, /story\.md · ชิ้น 2\/5 · เหลือราว 3 นาที/)
  h.ctx.renderTtsState({ job_id: 'tts-1', status: 'done', message: 'อ่านออกเสียงเสร็จแล้ว', total: 5, done: 5, duration_sec: 95, voice: 'anna', output_path: '/w/story.m4a', warnings: ['Skipped 1 code block(s)'] })
  assert.ok(h.els['tts-result'].classList.contains('show'))
  assert.match(h.els['tts-result-stats'].textContent, /5 ชิ้น · ความยาว 1\.6 นาที · เสียง anna/)
  assert.equal(h.els['tts-result-path'].textContent, '/w/story.m4a')
  assert.match(h.els['tts-result-warning'].textContent, /⚠ Skipped 1 code block/)
  h.ctx.openTtsOutput()
  assert.equal(JSON.stringify(h.sent), JSON.stringify([{ type: 'open_with_os', path: '/w/story.m4a' }]))
})

test('an error shows the reason in red and the panel can start again', () => {
  const h = harness()
  h.ctx.renderTtsVoices(READY)
  h.ctx.renderTtsState({ status: 'error', message: 'Text to Speech ไม่สำเร็จ', error: 'No speakable text was found in this file' })
  assert.match(h.els['tts-status'].textContent, /No speakable text/)
  assert.ok(h.els['tts-status'].classList.contains('error'))
  assert.equal(h.els['tts-start-btn'].disabled, false)
})

test('Start sends the chosen voice and speed, then polls until done', async () => {
  const calls = []
  const jobs = [{ status: 'running', job_id: 'j1', done: 1, total: 2 }, { status: 'done', job_id: 'j1', done: 2, total: 2, output_path: '/w/a.m4a', duration_sec: 5, voice: 'anna' }]
  const h = harness({
    ttsStart: async (voice, speed) => { calls.push(['start', voice, speed]); return { ok: true, job: { job_id: 'j1', status: 'queued' } } },
    ttsStatus: async id => { calls.push(['status', id]); return { ok: true, job: jobs.shift() } },
  })
  h.ctx.renderTtsVoices(READY)
  h.els['tts-voice'].value = 'bob'
  h.els['tts-speed'].value = '1.1'
  await h.ctx.startTts()
  assert.deepEqual(calls[0], ['start', 'bob', 1.1])
  assert.deepEqual(calls.slice(1), [['status', 'j1'], ['status', 'j1']])
  assert.equal(h.get('ttsState').status, 'done')
  assert.equal(h.get('ttsState').outputPath, '/w/a.m4a')
})

test('Start without a voice, with a cancelled dialog, or with a server error does not leave the panel stuck', async () => {
  const none = harness({ ttsStart: async () => assert.fail('must not start without a voice'), ttsStatus: async () => ({}) })
  await none.ctx.startTts()
  assert.match(none.els['tts-status'].textContent, /ยังไม่มีเสียง/)

  const cancelled = harness({ ttsStart: async () => ({ ok: false, cancelled: true }), ttsStatus: async () => ({}) })
  cancelled.ctx.renderTtsVoices(READY)
  await cancelled.ctx.startTts()
  assert.equal(cancelled.get('ttsState').status, 'idle')
  assert.equal(cancelled.els['tts-start-btn'].disabled, false)

  const refused = harness({ ttsStart: async () => ({ ok: false, error: 'เลือกเสียงที่ใช้ได้ก่อน' }), ttsStatus: async () => ({}) })
  refused.ctx.renderTtsVoices(READY)
  await refused.ctx.startTts()
  assert.match(refused.els['tts-status'].textContent, /เลือกเสียงที่ใช้ได้ก่อน/)
  assert.equal(refused.els['tts-start-btn'].disabled, false)
})

test('Pause, Resume and Cancel call the bridge with the job id and show the answer', async () => {
  const calls = []
  const reply = status => async id => { calls.push([status, id]); return { ok: true, job: { job_id: id, status, message: status } } }
  const h = harness({ ttsPause: reply('pausing'), ttsResume: reply('running'), ttsCancel: reply('cancelling'), ttsStatus: async () => ({ ok: true, job: { status: 'cancelled', job_id: 'j9' } }) })
  h.ctx.renderTtsVoices(READY)
  h.ctx.renderTtsState({ job_id: 'j9', status: 'running' })
  await h.get('pauseTts')()
  await h.get('resumeTts')()
  await h.get('cancelTts')()
  assert.deepEqual(calls.map(c => c[0]), ['pausing', 'running', 'cancelling'])
  assert.ok(calls.every(c => c[1] === 'j9'))
})

test('the speed slider label follows the slider', () => {
  const h = harness()
  h.els['tts-speed'].value = '0.85'
  h.ctx.updateTtsSpeedLabel()
  assert.equal(h.els['tts-speed-label'].textContent, '0.85×')
})

test('opening the TTS tab asks once whether to load the voice model; a "no" keeps it closed and loads nothing', async () => {
  assert.match(renderer, /if \(name === 'tts' && activePanel !== 'tts' && !ttsPanelMayOpen\(\)\) return/)
  const h = harness({}, [false, true])
  assert.equal(h.ctx.ttsPanelMayOpen(), false)               // not answered yet: asks, nothing opens meanwhile
  await new Promise(r => setImmediate(r))
  assert.deepEqual(h.opened, [], 'declined: the panel stays closed')
  assert.equal(h.ctx.ttsPanelMayOpen(), false)               // and the next open asks again
  await new Promise(r => setImmediate(r))
  assert.deepEqual(h.opened, ['tts'], 'accepted: the panel opens by itself')
  assert.equal(h.ctx.ttsPanelMayOpen(), true)                // remembered: not asked a third time
  assert.equal(h.asked.length, 2)
  const text = vm.runInContext('TTS_MODEL_PROMPT', h.ctx)
  assert.match(text, /OmniVoice/)
  assert.match(text, /RAM/)
  assert.match(text, /ปล่อยออกเมื่ออ่านเสร็จ/)
})

test('a reading cannot start before the question was answered yes', async () => {
  const h = harness({ ttsStart: async () => assert.fail('must not start'), ttsStatus: async () => ({}) }, [])
  h.ctx.renderTtsVoices(READY)
  await h.ctx.startTts()
  assert.match(h.els['tts-status'].textContent, /ยังไม่ได้ยืนยัน/)
})
