const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const vm = require('node:vm')

// RENDERER_PATH lets a pre-change source be checked as a negative control.
const source = fs.readFileSync(process.env.RENDERER_PATH || path.join(__dirname, '..', 'ui', 'renderer.js'), 'utf8')
const html = fs.readFileSync(path.join(__dirname, '..', 'ui', 'index.html'), 'utf8')
const SAVING = 'กำลังบันทึก…'
const SAVED = 'บันทึกแล้ว ✓'

function harness({ wsOpen = true } = {}) {
  const els = {}
  const timers = []
  const sent = []
  const mkClass = () => ({
    s: new Set(),
    add(c) { this.s.add(c) },
    remove(c) { this.s.delete(c) },
    contains(c) { return this.s.has(c) },
    toggle(c, on) { on ? this.s.add(c) : this.s.delete(c) },
  })
  const element = id => els[id] || (els[id] = { id, textContent: '', className: '', classList: mkClass(), attrs: {} })
  const buttons = ['apple', 'openthai'].map(value => ({
    dataset: { ocr: value }, attrs: {}, disabled: false, title: '', classList: mkClass(),
    setAttribute(k, v) { this.attrs[k] = v },
  }))
  const start = source.indexOf('const OCR_TITLES')
  const code = source.slice(start, source.indexOf('// ── Status ─'))
  const ctx = {
    document: { getElementById: element, querySelectorAll: () => buttons },
    INTERNET_SAVING_TEXT: SAVING, INTERNET_SAVED_TEXT: SAVED,
    setTimeout: (fn, ms) => { timers.push({ fn, ms }); return timers.length },
    wsSend: msg => { sent.push(msg); return wsOpen },
  }
  vm.createContext(ctx)
  vm.runInContext(code, ctx)
  return { ctx, element, timers, sent, buttons }
}

const READY = { apple: { available: true, detail: '' }, openthai: { available: true, detail: '' } }
const NO_OPENTHAI = { apple: { available: true, detail: '' }, openthai: { available: false, detail: 'model files missing in /x: det.onnx' } }

test('the switch has Apple Vision and OpenThai buttons inside Settings', () => {
  assert.match(html, /id="ocr-switch"[\s\S]*data-ocr="apple"[\s\S]*data-ocr="openthai"/)
  assert.ok(html.indexOf('id="ocr-switch"') > html.indexOf('id="left-settings"'))
  assert.match(html, /id="ocr-status"/)
})

test('the switch shows the saved engine and marks it selected', () => {
  const h = harness()
  h.ctx.renderOcrSwitch('openthai', READY, 'openthai')
  assert.equal(h.buttons[1].classList.contains('is-selected'), true)
  assert.equal(h.buttons[1].attrs['aria-checked'], 'true')
  assert.equal(h.buttons[0].attrs['aria-checked'], 'false')
  h.ctx.renderOcrSwitch('apple', READY, 'apple')
  assert.equal(h.buttons[0].classList.contains('is-selected'), true)
  h.ctx.renderOcrSwitch('nonsense', READY, 'apple')
  assert.equal(h.buttons[0].classList.contains('is-selected'), true, 'an unknown value falls back to the default engine')
})

test('an engine that cannot run is disabled and its tooltip says why; the selected one is never disabled', () => {
  const h = harness()
  h.ctx.renderOcrSwitch('apple', NO_OPENTHAI, 'apple')
  assert.equal(h.buttons[1].disabled, true)
  assert.match(h.buttons[1].title, /ยังไม่พร้อม: model files missing in \/x: det\.onnx/)
  assert.equal(h.buttons[0].disabled, false)
  h.ctx.renderOcrSwitch('openthai', NO_OPENTHAI, 'apple')
  assert.equal(h.buttons[1].disabled, false, 'the saved engine stays clickable so the state is not a trap')
})

test('choosing an engine sends set_ocr_engine at once and shows "saving…" until the reply', () => {
  const h = harness()
  h.ctx.renderOcrSwitch('apple', READY, 'apple')
  h.ctx.setOcrEngine('openthai')
  assert.equal(JSON.stringify(h.sent), JSON.stringify([{ type: 'set_ocr_engine', engine: 'openthai' }]))
  assert.equal(h.element('ocr-status').textContent, SAVING)
  h.ctx.applyOcrState({ engine: 'openthai', effective: 'openthai', engines: READY, error: '' })
  assert.equal(h.element('ocr-status').textContent, SAVED)
  assert.equal(h.buttons[1].classList.contains('is-selected'), true)
  h.timers.filter(t => t.ms === 2500).forEach(t => t.fn())
  assert.equal(h.element('ocr-status').textContent, '')
})

test('clicking the engine already selected sends nothing', () => {
  const h = harness()
  h.ctx.setOcrEngine('apple')
  assert.equal(h.sent.length, 0)
})

test('a busy refusal keeps the previous engine and says why, in red', () => {
  const h = harness()
  h.ctx.renderOcrSwitch('apple', READY, 'apple')
  h.ctx.setOcrEngine('openthai')
  h.ctx.applyOcrState({ engine: 'apple', effective: 'apple', engines: READY, error: 'agent is busy' })
  assert.equal(h.buttons[0].classList.contains('is-selected'), true)
  assert.match(h.element('ocr-status').textContent, /agent กำลังทำงาน/)
  assert.ok(h.element('ocr-status').classList.contains('err'))
})

test('a refusal because the engine is not installed is shown as the server wrote it', () => {
  const h = harness()
  h.ctx.setOcrEngine('openthai')
  h.ctx.applyOcrState({ engine: 'apple', effective: 'apple', engines: NO_OPENTHAI, error: 'openthai OCR is not ready: model files missing in /x: det.onnx' })
  assert.match(h.element('ocr-status').textContent, /openthai OCR is not ready/)
  assert.ok(h.element('ocr-status').classList.contains('err'))
  assert.equal(h.buttons[0].classList.contains('is-selected'), true)
})

test('when the saved engine is not the one in use, the status line says so', () => {
  const h = harness()
  h.ctx.renderOcrSwitch('openthai', NO_OPENTHAI, 'apple')
  assert.match(h.element('ocr-status').textContent, /OpenThai ใช้ไม่ได้ตอนนี้ — กำลังใช้ Apple Vision แทน/)
  assert.ok(h.element('ocr-status').classList.contains('err'))
  h.ctx.renderOcrSwitch('openthai', READY, 'openthai')
  assert.equal(h.element('ocr-status').textContent, '')
  assert.equal(h.element('ocr-status').classList.contains('err'), false)
})

test('a status poll must not erase a refusal or the saved confirmation (regression: poll cleared the red message at once)', () => {
  const h = harness()
  h.ctx.renderOcrSwitch('apple', NO_OPENTHAI, 'apple')
  h.ctx.setOcrEngine('openthai')
  h.ctx.applyOcrState({ engine: 'apple', effective: 'apple', engines: NO_OPENTHAI, error: 'openthai OCR is not ready: model files missing in /x: det.onnx' })
  assert.match(h.element('ocr-status').textContent, /not ready/)
  h.ctx.renderOcrSwitch('apple', NO_OPENTHAI, 'apple')   // what applyStatus does on every /status frame
  h.ctx.renderOcrSwitch('apple', NO_OPENTHAI, 'apple')
  assert.match(h.element('ocr-status').textContent, /not ready/, 'the refusal must stay until the owner acts again')
  assert.ok(h.element('ocr-status').classList.contains('err'))
  // a later successful choice replaces it
  h.ctx.setOcrEngine('apple')
  h.ctx.applyOcrState({ engine: 'apple', effective: 'apple', engines: READY, error: '' })
  assert.equal(h.element('ocr-status').textContent, SAVED)
})

test('the "in use differs from saved" notice still appears and clears on status polls', () => {
  const h = harness()
  h.ctx.renderOcrSwitch('openthai', NO_OPENTHAI, 'apple')
  const notice = h.element('ocr-status').textContent
  assert.match(notice, /กำลังใช้ Apple Vision แทน/)
  h.ctx.renderOcrSwitch('openthai', NO_OPENTHAI, 'apple')
  assert.equal(h.element('ocr-status').textContent, notice)
  h.ctx.renderOcrSwitch('openthai', READY, 'openthai')
  assert.equal(h.element('ocr-status').textContent, '', 'the notice goes away once the engine works again')
})

test('a dropped WebSocket or a silent server is reported, not left on "saving…"', () => {
  const closed = harness({ wsOpen: false })
  closed.ctx.setOcrEngine('openthai')
  assert.notEqual(closed.element('ocr-status').textContent, SAVING)
  assert.ok(closed.element('ocr-status').classList.contains('err'))

  const silent = harness()
  silent.ctx.setOcrEngine('openthai')
  silent.timers.filter(t => t.ms === 8000).forEach(t => t.fn())
  assert.notEqual(silent.element('ocr-status').textContent, SAVING)
  assert.ok(silent.element('ocr-status').classList.contains('err'))
})

test('the frame, the status poll and the click are wired in renderer.js', () => {
  assert.match(source, /case 'ocr_engine_state':\s*applyOcrState\(ev\)/)
  assert.match(source, /if \(!ocrPending && typeof s\.ocr_engine === 'string'\) renderOcrSwitch\(s\.ocr_engine, s\.ocr_engines, s\.ocr_effective\)/)
  assert.match(source, /ocrSwitch\.addEventListener\('click'[\s\S]*setOcrEngine\(button\.dataset\.ocr\)/)
})
