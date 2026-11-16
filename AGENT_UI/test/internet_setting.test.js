const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const vm = require('node:vm')

// RENDERER_PATH lets a pre-change source be checked as a negative control.
const source = fs.readFileSync(process.env.RENDERER_PATH || path.join(__dirname, '..', 'renderer.js'), 'utf8')
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8')
const SAVING = 'กำลังบันทึก…'

function harness({ wsOpen = true } = {}) {
  const els = {}
  const timers = []
  const sent = []
  const mkClass = () => ({ s: new Set(), add(c) { this.s.add(c) }, remove(c) { this.s.delete(c) }, contains(c) { return this.s.has(c) } })
  const element = id => els[id] || (els[id] = { id, textContent: '', className: '', classList: mkClass(), attrs: {} })
  const buttons = ['online', 'offline'].map(value => ({
    dataset: { internet: value }, attrs: {}, classes: new Set(),
    classList: { toggle(c, on) { on ? buttons.find(b => b.dataset.internet === value).classes.add(c) : buttons.find(b => b.dataset.internet === value).classes.delete(c) } },
    setAttribute(k, v) { this.attrs[k] = v },
  }))
  const start = source.indexOf('const INTERNET_SAVING_TEXT')
  const code = source.slice(start, source.indexOf('// ── Status ─'))
  const ctx = {
    document: { getElementById: element, querySelectorAll: () => buttons },
    setDot(dotId, lblId, up, label) { element(dotId).className = 'dot ' + (up ? 'green' : 'red'); element(lblId).textContent = label },
    setTimeout: (fn, ms) => { timers.push({ fn, ms }); return timers.length },
    wsSend: msg => { sent.push(msg); return wsOpen },
  }
  vm.createContext(ctx)
  vm.runInContext(code, ctx)
  return { ctx, element, timers, sent, buttons }
}

test('the switch has Internet and Offline buttons inside Settings', () => {
  assert.match(html, /id="internet-switch"[\s\S]*data-internet="online"[\s\S]*data-internet="offline"/)
  assert.ok(html.indexOf('id="internet-switch"') > html.indexOf('id="left-settings"'))
})

test('the switch shows the saved choice; the header dot is a separate reachability display', () => {
  assert.match(source, /setDot\('dot-net', 'lbl-net', s\.online, 'Internet'\)/)
  const h = harness()
  h.ctx.renderInternetSwitch(false)
  assert.equal(h.buttons[1].classes.has('is-selected'), true)
  assert.equal(h.buttons[1].attrs['aria-checked'], 'true')
  assert.equal(h.buttons[0].attrs['aria-checked'], 'false')
  assert.equal(h.element('dot-net').className, '', 'choosing Offline must not repaint the reachability dot')
  h.ctx.renderInternetSwitch(true)
  assert.equal(h.buttons[0].classes.has('is-selected'), true)
})

test('choosing a mode sends set_internet at once and shows "saving…" until the reply', () => {
  const h = harness()
  h.ctx.setInternet(false)
  assert.equal(JSON.stringify(h.sent), JSON.stringify([{ type: 'set_internet', online: false }]))
  assert.equal(h.element('internet-status').textContent, SAVING)
  h.ctx.applyInternetState({ enabled: false, error: '' })
  assert.equal(h.element('internet-status').textContent, 'บันทึกแล้ว ✓')
  assert.equal(h.buttons[1].classes.has('is-selected'), true)
  assert.equal(h.element('dot-net').className, '')
  h.timers.filter(t => t.ms === 2500).forEach(t => t.fn())
  assert.equal(h.element('internet-status').textContent, '')
})

test('clicking the mode already selected sends nothing', () => {
  const h = harness()
  h.ctx.setInternet(true)
  assert.equal(h.sent.length, 0)
})

test('a status frame is ignored while a change is pending (guard is in applyStatus)', () => {
  assert.match(source, /if \(typeof s\.internet_enabled === 'boolean' && !internetPending\) renderInternetSwitch\(s\.internet_enabled\)/)
})

test('a refusal keeps the previous mode and says why, in red', () => {
  const h = harness()
  h.ctx.setInternet(false)
  h.ctx.applyInternetState({ enabled: true, error: 'agent is busy' })
  assert.equal(h.buttons[0].classes.has('is-selected'), true)
  assert.match(h.element('internet-status').textContent, /agent กำลังทำงาน/)
  assert.ok(h.element('internet-status').classList.contains('err'))
})

test('a dropped WebSocket or a silent server is reported, not left on "saving…"', () => {
  const closed = harness({ wsOpen: false })
  closed.ctx.setInternet(false)
  assert.notEqual(closed.element('internet-status').textContent, SAVING)
  assert.ok(closed.element('internet-status').classList.contains('err'))

  const silent = harness()
  silent.ctx.setInternet(false)
  silent.timers.filter(t => t.ms === 8000).forEach(t => t.fn())
  assert.notEqual(silent.element('internet-status').textContent, SAVING)
  assert.ok(silent.element('internet-status').classList.contains('err'))
})

test('the renderer routes internet_state frames to applyInternetState', () => {
  assert.match(source, /case 'internet_state':\s*applyInternetState\(ev\)/)
})
