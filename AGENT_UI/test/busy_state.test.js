const { test } = require('node:test')
const assert = require('node:assert')
const { applyClose, applyOpen } = require('../lib/busy_state')

// Build an fx object of spies. Each spy records its call args so a test can
// assert the connection handler invoked the load-bearing effect (UI-R8-1).
function spyFx() {
  const calls = {}
  const rec = name => (...args) => { (calls[name] ||= []).push(args) }
  const fx = {
    setStreamingActive: rec('setStreamingActive'),
    setBusy: rec('setBusy'),
    showCancelButton: rec('showCancelButton'),
    setRunId: rec('setRunId'),
    discardStreaming: rec('discardStreaming'),
    removeThinking: rec('removeThinking'),
    disableInput: rec('disableInput'),
    setPhase: rec('setPhase'),
    scheduleReconnect: rec('scheduleReconnect'),
    hideLabelNotice: rec('hideLabelNotice'),
  }
  return { fx, calls }
}

// ── close while busy (the UI-R8-1 regression guard) ──────────────────────────
test('close while busy resets isBusy via setBusy(false)', () => {
  const { fx, calls } = spyFx()
  applyClose({ isBusy: true }, fx)
  // The bug: button stuck busy after reconnect. setBusy(false) is the fix.
  assert.deepStrictEqual(calls.setBusy, [[false]])
  assert.ok(calls.discardStreaming, 'discardStreaming must run to drop the partial stream')
  assert.ok(calls.removeThinking, 'removeThinking must clear the thinking indicator')
  assert.deepStrictEqual(calls.showCancelButton, [[false]])
  assert.deepStrictEqual(calls.setRunId, [[null]])
  assert.deepStrictEqual(calls.setStreamingActive, [[false]])
})

test('close while busy also disables input + schedules reconnect', () => {
  const { fx, calls } = spyFx()
  applyClose({ isBusy: true }, fx)
  assert.deepStrictEqual(calls.disableInput, [[true]])
  assert.ok(calls.setPhase, 'setPhase must show the reconnecting notice')
  assert.ok(calls.scheduleReconnect, 'must schedule a reconnect')
})

// ── close while idle ─────────────────────────────────────────────────────────
test('close while idle does NOT touch busy teardown but still locks input', () => {
  const { fx, calls } = spyFx()
  applyClose({ isBusy: false }, fx)
  // No in-flight turn → no teardown.
  assert.strictEqual(calls.setBusy, undefined)
  assert.strictEqual(calls.discardStreaming, undefined)
  assert.strictEqual(calls.removeThinking, undefined)
  // But input must still lock + reconnect must still be scheduled.
  assert.deepStrictEqual(calls.disableInput, [[true]])
  assert.ok(calls.scheduleReconnect)
})

// ── open recovery ──────────────────────────────────────────────────────────────
test('open while idle re-enables input', () => {
  const { fx, calls } = spyFx()
  applyOpen({ isBusy: false }, fx)
  assert.deepStrictEqual(calls.disableInput, [[false]])
  assert.ok(calls.hideLabelNotice)
})

test('open while still busy keeps input disabled (no re-enable)', () => {
  const { fx, calls } = spyFx()
  applyOpen({ isBusy: true }, fx)
  // A turn survived the reconnect → must NOT re-enable input under it.
  assert.strictEqual(calls.disableInput, undefined)
  assert.ok(calls.hideLabelNotice)
})
