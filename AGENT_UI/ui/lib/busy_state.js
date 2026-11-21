// busy_state.js — WS connection lifecycle effects (busy / input recovery).
//
// Single-source dual-mode module (same pattern as format.js): loaded as a
// <script> in index.html so renderer.js calls these globals, AND require()'d
// by test/busy_state.test.js. The decision logic lives here; renderer.js is a
// thin adapter that injects the real DOM/state effects via `fx`.
//
// Guards UI-R8-1: a WS close mid-turn must reset isBusy (else the send button
// stays stuck after a reconnect). The effects are injected so a test can spy on
// them — if someone deletes fx.setBusy(false) from the busy branch, the test fails.

// fx (all optional except where noted):
//   setStreamingActive(bool), setBusy(bool), showCancelButton(bool),
//   setRunId(value), discardStreaming(), removeThinking(),
//   disableInput(bool), setPhase(text, spinning), scheduleReconnect(),
//   hideLabelNotice()
function applyClose(state, fx) {
  // Mid-turn drop: tear down the in-flight turn so the UI isn't stuck busy.
  if (state.isBusy) {
    fx.setStreamingActive(false)
    fx.setBusy(false)            // UI-R8-1: the load-bearing reset
    fx.showCancelButton(false)
    fx.setRunId(null)
    fx.discardStreaming()
    fx.removeThinking()
  }
  // Always disable input + show reconnecting — even when idle. Without this the
  // button looks enabled but sendMessage() silently no-ops while disconnected.
  fx.disableInput(true)
  fx.setPhase('ตัดการเชื่อมต่อ — กำลังเชื่อมใหม่…', false)
  fx.scheduleReconnect()
}

function applyOpen(state, fx) {
  fx.hideLabelNotice()
  // Re-enable input only if a turn isn't still running (busy state survives a
  // reconnect; the in-flight turn keeps the input locked until it completes).
  if (!state.isBusy) {
    fx.disableInput(false)
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { applyClose, applyOpen }
}
