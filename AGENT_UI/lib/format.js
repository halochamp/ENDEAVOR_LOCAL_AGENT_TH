// Pure formatting helpers shared by renderer.js. Loaded as a plain <script> in the
// renderer (functions become globals) and via require() in node --test.

// MLX inference server label — show the real port from the status URL (dev can
// override MLX_BASE_URL to a different port). Falls back to plain "MLX".
function mlxLabel(url) {
  const m = typeof url === 'string' ? url.match(/:(\d+)/) : null
  return m ? 'MLX :' + m[1] : 'MLX'
}

// The local model the agent actually calls (config.MODEL). Header shows the short
// name (after the last "/"); full id is left for the caller to use as a tooltip.
function shortModelName(model) {
  if (!model) return ''
  return model.includes('/') ? model.split('/').pop() : model
}

// HH:MM:SS timestamp for activity-list entries.
function actTimestamp(date) {
  const d = date || new Date()
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
}

function selectValueAfterRefresh(currentValue, authoritativeValue, isFocused, optionValues) {
  const current = String(currentValue ?? '')
  const authoritative = String(authoritativeValue ?? '')
  const valid = new Set((optionValues || []).map(value => String(value)))
  return isFocused && valid.has(current) ? current : authoritative
}

function runtimeStatusText(error, switchState) {
  const problem = String(error ?? '').trim()
  const state = String(switchState ?? 'idle')
  if (problem) return `⚠ ${problem}`
  if (state === 'switching') return '⏳ กำลังปรับ Runtime…'
  if (state === 'ready') return '✓ Runtime พร้อมใช้งาน'
  return ''
}

function modelServerStatusText(server) {
  const s = server || {}
  const action = String(s.action_state || '')
  if (action === 'starting') return 'กำลัง Start…'
  if (action === 'stopping') return 'กำลัง Stop…'
  if (action === 'resetting') return 'กำลัง Reset…'
  if (action === 'recovering') return 'Watchdog กำลังกู้คืน…'
  const state = String(s.state || 'stopped')
  const problem = String(s.error || '').trim()
  if (state === 'shared_ready' && s.healthy) return `Shared พร้อม · ${shortModelName(s.loaded_model || s.selected_model)}`
  if (state === 'ready' && s.healthy) return `พร้อม · ${shortModelName(s.loaded_model || s.selected_model)}`
  if (state === 'shared_offline') return 'Shared MAX offline'
  if (state === 'stopped') return s.desired_state === 'running' ? 'หยุดอยู่ · รอ Watchdog' : 'ปิดอยู่'
  if (state === 'foreign') return 'Port ถูกใช้งานโดย process อื่น'
  if (problem) return `⚠ ${problem}`
  return state
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    mlxLabel, shortModelName, actTimestamp, selectValueAfterRefresh,
    runtimeStatusText, modelServerStatusText,
  }
}
