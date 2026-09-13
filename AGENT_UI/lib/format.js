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

function byteRateText(value) {
  if (value === null || value === undefined || value === '') return '—'
  let n = Number(value)
  if (!Number.isFinite(n) || n < 0) return '—'
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s']
  let idx = 0
  while (Math.abs(n) >= 1024 && idx < units.length - 1) {
    n /= 1024
    idx += 1
  }
  return `${n.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`
}

function systemTelemetryText(sample) {
  const s = sample || {}
  const toNumber = value => (
    value === null || value === undefined || value === '' ? NaN : Number(value)
  )
  const pct = (value, digits = 0) => {
    const n = toNumber(value)
    return Number.isFinite(n) ? `${n.toFixed(digits)}%` : '—'
  }
  const used = toNumber(s.ram_used_bytes)
  const total = toNumber(s.ram_total_bytes)
  let ram = '—'
  if (Number.isFinite(used) && used >= 0 && Number.isFinite(total) && total > 0) {
    ram = `${(used / (1024 ** 3)).toFixed(1)}/${(total / (1024 ** 3)).toFixed(0)} GB`
  } else if (Number.isFinite(toNumber(s.ram_percent))) {
    ram = pct(s.ram_percent)
  }
  const up = byteRateText(s.network_up_bytes_per_second)
  const down = byteRateText(s.network_down_bytes_per_second)
  const net = up === '—' || down === '—' ? '…' : `↑${up} ↓${down}`
  const tokenRate = toNumber(s.tokens_per_second_5s)
  const tok = Number.isFinite(tokenRate) && tokenRate >= 0 ? `${tokenRate.toFixed(1)} t/s` : '—'
  return `CPU ${pct(s.cpu_percent, 1)} · GPU ${pct(s.gpu_percent)} · RAM ${ram} · NET ${net} · TOK ${tok}`
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    mlxLabel, shortModelName, actTimestamp, selectValueAfterRefresh,
    runtimeStatusText, modelServerStatusText,
    byteRateText, systemTelemetryText,
  }
}
