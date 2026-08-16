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

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { mlxLabel, shortModelName, actTimestamp }
}
