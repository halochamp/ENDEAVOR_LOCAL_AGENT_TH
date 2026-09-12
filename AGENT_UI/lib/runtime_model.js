// Pure helpers for discovering the model owned by an already-running MLX listener.
// Kept free of Electron APIs so the parsing contract is deterministic and unit-testable.

function listenerPidFromLsof(output) {
  const match = String(output || '').match(/^p(\d+)$/m)
  return match ? Number(match[1]) : 0
}

function modelFromCommand(command) {
  const text = String(command || '')
  const match = text.match(/(?:^|\s)--model(?:=|\s+)(?:"([^"]+)"|'([^']+)'|([^\s]+))/)
  return match ? (match[1] || match[2] || match[3] || '') : ''
}

function classifyServerPresence({ pid = 0, model = '', apiReady = false } = {}) {
  if (!pid && !apiReady) return 'free'
  if (!pid || !model) return 'external_unknown'
  return apiReady ? 'shared_ready' : 'shared_loading'
}

function requiresLargeModelWarning({ model = '', highQualityModel = '', ramBytes = 0, thresholdBytes = 0 } = {}) {
  const ram = Number(ramBytes || 0)
  const threshold = Number(thresholdBytes || 0)
  return String(model || '') === String(highQualityModel || '') && ram > 0 && threshold > 0 && ram < threshold
}

module.exports = {
  listenerPidFromLsof,
  modelFromCommand,
  classifyServerPresence,
  requiresLargeModelWarning,
}
