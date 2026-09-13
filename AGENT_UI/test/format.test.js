const test = require('node:test')
const assert = require('node:assert')
const {
  mlxLabel, shortModelName, actTimestamp, selectValueAfterRefresh,
  runtimeStatusText, modelServerStatusText, byteRateText, systemTelemetryText,
} = require('../lib/format')

test('mlxLabel extracts port from URL', () => {
  assert.strictEqual(mlxLabel('http://localhost:8085/v1'), 'MLX :8085')
  assert.strictEqual(mlxLabel('http://localhost:8888'), 'MLX :8888')
})

test('mlxLabel falls back to plain "MLX" without a port', () => {
  assert.strictEqual(mlxLabel('http://localhost'), 'MLX')
  assert.strictEqual(mlxLabel(undefined), 'MLX')
  assert.strictEqual(mlxLabel(null), 'MLX')
})

test('shortModelName keeps the segment after the last "/"', () => {
  assert.strictEqual(shortModelName('unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit'), 'Qwen3.6-35B-A3B-UD-MLX-4bit')
})

test('shortModelName returns input unchanged when there is no "/"', () => {
  assert.strictEqual(shortModelName('llama3.1:8b'), 'llama3.1:8b')
})

test('shortModelName returns empty string for falsy input', () => {
  assert.strictEqual(shortModelName(''), '')
  assert.strictEqual(shortModelName(null), '')
  assert.strictEqual(shortModelName(undefined), '')
})

test('actTimestamp formats HH:MM:SS with zero-padding', () => {
  const d = new Date(2026, 0, 1, 9, 5, 3)
  assert.strictEqual(actTimestamp(d), '09:05:03')
})

test('actTimestamp defaults to "now" when no date given', () => {
  const before = Date.now()
  const ts = actTimestamp()
  assert.match(ts, /^\d{2}:\d{2}:\d{2}$/)
  assert.ok(Date.now() - before < 1000)
})

test('selectValueAfterRefresh preserves a focused valid user choice', () => {
  assert.strictEqual(
    selectValueAfterRefresh('512', '256', true, ['256', '512', '1024']),
    '512',
  )
})

test('selectValueAfterRefresh uses authoritative value when selector is not focused', () => {
  assert.strictEqual(
    selectValueAfterRefresh('512', '256', false, ['256', '512', '1024']),
    '256',
  )
})

test('selectValueAfterRefresh rejects a focused value no longer present in options', () => {
  assert.strictEqual(
    selectValueAfterRefresh('999', '512', true, ['256', '512', '1024']),
    '512',
  )
})

test('runtimeStatusText exposes runtime switch lifecycle', () => {
  assert.strictEqual(runtimeStatusText('', 'switching'), '⏳ กำลังปรับ Runtime…')
  assert.strictEqual(runtimeStatusText('', 'ready'), '✓ Runtime พร้อมใช้งาน')
  assert.strictEqual(runtimeStatusText('', 'idle'), '')
  assert.strictEqual(runtimeStatusText('boom', 'error'), '⚠ boom')
})

test('modelServerStatusText distinguishes local and external server states', () => {
  assert.strictEqual(modelServerStatusText({
    state: 'ready', healthy: true, loaded_model: 'Qwen/Qwen3-14B-MLX-4bit',
  }), 'พร้อม · Qwen3-14B-MLX-4bit')
  assert.strictEqual(modelServerStatusText({
    state: 'shared_ready', healthy: true, loaded_model: 'unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit',
  }), 'Shared พร้อม · Qwen3.6-35B-A3B-UD-MLX-4bit')
  assert.strictEqual(modelServerStatusText({ state: 'shared_offline' }), 'External server offline')
})

test('system telemetry formats portable CPU GPU RAM TOK and network on one line', () => {
  const line = systemTelemetryText({
    cpu_percent: 12.34,
    gpu_percent: 75,
    ram_used_bytes: 24 * 1024 ** 3,
    ram_total_bytes: 48 * 1024 ** 3,
    network_up_bytes_per_second: 1536,
    network_down_bytes_per_second: 2 * 1024 ** 2,
    tokens_per_second_5s: 42.64,
  })
  assert.strictEqual(
    line,
    'CPU 12.3% · GPU 75% · RAM 24.0/48 GB · TOK 42.6 t/s · NET ↑1.5 KB/s ↓2.0 MB/s',
  )
  assert.strictEqual(byteRateText(null), '—')
  assert.strictEqual(systemTelemetryText({}), 'CPU — · GPU — · RAM — · TOK — · NET …')
})
