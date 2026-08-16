const test = require('node:test')
const assert = require('node:assert')
const { mlxLabel, shortModelName, actTimestamp } = require('../lib/format')

test('mlxLabel extracts port from URL', () => {
  assert.strictEqual(mlxLabel('http://localhost:8080/v1'), 'MLX :8080')
  assert.strictEqual(mlxLabel('http://localhost:8085'), 'MLX :8085')
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
