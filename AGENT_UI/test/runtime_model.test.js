const test = require('node:test')
const assert = require('node:assert')
const {
  listenerPidFromLsof,
  modelFromCommand,
  classifyServerPresence,
  requiresLargeModelWarning,
} = require('../lib/runtime_model')

test('listenerPidFromLsof extracts listener pid', () => {
  assert.strictEqual(listenerPidFromLsof('p84733\n'), 84733)
  assert.strictEqual(listenerPidFromLsof('p84733\nf15\nPTCP\n'), 84733)
})

test('listenerPidFromLsof returns 0 when no listener pid exists', () => {
  assert.strictEqual(listenerPidFromLsof(''), 0)
  assert.strictEqual(listenerPidFromLsof('f15\nPTCP\n'), 0)
})

test('modelFromCommand parses standard mlx_vlm.server model', () => {
  assert.strictEqual(
    modelFromCommand('/opt/python -m mlx_vlm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --port 8085'),
    'unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit',
  )
})

test('modelFromCommand parses patched launcher and equals syntax', () => {
  assert.strictEqual(
    modelFromCommand('/opt/python scripts/run_vlm_server_patched.py --model=Qwen/Qwen3-14B-MLX-4bit --port 8085'),
    'Qwen/Qwen3-14B-MLX-4bit',
  )
  assert.strictEqual(
    modelFromCommand('/opt/python -m mlx_vlm.server --model=mlx-community/Qwen3.5-9B-4bit --port 8085'),
    'mlx-community/Qwen3.5-9B-4bit',
  )
})

test('modelFromCommand supports quoted model and fails closed without --model', () => {
  assert.strictEqual(modelFromCommand('python server.py --model "model with spaces" --port 8085'), 'model with spaces')
  assert.strictEqual(modelFromCommand('python server.py --port 8085'), '')
})


test('classifyServerPresence starts only when both listener and API are absent', () => {
  assert.strictEqual(classifyServerPresence({ pid: 0, model: '', apiReady: false }), 'free')
  assert.strictEqual(classifyServerPresence({ pid: 84733, model: '', apiReady: false }), 'external_unknown')
  assert.strictEqual(classifyServerPresence({ pid: 0, model: '', apiReady: true }), 'external_unknown')
})

test('classifyServerPresence protects loading and ready shared listeners', () => {
  assert.strictEqual(
    classifyServerPresence({ pid: 84733, model: 'unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit', apiReady: false }),
    'shared_loading',
  )
  assert.strictEqual(
    classifyServerPresence({ pid: 84733, model: 'unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit', apiReady: true }),
    'shared_ready',
  )
})

test('requiresLargeModelWarning warns only for 35B below 24GB', () => {
  const GB = 1024 ** 3
  const high = 'unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit'
  const small = 'Qwen/Qwen3-14B-MLX-4bit'
  const compactVlm = 'mlx-community/Qwen3.5-9B-4bit'
  assert.strictEqual(requiresLargeModelWarning({
    model: high, highQualityModel: high, ramBytes: 16 * GB, thresholdBytes: 24 * GB,
  }), true)
  assert.strictEqual(requiresLargeModelWarning({
    model: high, highQualityModel: high, ramBytes: 24 * GB, thresholdBytes: 24 * GB,
  }), false)
  assert.strictEqual(requiresLargeModelWarning({
    model: small, highQualityModel: high, ramBytes: 16 * GB, thresholdBytes: 24 * GB,
  }), false)
  assert.strictEqual(requiresLargeModelWarning({
    model: compactVlm, highQualityModel: high, ramBytes: 16 * GB, thresholdBytes: 24 * GB,
  }), false)
})
