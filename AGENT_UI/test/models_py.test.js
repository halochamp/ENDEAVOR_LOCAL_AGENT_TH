// tools/models.py (the Model page backend) against an empty Hugging Face cache and an unreachable endpoint.
// Offline, one `list` once took over 5 minutes (huggingface_hub retries) while the page polls every 3 s, so the page froze
// and python processes piled up on any Mac that is offline and missing a model.
const assert = require('node:assert/strict')
const { spawnSync } = require('node:child_process')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const ROOT = path.join(__dirname, '..')
const script = path.join(ROOT, 'tools', 'models.py')

// The app runs models.py with the agent's `mlx` python (it has huggingface_hub). A bare python3 without it fails fast and
// would hide the slow offline path, so this test needs that python (set MLX_PYTHON, or run it from the activated mlx env).
const PY = process.env.MLX_PYTHON || 'python3'
const hasHub = spawnSync(PY, ['-c', 'import huggingface_hub'], { encoding: 'utf8' }).status === 0

function list(env, ...extra) {
  const t0 = Date.now()
  const r = spawnSync(PY, [script, 'list', ...extra], { env: { ...process.env, ...env }, encoding: 'utf8', timeout: 60_000 })
  return { ms: Date.now() - t0, out: r.stdout ? JSON.parse(r.stdout.trim().split('\n').pop()) : null, status: r.status, signal: r.signal }
}

test('offline with nothing downloaded: a plain list never touches the network; --check answers within seconds', { skip: !hasHub && 'needs the mlx python (huggingface_hub): set MLX_PYTHON' }, () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'agentth-models-'))
  const env = { HUGGINGFACE_HUB_CACHE: path.join(tmp, 'hf'), HF_ENDPOINT: 'http://10.255.255.1', AGENT_TH_UI_STATE: path.join(tmp, 'state') }
  fs.mkdirSync(env.HUGGINGFACE_HUB_CACHE)
  const plain = list(env)
  assert.equal(plain.signal, null, 'not killed by the 60 s guard')
  assert.ok(plain.out && plain.out.ok, 'answers')
  assert.equal(plain.out.models.length, 5, 'the five native-AR models')
  assert.ok(plain.out.models.every(m => m.state === 'missing'), 'nothing is downloaded')
  assert.ok(plain.out.models.every(m => m.variants.length === 1 && m.variants[0].acceleration === 'off'), 'native AR only: no accelerator variant')
  assert.ok(plain.ms < 3_000, `plain list took ${plain.ms} ms (the network is asked only on open / Refresh)`)
  const checked = list(env, '--check')
  assert.ok(checked.out && checked.out.ok)
  assert.ok(checked.ms < 12_000, `list --check took ${checked.ms} ms (one 5 s attempt, then the rest are skipped)`)
  fs.rmSync(tmp, { recursive: true, force: true })
})

test('the Model page asks once on open and on Refresh; it polls only while something downloads', () => {
  const src = fs.readFileSync(path.join(ROOT, 'ui', 'models.js'), 'utf8')
  assert.match(src, /if \(inFlight\) return/)
  assert.match(src, /open\(\) \{\s*refresh\(true\)/)
  assert.match(src, /getElementById\('model-refresh'\)/)
  assert.match(src, /const downloading = data\.models\.some\(m => m\.state === 'downloading' \|\| m\.state === 'queued'\)/)
})
