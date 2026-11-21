// The new shell's sources for the contract tests that used to read Electron's main.js / preload.js.
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.join(__dirname, '..')
const SRC = path.join(ROOT, 'src-tauri', 'src')
const rust = fs.readdirSync(SRC).filter(f => f.endsWith('.rs')).sort().map(f => fs.readFileSync(path.join(SRC, f), 'utf8')).join('\n')
const bridge = fs.readFileSync(path.join(ROOT, 'ui', 'bridge.js'), 'utf8')
/** The body of one Rust function: from its `fn name(` to the first line that is only `}` (rustfmt closes items there). */
function rustFn(name) {
  const start = rust.indexOf(`fn ${name}(`)
  if (start < 0) return ''
  const end = rust.indexOf('\n}\n', start)
  return rust.slice(start, end < 0 ? undefined : end + 2)
}
module.exports = { ROOT, rust, bridge, rustFn }
