// Shell contract: what the old Electron preload / main.js tests guarded, checked on the Tauri shell.
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const { ROOT, rust, bridge, rustFn } = require('./shell_sources')

const renderer = fs.readFileSync(path.join(ROOT, 'ui', 'renderer.js'), 'utf8')
const html = fs.readFileSync(path.join(ROOT, 'ui', 'index.html'), 'utf8')
const models = fs.readFileSync(path.join(ROOT, 'ui', 'models.js'), 'utf8')
const lib = fs.readFileSync(path.join(ROOT, 'src-tauri', 'src', 'lib.rs'), 'utf8')

test('every electronAPI method the page calls exists in the bridge', () => {
  const used = new Set([...`${renderer}\n${models}`.matchAll(/electronAPI\.([a-zA-Z]+)/g)].map(m => m[1]))
  const offered = new Set([...bridge.matchAll(/^\s{4}([a-zA-Z]+): /gm)].map(m => m[1]))
  for (const name of used) assert.ok(offered.has(name), `bridge lacks ${name}`)
})

test('every command the bridge invokes is registered, and nothing else is', () => {
  const invoked = new Set([...bridge.matchAll(/invoke\('([a-z_]+)'/g)].map(m => m[1]))
  const handler = lib.slice(lib.indexOf('generate_handler!['), lib.indexOf('])', lib.indexOf('generate_handler![')))
  const registered = new Set([...handler.matchAll(/commands::([a-z_]+)/g)].map(m => m[1]))
  for (const name of invoked) assert.ok(registered.has(name), `${name} is not registered`)
  for (const name of registered) assert.ok(invoked.has(name), `${name} is registered but nothing calls it`)
})

test('the page gets no plugin permissions: every file, dialog and process action is one of the app commands', () => {
  const cap = JSON.parse(fs.readFileSync(path.join(ROOT, 'src-tauri', 'capabilities', 'default.json'), 'utf8'))
  assert.deepEqual(cap.windows, ['main'])
  assert.deepEqual(cap.permissions, ['core:default'])
})

test('the window only ever shows the app; web links go to the browser, nothing else navigates', () => {
  assert.match(lib, /\.on_navigation\(move \|url\| \{\s*if is_app_url\(url\) \{\s*return true;\s*\}/)
  assert.match(lib, /matches!\(url\.scheme\(\), "http" \| "https"\)/)
  assert.match(lib, /fn is_app_url[\s\S]*?"tauri"[\s\S]*?"tauri\.localhost"/)
  assert.match(rust, /pub fn open_url\(app: AppHandle, url: String\) \{\s*if url\.starts_with\("http:\/\/"\) \|\| url\.starts_with\("https:\/\/"\)/)
  assert.match(bridge, /\/\^https\?:\\\/\\\/\/i\.test\(href\)/)
})

test('the agent port is one number in the shell, the page and its CSP; the token travels only in the env and the header', () => {
  assert.match(rust, /pub const AGENT_PORT: u16 = 8765;/)
  assert.match(renderer, /const WS_URL = 'ws:\/\/localhost:8765\/ws/)
  assert.match(html, /connect-src 'self' ipc: http:\/\/ipc\.localhost ws:\/\/localhost:8765/)
  assert.match(rust, /\.env\("AGENT_SERVER_TOKEN", &self\.token\)/)
  assert.match(rust, /\.header\("X-Auth-Token", token\)/)
  assert.doesNotMatch(bridge.replace(/getToken[^\n]*/, ''), /token/i, 'only getToken (the WebSocket handshake) hands the token to the page')
})

test('edit and delete are decided by the agent backend; delete asks first and never leaves the approved scope', () => {
  const edit = rustFn('edit_file')
  const del = rustFn('delete_file')
  assert.ok(edit.indexOf('"/edit-access/authorize"') >= 0 && edit.indexOf('"/edit-access/authorize"') < edit.indexOf('open_path('), 'authorize before opening')
  assert.ok(del.indexOf('blocking_show()') < del.indexOf('"/edit-access/delete"'), 'confirm before asking the agent to delete')
  assert.doesNotMatch(del, /remove_file\(/, 'the shell never unlinks by itself')
})

test('the agent is the only thing this shell supervises; the model server is left to the Python owner', () => {
  assert.doesNotMatch(rust, /mlx_vlm\.server|run_mlx_server_owned/)
  assert.match(rust, /\.arg\("agent_server\.py"\)/)
  assert.match(rust, /libc::kill\(-\(pid as i32\), libc::SIGTERM\)/)
})

test('the new layout keeps Workspace, Active, History and Activity beside the chat and the rest full-page', () => {
  assert.match(renderer, /const FULL_PAGE_PANELS = new Set\(\['settings', 'model', 'tts', 'pdf', 'translate'\]\)/)
  for (const split of ['workspace', 'focus', 'history', 'activity']) assert.doesNotMatch(renderer.match(/FULL_PAGE_PANELS = new Set\(\[[^\]]*\]\)/)[0], new RegExp(`'${split}'`))
  assert.doesNotMatch(html, /btn-telegram|left-telegram/, 'Agent TH has no Telegram bridge')
})

function railHarness(ttsAllowed = () => true) {
  const vm = require('node:vm')
  const start = renderer.indexOf('const FULL_PAGE_PANELS')
  const end = renderer.indexOf('function applyApprovedEditFolders(')
  const els = {}
  const el = id => (els[id] ||= { id, classList: { set: new Set(), add(c) { this.set.add(c) }, remove(c) { this.set.delete(c) }, toggle(c, on) { on ? this.set.add(c) : this.set.delete(c) }, contains(c) { return this.set.has(c) } } })
  const ctx = {
    activePanel: null, modelServerStatusTimer: null, workspaceRoot: '', focusRoot: '', focusDirPath: '', editAccessState: {},
    document: { getElementById: el, querySelectorAll: () => [] },
    wsSend() {}, refreshTtsVoices() {}, restoreTranslateJob() {}, renderFocusFiles() {}, ttsPanelMayOpen: ttsAllowed,
    window: {}, setInterval() { return 1 }, clearInterval() {},
  }
  vm.createContext(ctx)
  vm.runInContext(renderer.slice(start, end).replace(/^let lastAppPanel/m, 'var lastAppPanel') + '\nthis.toggle = togglePanel; this.state = () => activePanel', ctx)
  return { ctx, el }
}

test('the rail: แชต closes any open page (an app page too), แอป reopens the last app, an open app card stays', () => {
  const { ctx, el } = railHarness()
  ctx.toggle('apps')
  assert.equal(ctx.state(), 'pdf', 'the first app is PDF → Markdown (TTS asks the OmniVoice licence question first)')
  ctx.toggle('pdf') // the open app's own card
  assert.equal(ctx.state(), 'pdf')
  ctx.toggle('translate')
  assert.equal(ctx.state(), 'translate')
  ctx.toggle('chat') // live bug in the sibling app: แชต did nothing while an app page was open
  assert.equal(ctx.state(), null)
  assert.ok(!el('main').classList.contains('page-full'))
  ctx.toggle('apps')
  assert.equal(ctx.state(), 'translate', 'แอป reopens the last app used')
  ctx.toggle('settings')
  assert.ok(el('main').classList.contains('page-full'))
  ctx.toggle('chat')
  assert.equal(ctx.state(), null)
  ctx.toggle('workspace')
  assert.ok(!el('main').classList.contains('page-full'), 'Workspace stays beside the chat')
})

test('saying no to the TTS licence prompt opens nothing, and แอป does not retry it', () => {
  let asked = 0
  const { ctx } = railHarness(() => { asked += 1; return false })
  ctx.toggle('tts')
  assert.equal(ctx.state(), null)
  assert.equal(asked, 1)
  ctx.toggle('apps') // lastAppPanel was never moved to tts
  assert.equal(ctx.state(), 'pdf')
  assert.equal(asked, 1)
})

test('the page never calls window.confirm: the Tauri dialog plugin replaces it with an async function (always truthy)', async () => {
  for (const [name, src] of [['renderer.js', renderer], ['models.js', models]]) {
    assert.doesNotMatch(src.replace(/\/\/.*$/gm, ''), /(^|[^A-Za-z.])(window\.)?confirm\(/m, `${name} must ask through askConfirm`)
  }
  assert.match(bridge, /confirm: function \(message\) \{ return invoke\('confirm_dialog'/)
  const ask = renderer.slice(renderer.indexOf('function askConfirm('), renderer.indexOf('// In the Tauri app local files load'))
  assert.ok(ask.length > 50)
  const vm = require('node:vm')
  const run = async api => {
    const ctx = { window: { electronAPI: api }, Promise }
    vm.createContext(ctx)
    vm.runInContext(ask + '\nthis.ask = askConfirm', ctx)
    return ctx.ask('?')
  }
  assert.equal(await run(undefined), false, 'no way to ask means no')
  assert.equal(await run({ confirm: async () => true }), true)
  assert.equal(await run({ confirm: async () => false }), false)
  assert.equal(await run({ confirm: async () => ({}) }), false, 'only an exact true is a yes')
  assert.equal(await run({ confirm: async () => { throw new Error('x') } }), false)
})

test('the low-RAM warning waits for the answer before it re-sends the settings', () => {
  const block = renderer.slice(renderer.indexOf('if (ev.confirmation_required && previousPending)'), renderer.indexOf('// Only an explicit backend terminal state'))
  assert.match(block, /askConfirm\(/)
  assert.ok(block.indexOf('.then(confirmed') < block.indexOf('confirmed_low_ram: true'))
})
