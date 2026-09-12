const { app, BrowserWindow, ipcMain, shell, dialog, nativeTheme } = require('electron')
const path = require('path')
const fs = require('fs')
const { spawn, exec } = require('child_process')
const http = require('http')
const crypto = require('crypto')
const { isInsideWorkspace } = require('./lib/workspace_guard')
const { listenerPidFromLsof, modelFromCommand, classifyServerPresence } = require('./lib/runtime_model')

// Static auth token (P2 fix) — generated once per launch, shared with the Python
// server via env and with the renderer via IPC. The agent server requires it on
// every WebSocket/REST request, so a drive-by browser site (which can't read it)
// cannot drive the agent or read workspace files.
const AGENT_TOKEN = crypto.randomBytes(32).toString('base64url')

// The TH backend lives directly at the repo root (this folder's parent).
const PROJECT_DIR = path.join(__dirname, '..')
const AGENT_DIR = PROJECT_DIR

// Resolve the conda 'mlx' env directory portably — works on any machine/path.
// Priority: explicit env var > conda info --json (any install) > common locations > bare binary.
function _findCondaEnvDir(envName) {
  try {
    const { execSync } = require('child_process')
    const out = execSync('conda info --json', { encoding: 'utf8', timeout: 5000 })
    const info = JSON.parse(out)
    const match = (info.envs || []).find(e => e === envName || e.endsWith(path.sep + envName))
    if (match && fs.existsSync(match)) return match
  } catch {}
  const HOME = process.env.HOME || process.env.USERPROFILE || ''
  for (const base of [
    path.join(HOME, 'opt', 'anaconda3'),
    '/opt/homebrew/anaconda3',
    path.join(HOME, 'anaconda3'),
    path.join(HOME, 'miniconda3'),
    '/opt/anaconda3',
    '/usr/local/anaconda3',
  ]) {
    const envDir = path.join(base, 'envs', envName)
    if (fs.existsSync(path.join(envDir, 'bin', 'python'))) return envDir
  }
  return null
}

const _CONDA_ENV  = process.env.MLX_CONDA_ENV || 'mlx'
const _condaDir   = _findCondaEnvDir(_CONDA_ENV)
const PYTHON = process.env.MLX_PYTHON || (_condaDir ? path.join(_condaDir, 'bin', 'python') : 'python3')
const AGENT_PORT = 8765

// Mirrors config.py's own override rule exactly (README's RAM<48GB guidance
// tells users to set both together) — MODEL only overrides when BOTH
// MLX_BASE_URL and V2_MODEL are set, otherwise always the production model.
// A hardcoded 35B/port-8085 here would silently ignore that documented path
// and try to load a model too big for the RAM this override exists for.
const _DEFAULT_MLX_URL = 'http://localhost:8085/v1'
const _MLX_BASE_URL = process.env.MLX_BASE_URL || _DEFAULT_MLX_URL
const _V2_MODEL = process.env.V2_MODEL || ''
const MLX_PORT = Number(new URL(_MLX_BASE_URL).port) || 8085
const _DEFAULT_MODEL = 'unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit'
const _MODEL_CHOICES = new Set([_DEFAULT_MODEL, 'Qwen/Qwen3-14B-MLX-4bit'])
const _runtimeSettingsOverride = String(process.env.V2_RUNTIME_SETTINGS_PATH || '').trim()
const _RUNTIME_SETTINGS_PATH = _runtimeSettingsOverride
  ? (path.isAbsolute(_runtimeSettingsOverride)
      ? _runtimeSettingsOverride
      : path.join(PROJECT_DIR, _runtimeSettingsOverride))
  : path.join(PROJECT_DIR, 'workspace', 'runtime_settings.json')

function _persistedModel() {
  try {
    const data = JSON.parse(fs.readFileSync(_RUNTIME_SETTINGS_PATH, 'utf8'))
    return data && data.owner === 'agent_th' && _MODEL_CHOICES.has(data.model) ? data.model : ''
  } catch { return '' }
}

let activeModel = (_V2_MODEL && _MLX_BASE_URL !== _DEFAULT_MLX_URL)
  ? _V2_MODEL
  : (_persistedModel() || _DEFAULT_MODEL)
let _mlxOwnedByThisApp = false
let _sharedMlxExternal = false

// Keep mlx_vlm APC defaults aligned with the documented production profile. APC_ENABLED=0 or
// other explicit environment overrides still win. Exact entries=2 is the
// smallest useful capacity for the current guarded exact-prefix strategy:
// one reusable guarded checkpoint + one full-prompt snapshot.
const MLX_APC_ENV = {
  APC_ENABLED: process.env.APC_ENABLED || '1',
  APC_EXACT_CACHE_ENTRIES: process.env.APC_EXACT_CACHE_ENTRIES || '2',
  APC_EXACT_PREFIX_GUARD_TOKENS: process.env.APC_EXACT_PREFIX_GUARD_TOKENS || '64',
}

let mainWindow = null
let mlxProcess = null
let agentServerProcess = null

let _crashCount = 0
let _firstCrashAt = 0
const _MAX_CRASHES = 20
const _CRASH_WINDOW_MS = 300_000

let _mlxRestarting = false      // guard: only one MLX restart at a time
let _mlxMonitorInterval = null  // proactive MLX health check interval

// ── Utilities ──────────────────────────────────────────────────────────────────

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function killPort(port) {
  return new Promise(resolve => exec(`lsof -ti:${port} | xargs kill -9 2>/dev/null`, () => resolve()))
}

// A previous launch that crashed or was force-quit can leave its main Electron
// process running. That stale instance still manages agent_server.py/mlx_vlm.server on
// the same ports, racing the new instance into a crash-restart loop (each kills
// the other's agent_server).
//
// `node_modules/electron/dist` is a symlink to a shared Electron install (see
// ensure_electron_symlink.js), so every ENDEAVOR Electron app launches the
// identical resolved binary — matching on that path alone would pgrep-match
// *and kill* sibling apps, not just stale copies of this one. The Electron
// main process's argv is also uninformative here ("Electron .", no --app-path
// — only helper processes carry that flag). cwd is the one signal that
// distinguishes "another instance of this same app" from "a different
// ENDEAVOR app" or an unrelated process, since `npm start` always launches
// Electron from this directory.
function _processCwd(pid) {
  return new Promise(resolve => {
    exec(`lsof -a -p ${pid} -d cwd -Fn`, (_err, stdout) => {
      const line = stdout.split('\n').find(l => l.startsWith('n'))
      resolve(line ? line.slice(1) : '')
    })
  })
}

function killOtherAppInstances() {
  const electronBin = fs.realpathSync(
    path.join(__dirname, 'node_modules', 'electron', 'dist', 'Electron.app', 'Contents', 'MacOS', 'Electron'),
  )
  const ownDir = fs.realpathSync(__dirname)
  return new Promise(resolve => {
    exec(`pgrep -f ${JSON.stringify(electronBin)}`, async (_err, stdout) => {
      const pids = stdout.split('\n').map(s => s.trim()).filter(Boolean).map(Number)
        .filter(pid => pid !== process.pid)
      if (pids.length === 0) { resolve(); return }
      const cwds = await Promise.all(pids.map(async pid => [pid, await _processCwd(pid)]))
      const samePidsOnly = cwds
        .filter(([, cwd]) => { try { return cwd && fs.realpathSync(cwd) === ownDir } catch { return false } })
        .map(([pid]) => pid)
      if (samePidsOnly.length === 0) { resolve(); return }
      console.log('[startup] found stale app instance(s), killing:', samePidsOnly.join(', '))
      for (const pid of samePidsOnly) {
        try { process.kill(pid, 'SIGKILL') } catch {}
      }
      resolve()
    })
  })
}

// Poll until nothing holds the port (or timeout). Much more reliable than a
// fixed sleep after kill -9, which races the OS TCP TIME_WAIT / process teardown.
function waitPortFree(port, timeoutMs = 5000) {
  return new Promise(resolve => {
    const start = Date.now()
    function check() {
      exec(`lsof -ti:${port} 2>/dev/null`, (_err, stdout) => {
        if (!stdout.trim()) { resolve(true); return }
        if (Date.now() - start >= timeoutMs) { resolve(false); return }
        setTimeout(check, 150)
      })
    }
    check()
  })
}

function getMLXModels(port) {
  return new Promise(resolve => {
    const req = http.get(`http://localhost:${port}/v1/models`, res => {
      if (res.statusCode >= 500) { res.resume(); resolve([]); return }
      let body = ''
      res.on('data', d => { body += d })
      res.on('end', () => {
        try {
          const data = JSON.parse(body)
          resolve((data.data || []).map(m => String(m.id || '')).filter(Boolean))
        } catch { resolve([]) }
      })
    })
    req.on('error', () => resolve([]))
    req.setTimeout(3000, () => { req.destroy(); resolve([]) })
  })
}

function getListenerProcess(port) {
  return new Promise(resolve => {
    exec(`lsof -nP -iTCP:${port} -sTCP:LISTEN -Fp`, (_err, stdout) => {
      const pid = listenerPidFromLsof(stdout)
      if (!pid) { resolve({ pid: 0, command: '' }); return }
      exec(`ps -p ${pid} -o command=`, (_psErr, command) => {
        resolve({ pid, command: String(command || '').trim() })
      })
    })
  })
}

async function getMLXServerInfo(port) {
  // Listener ownership comes first. /v1/models is only an API-health/advertisement
  // signal and must never decide which model is active: MAX may advertise many
  // supported models while one process owns exactly one --model.
  const { pid, command } = await getListenerProcess(port)
  const models = await getMLXModels(port)
  if (!pid && !models.length) return null
  return {
    models,
    pid,
    command,
    model: modelFromCommand(command),
    api_ready: models.length > 0,
  }
}

// Verifies that the model Agent TH currently intends to call is both process-owned
// and reachable through the API. The advertised model list is health-only metadata;
// it is deliberately not consulted for active-model identity.
async function checkMLXReady(port) {
  const info = await getMLXServerInfo(port)
  return !!(info && info.pid && info.model === activeModel && info.api_ready)
}

function sendStatus(msg, phase = 'info') {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('startup-status', { msg, phase })
  }
  console.log(`[${phase}] ${msg}`)
}

// ── Server management ──────────────────────────────────────────────────────────

// The public app owns one mlx_vlm.server for both text and direct-vision
// requests. Spawn it directly: PYTHON already resolves to the conda env's
// absolute python binary (_findCondaEnvDir above), so no `conda run` or PATH
// shim is needed.
function startMlxServer() {
  _mlxOwnedByThisApp = true
  _sharedMlxExternal = false
  mlxProcess = spawn(PYTHON, ['-m', 'mlx_vlm.server', '--model', activeModel, '--host', '127.0.0.1', '--port', String(MLX_PORT)], {
    cwd: PROJECT_DIR,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, ...MLX_APC_ENV },
  })
  mlxProcess.stdout.on('data', d => console.log('[mlx]', d.toString().trim()))
  mlxProcess.stderr.on('data', d => {
    const msg = d.toString().trim()
    if (msg) console.log('[mlx]', msg)
  })
  mlxProcess.on('exit', code => console.log('[mlx] exited', code))
}

function startAgentServer() {
  agentServerProcess = spawn(PYTHON, ['agent_server.py'], {
    cwd: AGENT_DIR,
    // detached: own process group — insulates the agent from Chromium's
    // group-wide SIGKILL that fires when the network service utility crashes.
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,
    env: {
      ...process.env,
      AGENT_SERVER_TOKEN: AGENT_TOKEN,
      // Special shared-server mode only: when :8085 pre-existed before TH
      // launched, force the agent client to use the model that server exposes.
      TH_SHARED_MLX_MODEL: _sharedMlxExternal ? activeModel : '',
    },
  })
  // With detached:true Node.js would keep the event loop alive for this child.
  // Unreffing lets the Electron process exit normally even if the agent hangs.
  agentServerProcess.unref()
  agentServerProcess.stdout.on('data', d => console.log('[agent]', d.toString().trim()))
  agentServerProcess.stderr.on('data', d => {
    const msg = d.toString().trim()
    if (msg && !msg.includes('INFO:')) console.log('[agent]', msg)
  })
  agentServerProcess.on('exit', (code, signal) => {
    console.log('[agent] exited', code, signal ? `(signal: ${signal})` : '')

    const now = Date.now()
    if (now - _firstCrashAt > _CRASH_WINDOW_MS) {
      _crashCount = 0
      _firstCrashAt = now
    }
    _crashCount++

    if (_crashCount > _MAX_CRASHES) {
      sendStatus(`Agent crashed ${_crashCount} times — giving up. Restart the app to retry.`, 'error')
      return
    }

    const delay = Math.min(3000 * Math.pow(2, _crashCount - 1), 30000)
    sendStatus(`Agent stopped (crash ${_crashCount}/${_MAX_CRASHES}) — retry in ${delay / 1000}s…`, 'warn')
    setTimeout(async () => {
      if (!app.isReady()) return

      const mlxOk = await checkMLXReady(MLX_PORT)
      if (!mlxOk) {
        if (!_mlxOwnedByThisApp) {
          sendStatus('Shared MLX server is offline — leaving external :8085 untouched', 'warn')
          return
        }
        const restored = await restartMLXNow('agent crash')
        if (!restored) return
        _crashCount = 0
      }

      // The dominant crash is the agent port still being held (Errno 48). The
      // retry is pointless unless we free it first, otherwise we just loop into
      // the same bind failure until _MAX_CRASHES gives up.
      await killPort(AGENT_PORT)
      await waitPortFree(AGENT_PORT)
      startAgentServer()
    }, delay)
  })
}

async function waitForMLX(timeoutMs = 180000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    if (await checkMLXReady(MLX_PORT)) return true
    await sleep(3000)
  }
  return false
}

// ── MLX restart (shared by agent crash handler + proactive monitor) ────────────

async function restartMLXNow(reason) {
  if (!_mlxOwnedByThisApp) {
    sendStatus(`Shared MLX server unavailable (${reason}) — Agent TH will not restart an external server`, 'warn')
    return false
  }
  if (_mlxRestarting) return false
  _mlxRestarting = true
  try {
    sendStatus(`MLX server offline (${reason}) — restarting...`, 'warn')
    await killPort(MLX_PORT)
    await sleep(600)
    if (mlxProcess) {
      mlxProcess.removeAllListeners('exit')
      mlxProcess.kill('SIGKILL')
      mlxProcess = null
    }
    startMlxServer()
    sendStatus('Waiting for MLX to restart (up to 3 min)...', 'info')
    const ready = await waitForMLX(180_000)
    if (ready) {
      sendStatus('MLX server restored ✓', 'ok')
      return true
    }
    sendStatus('MLX server failed to restart', 'error')
    return false
  } finally {
    _mlxRestarting = false
  }
}

// ── Proactive MLX monitor ──────────────────────────────────────────────────────

function startMLXMonitor() {
  if (!_mlxOwnedByThisApp) return
  if (_mlxMonitorInterval) return
  _mlxMonitorInterval = setInterval(async () => {
    if (_mlxRestarting) return              // restart already in progress
    if (await checkMLXReady(MLX_PORT)) return  // healthy
    // Confirm: wait 15s then check again (avoids reacting to transient blips)
    await sleep(15_000)
    if (await checkMLXReady(MLX_PORT)) return  // recovered on its own
    await restartMLXNow('proactive monitor')
  }, 30_000)
  console.log('[mlx-monitor] proactive check every 30s')
}

// ── Startup sequence ───────────────────────────────────────────────────────────

async function applyRuntimeModel(model) {
  const requested = String(model || '').trim()
  if (!_MODEL_CHOICES.has(requested)) {
    return { ok: false, error: `unsupported model: ${requested}` }
  }

  if (_sharedMlxExternal) {
    const info = await getMLXServerInfo(MLX_PORT)
    const sharedModel = info && info.model ? info.model : ''
    if (!sharedModel) {
      return {
        ok: false,
        shared: true,
        model: activeModel,
        error: `cannot determine active model on shared :${MLX_PORT}; external server left untouched`,
      }
    }
    activeModel = sharedModel
    if (requested !== sharedModel) {
      return {
        ok: false,
        shared: true,
        model: activeModel,
        error: `shared :${MLX_PORT} is running ${sharedModel}; Agent TH will not switch an external server`,
      }
    }
    return { ok: true, shared: true, model: activeModel }
  }

  if (!_mlxOwnedByThisApp) {
    return { ok: false, error: 'MLX server is not ready yet' }
  }
  if (requested === activeModel && await checkMLXReady(MLX_PORT)) {
    return { ok: true, shared: false, model: activeModel }
  }

  const previous = activeModel
  activeModel = requested
  const switched = await restartMLXNow('model switch')
  if (switched) return { ok: true, shared: false, model: activeModel }

  activeModel = previous
  const restored = await restartMLXNow('model rollback')
  return {
    ok: false,
    shared: false,
    model: activeModel,
    error: restored
      ? `model switch failed; restored ${previous}`
      : `model switch failed and rollback to ${previous} also failed`,
  }
}

async function startup() {
  // Special shared-server branch: if a healthy OpenAI-compatible MLX server
  // already exists, adopt its actual model and never kill/restart that process.
  // Normal public behavior remains unchanged when the port is free: TH starts
  // and owns its own mlx_vlm.server.
  const existing = await getMLXServerInfo(MLX_PORT)
  const presence = classifyServerPresence({
    pid: existing && existing.pid,
    model: existing && existing.model,
    apiReady: !!(existing && existing.api_ready),
  })
  if (presence !== 'free') {
    _sharedMlxExternal = true
    _mlxOwnedByThisApp = false
    if (presence === 'external_unknown') {
      const detail = existing && existing.pid
        ? `listener PID ${existing.pid} exists but --model cannot be determined`
        : 'an API responds but no local listener process can be identified'
      sendStatus(`Found external MLX state on :${MLX_PORT} (${detail}) — leaving it untouched`, 'error')
      return
    }
    activeModel = existing.model
    if (presence === 'shared_loading') {
      sendStatus(`Shared MLX :${MLX_PORT} · ${activeModel} is still loading — waiting without taking ownership...`, 'info')
      const ready = await waitForMLX()
      if (!ready) {
        sendStatus(`Shared MLX :${MLX_PORT} did not become ready — leaving external process untouched`, 'error')
        return
      }
    }
    sendStatus(`Using shared MLX :${MLX_PORT} · ${activeModel} ✓`, 'ok')
  } else {
    sendStatus('Clearing stale ports...', 'info')
    await killPort(MLX_PORT)
    await sleep(600)
    sendStatus(`Starting mlx_vlm.server · ${activeModel}...`, 'info')
    startMlxServer()
    sendStatus('Waiting for MLX server (this may take 1-3 min)...', 'info')
    const ready = await waitForMLX()
    if (ready) {
      sendStatus('MLX server ready ✓', 'ok')
    } else {
      sendStatus('MLX server timeout — continuing anyway', 'warn')
    }
  }

  sendStatus('Starting agent server...', 'info')
  // Self-heal "[Errno 48] address already in use": a stale agent server from a
  // previous launch may still hold port 8765. We can't adopt it — each launch
  // mints a fresh AGENT_TOKEN, so the old server would reject the new renderer's
  // auth. Kill whatever holds the port, then poll until the OS releases it.
  await killPort(AGENT_PORT)
  await waitPortFree(AGENT_PORT)
  startAgentServer()
  await sleep(2000)
  startMLXMonitor()
  sendStatus('Ready — connecting...', 'ok')
  mainWindow.webContents.send('startup-done')
}

// ── Window ─────────────────────────────────────────────────────────────────────

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1300,
    height: 840,
    minWidth: 900,
    minHeight: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#1e1e1e',
    show: false,
  })

  mainWindow.loadFile('index.html')
  mainWindow.once('ready-to-show', () => {
    mainWindow.show()
    startup()
  })

  // Open external links in browser, not Electron
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
}

// ── App lifecycle ──────────────────────────────────────────────────────────────

// Run the network service in-process (same process as the browser) so a network
// utility crash does not appear as a separate-process crash and does not cascade
// into dropping the renderer's WebSocket connections. This is the primary fix
// for the recurring network_service_instance_impl crash on macOS.
app.commandLine.appendSwitch('--enable-features', 'NetworkServiceInProcess')

// The renderer is dark-themed via its own CSS regardless of OS, but native chrome
// (showMessageBox dialogs) follows the system appearance — on a light-mode Mac the
// exit/delete confirms render white-gray, off-theme. Force dark so native dialogs
// match the app.
app.whenReady().then(async () => {
  await killOtherAppInstances()
  nativeTheme.themeSource = 'dark'
  createWindow()
})

app.on('before-quit', () => {
  if (_mlxMonitorInterval) { clearInterval(_mlxMonitorInterval); _mlxMonitorInterval = null }
  if (agentServerProcess) {
    agentServerProcess.removeAllListeners('exit')
    // detached process → kill its own process group (negative pid) so child
    // threads spawned by uvicorn are also terminated.
    try { process.kill(-agentServerProcess.pid, 'SIGTERM') } catch {}
  }
  if (mlxProcess) mlxProcess.kill()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})

ipcMain.on('open-workspace', () => {
  const ws = path.join(AGENT_DIR, 'workspace')
  shell.openPath(ws)
})

ipcMain.handle('get-token', () => AGENT_TOKEN)
ipcMain.handle('apply-runtime-model', async (_e, model) => applyRuntimeModel(model))

// Compact mode: renderer toggles its CSS, but only the main process can resize
// the BrowserWindow. Shrink to a floating mini-chat pinned above other windows;
// restore the exact pre-compact bounds when toggled back.
let _preCompactBounds = null
ipcMain.on('set-compact', (_e, compact) => {
  if (!mainWindow || mainWindow.isDestroyed()) return
  if (compact) {
    _preCompactBounds = mainWindow.getBounds()
    mainWindow.setMinimumSize(360, 420)
    mainWindow.setSize(420, 560)
    mainWindow.setAlwaysOnTop(true, 'floating')
  } else {
    mainWindow.setAlwaysOnTop(false)
    mainWindow.setMinimumSize(900, 600)
    if (_preCompactBounds) mainWindow.setBounds(_preCompactBounds)
    else mainWindow.setSize(1300, 840)
  }
})

// Top-level /exit from the renderer → confirm in a native dialog, then quit.
// before-quit (above) tears down the agent server + mlx_vlm.server processes.
let _exitConfirming = false
ipcMain.on('request-exit', async () => {
  if (_exitConfirming) return
  _exitConfirming = true
  try {
    const { response } = await dialog.showMessageBox(mainWindow, {
      type: 'question',
      buttons: ['ยกเลิก', 'ออกจากโปรแกรม'],
      defaultId: 0,
      cancelId: 0,
      message: 'ออกจาก ENDEAVOR Agent?',
      detail: 'การทำงานที่ค้างอยู่จะถูกหยุด และปิดเซิร์ฟเวอร์ทั้งหมด',
    })
    if (response === 1) app.quit()
  } finally {
    _exitConfirming = false
  }
})

// Edit → open the file in the OS default app (shell.openPath). Returns '' on
// success or an error string the renderer can surface.
ipcMain.handle('edit-file', async (_e, filePath) => {
  if (typeof filePath !== 'string' || !isInsideWorkspace(filePath, path.join(AGENT_DIR, 'workspace'))) return 'outside workspace'
  try {
    return (await shell.openPath(filePath)) || ''
  } catch (e) {
    return String(e && e.message || e)
  }
})

// Delete → confirm in a native dialog (destructive), then unlink. Only files, never
// directories. Returns {deleted} / {cancelled} / {error} for the renderer.
ipcMain.handle('delete-file', async (_e, filePath) => {
  if (typeof filePath !== 'string' || !isInsideWorkspace(filePath, path.join(AGENT_DIR, 'workspace'))) {
    return { deleted: false, error: 'outside workspace' }
  }
  try {
    if (fs.statSync(filePath).isDirectory()) return { deleted: false, error: 'is a directory' }
  } catch (e) {
    return { deleted: false, error: String(e && e.message || e) }
  }
  const name = path.basename(filePath)
  const { response } = await dialog.showMessageBox(mainWindow, {
    type: 'warning',
    buttons: ['ยกเลิก', 'ลบ'],
    defaultId: 0,
    cancelId: 0,
    message: `ลบไฟล์ "${name}"?`,
    detail: 'การลบนี้ย้อนกลับไม่ได้',
  })
  if (response !== 1) return { deleted: false, cancelled: true }
  try {
    await fs.promises.unlink(filePath)
    return { deleted: true }
  } catch (e) {
    return { deleted: false, error: String(e && e.message || e) }
  }
})

ipcMain.handle('show-open-dialog', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
  })
  return canceled ? null : (filePaths[0] || null)
})
