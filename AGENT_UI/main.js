const { app, BrowserWindow, ipcMain, shell, dialog, nativeTheme } = require('electron')
const path = require('path')
const fs = require('fs')
const { spawn, exec } = require('child_process')
const http = require('http')
const crypto = require('crypto')
const { isInsideWorkspace } = require('./lib/workspace_guard')

// Static auth token (P2 fix) — generated once per launch, shared with the Python
// server via env and with the renderer via IPC. The agent server requires it on
// every WebSocket/REST request, so a drive-by browser site (which can't read it)
// cannot drive the agent or read workspace files.
const AGENT_TOKEN = crypto.randomBytes(32).toString('base64url')

// ENDEAVOR_LOCAL_AGENT_TH fork of AGENT_UI_MAX (2026-08-16): TH's backend lives
// directly at the repo root (this folder's parent), not in a nested
// ENDEAVOR_LOCAL_AGENT_MAX sibling like MAX's multi-project layout.
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
const PROD_MODEL = (_V2_MODEL && _MLX_BASE_URL !== _DEFAULT_MLX_URL)
  ? _V2_MODEL
  : 'unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit'

// Keep mlx_vlm APC defaults aligned with MAX_VLM production. APC_ENABLED=0 or
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

function checkPort(port, path = '/v1/models') {
  return new Promise(resolve => {
    const req = http.get(`http://localhost:${port}${path}`, res => {
      resolve(res.statusCode < 500)
      res.resume()
    })
    req.on('error', () => resolve(false))
    req.setTimeout(2000, () => { req.destroy(); resolve(false) })
  })
}

// Verifies that the production model is actually loaded — not just any server on the port.
function checkMLXReady(port) {
  return new Promise(resolve => {
    const req = http.get(`http://localhost:${port}/v1/models`, res => {
      if (res.statusCode >= 500) { res.resume(); resolve(false); return }
      let body = ''
      res.on('data', d => { body += d })
      res.on('end', () => {
        try {
          const data = JSON.parse(body)
          const ids = (data.data || []).map(m => m.id)
          resolve(ids.includes(PROD_MODEL))
        } catch { resolve(false) }
      })
    })
    req.on('error', () => resolve(false))
    req.setTimeout(3000, () => { req.destroy(); resolve(false) })
  })
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
  mlxProcess = spawn(PYTHON, ['-m', 'mlx_vlm.server', '--model', PROD_MODEL, '--host', '127.0.0.1', '--port', String(MLX_PORT)], {
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
    env: { ...process.env, AGENT_SERVER_TOKEN: AGENT_TOKEN },
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

async function startup() {
  // Check before killing: if the model server is already warm, adopt it instead of
  // force-killing it (which would discard the KV cache and cost 1–3 min reload).
  const alreadyUp = await checkMLXReady(MLX_PORT)
  if (alreadyUp) {
    sendStatus('MLX server already running (model verified) ✓', 'ok')
  } else {
    sendStatus('Clearing stale ports...', 'info')
    await killPort(MLX_PORT)
    await sleep(600)
    sendStatus('Starting mlx_vlm.server...', 'info')
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
