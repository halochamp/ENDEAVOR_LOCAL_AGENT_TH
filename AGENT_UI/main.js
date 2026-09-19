const { app, BrowserWindow, ipcMain, shell, dialog, nativeTheme } = require('electron')
const path = require('path')
const fs = require('fs')
const { spawn, exec } = require('child_process')
const crypto = require('crypto')
const http = require('http')

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

// Model-server ownership lives in Python model_runtime.py. Electron never owns,
// adopts, kills, or switches the MLX process; it only hosts the UI/backend.
let mainWindow = null
let agentServerProcess = null

let _crashCount = 0
let _firstCrashAt = 0
const _MAX_CRASHES = 20
const _CRASH_WINDOW_MS = 300_000

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
      // model_server.py owns model recovery independently. Electron only keeps
      // the authenticated Agent/UI backend alive.
      await killPort(AGENT_PORT)
      await waitPortFree(AGENT_PORT)
      startAgentServer()
    }, delay)
  })
}

async function startup() {
  // Agent TH's Python backend owns model-server lifecycle, watchdog, port and
  // the special read-only MAX sharing rule. Electron is only the presentation
  // host plus authenticated agent_server.py lifecycle.
  sendStatus('Starting Agent TH runtime...', 'info')
  await killPort(AGENT_PORT)
  await waitPortFree(AGENT_PORT)
  startAgentServer()
  await sleep(2000)
  sendStatus('Ready — connecting...', 'ok')
  mainWindow.webContents.send('startup-done')
}

// ── Window ─────────────────────────────────────────────────────────────────────

const DEFAULT_WINDOW_WIDTH = 900
const DEFAULT_WINDOW_HEIGHT = 600

function createWindow() {
  mainWindow = new BrowserWindow({
    width: DEFAULT_WINDOW_WIDTH,
    height: DEFAULT_WINDOW_HEIGHT,
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
  if (agentServerProcess) {
    agentServerProcess.removeAllListeners('exit')
    // detached process → kill its own process group (negative pid) so child
    // threads spawned by uvicorn are also terminated.
    try { process.kill(-agentServerProcess.pid, 'SIGTERM') } catch {}
  }
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

function _agentJsonRequest({ method = 'GET', route, headers = {}, body = null, filePath = '' }) {
  return new Promise(resolve => {
    const req = http.request({
      hostname: '127.0.0.1', port: AGENT_PORT, path: route, method,
      headers: { 'X-Auth-Token': AGENT_TOKEN, ...headers },
    }, res => {
      const chunks = []
      let total = 0
      res.on('data', chunk => {
        total += chunk.length
        if (total > 1024 * 1024) {
          req.destroy(new Error('agent response too large'))
          return
        }
        chunks.push(chunk)
      })
      res.on('end', () => {
        try {
          const parsed = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')
          parsed.http_status = res.statusCode || 0
          resolve(parsed)
        } catch {
          resolve({ ok: false, error: 'invalid agent response', http_status: res.statusCode || 0 })
        }
      })
    })
    req.on('error', err => resolve({ ok: false, error: String(err && err.message || err) }))
    req.setTimeout(300_000, () => req.destroy(new Error('PDF upload timed out')))
    if (filePath) {
      const stream = fs.createReadStream(filePath)
      stream.on('error', err => req.destroy(err))
      stream.pipe(req)
    } else {
      req.end(body == null ? undefined : body)
    }
  })
}

ipcMain.handle('pdf-to-text-start', async (_e, rewriteThai) => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    title: 'เลือก PDF เพื่อแปลงเป็นข้อความ',
    properties: ['openFile'],
    filters: [{ name: 'PDF', extensions: ['pdf'] }],
  })
  if (canceled || !filePaths.length) return { ok: false, cancelled: true }
  const filePath = filePaths[0]
  try {
    const info = await fs.promises.stat(filePath)
    if (!info.isFile() || info.size <= 0) return { ok: false, error: 'PDF ว่างหรืออ่านไม่ได้' }
    return await _agentJsonRequest({
      method: 'POST',
      route: '/pdf-to-text/start',
      filePath,
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Length': String(info.size),
        'X-File-Name': encodeURIComponent(path.basename(filePath)),
        'X-Rewrite-Thai': rewriteThai ? '1' : '0',
      },
    })
  } catch (e) {
    return { ok: false, error: String(e && e.message || e) }
  }
})

ipcMain.handle('pdf-to-text-status', async (_e, jobId) => {
  const id = String(jobId || '')
  return _agentJsonRequest({ route: `/pdf-to-text/status?job_id=${encodeURIComponent(id)}` })
})

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
    else mainWindow.setSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
  }
})

// Top-level /exit from the renderer → confirm in a native dialog, then quit.
// before-quit tears down this Electron-owned agent_server process only. The
// model server is independent owner state and remains available until the user
// presses Stop in Settings or runs agent_stop.command.
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
      detail: 'การทำงานที่ค้างอยู่จะถูกหยุด และปิด Agent/UI backend; Model Server จะคงสถานะตาม Settings',
    })
    if (response === 1) app.quit()
  } finally {
    _exitConfirming = false
  }
})

// Edit → open the file in the OS default app (shell.openPath). Returns '' on
// success or an error string the renderer can surface.
ipcMain.handle('edit-file', async (_e, filePath) => {
  if (typeof filePath !== 'string') return 'invalid path'
  const authorized = await _agentJsonRequest({
    method: 'POST', route: '/edit-access/authorize',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: filePath }),
  })
  if (!authorized || !authorized.path) return authorized.error || 'outside approved edit scope'
  try {
    return (await shell.openPath(authorized.path)) || ''
  } catch (e) {
    return String(e && e.message || e)
  }
})

// Delete → confirm in a native dialog (destructive), then unlink. Only files, never
// directories. Returns {deleted} / {cancelled} / {error} for the renderer.
ipcMain.handle('delete-file', async (_e, filePath) => {
  if (typeof filePath !== 'string') return { deleted: false, error: 'invalid path' }
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
  const result = await _agentJsonRequest({
    method: 'POST', route: '/edit-access/delete',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: filePath }),
  })
  return result && result.deleted
    ? { deleted: true }
    : { deleted: false, error: result && result.error || 'outside approved edit scope' }
})

ipcMain.handle('show-open-dialog', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
  })
  return canceled ? null : (filePaths[0] || null)
})

// Pin selection may come from anywhere the user can browse. Renderer state is
// per Electron launch only; the Python backend is the policy authority and
// revalidates each path with the same protected-path guard as read_file on every
// normal query turn before any read occurs.
ipcMain.handle('show-pin-dialog', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    title: 'เลือกไฟล์เพื่อ Pin',
    properties: ['openFile', 'multiSelections'],
  })
  if (canceled) return { paths: [], rejected: 0 }
  const paths = []
  let rejected = 0
  for (const selected of filePaths) {
    try {
      const real = fs.realpathSync(selected)
      const info = fs.statSync(real)
      if (!info.isFile()) { rejected += 1; continue }
      paths.push(real)
    } catch {
      rejected += 1
    }
  }
  return { paths, rejected }
})

// Approved Edit / Focus selection is directory-only. The backend remains the
// policy authority and validates every returned path again before persisting it.
ipcMain.handle('show-approved-edit-folder-dialog', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    title: 'เลือกโฟลเดอร์สำหรับสิทธิ์แก้ไข',
    properties: ['openDirectory', 'multiSelections'],
  })
  if (canceled) return { paths: [], rejected: 0 }
  const paths = []
  let rejected = 0
  for (const selected of filePaths) {
    try {
      const real = fs.realpathSync(selected)
      const info = fs.statSync(real)
      if (!info.isDirectory()) { rejected += 1; continue }
      paths.push(real)
    } catch {
      rejected += 1
    }
  }
  return { paths, rejected }
})
