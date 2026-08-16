/* renderer.js — WebSocket client + UI logic */

const WS_URL = 'ws://localhost:8765/ws'
const MAX_ACTIVITY = 500

const FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
let frameIdx = 0
let spinTimer = null

let ws = null
let wsConnected = false
let isBusy = false
let reconnectTimer = null
let currentSkill = ''
let currentHandoffLabel = ''  // Agent Handoff persona display name ('' = normal Agent)
let availableSkills = []
let builtinCmds = []
let thinkingEl = null
let firstAgentBubbleThisTurn = null  // refs must attach to this turn's main answer, not a later '(ต่อ)' continuation bubble
let streamingActive = false  // true while token stream is building response bubble
let workspaceRoot = ''
let currentDirPath = ''

let suggestIdx = -1
let suggestItems = []

let wsToken = null  // cached once; '' in browser dev mode (server needs AGENT_AUTH_DISABLED=1)
let currentRunId = null  // set on 'start' event, cleared on done/error/cancelled
let cancelPending = false  // true between clicking stop and the turn actually stopping; locks the "stopping…" notice so late phase/progress events don't overwrite it
let activePanel = null  // 'workspace' | 'activity' | null
let attachedFilePath = null  // path of file attached via 📎 button
let currentToolName = ''
let pendingTurnQuery = ''
let waitingSummaryTimer = null

const THINKING_SUMMARY_DELAY_MS = 6000

const _ATTACH_IMAGE_EXT = new Set(['.png','.jpg','.jpeg','.gif','.bmp','.webp','.heic','.heif','.tiff','.tif'])
const _ATTACH_AUDIO_EXT = new Set(['.m4a','.mp3','.wav','.aiff','.aif','.caf','.flac','.aac'])
const _ATTACH_VIDEO_EXT = new Set(['.mp4','.mov','.m4v'])
function _fileHint(p) {
  const ext = p.lastIndexOf('.') >= 0 ? p.slice(p.lastIndexOf('.')).toLowerCase() : ''
  if (_ATTACH_IMAGE_EXT.has(ext)) return `[ไฟล์แนบ (รูปภาพ): ${p}]\nใช้ tool: read_image`
  if (_ATTACH_AUDIO_EXT.has(ext)) return `[ไฟล์แนบ (เสียง): ${p}]\nใช้ tool: read_file (จะถอดเสียงเป็นข้อความอัตโนมัติ)`
  if (_ATTACH_VIDEO_EXT.has(ext)) return `[ไฟล์แนบ (วิดีโอ): ${p}]\nใช้ tool: read_file (จะถอดเสียงจากวิดีโอเป็นข้อความอัตโนมัติ)`
  return `[ไฟล์แนบ: ${p}]\nใช้ tool: read_file`
}

// ── Theme ──────────────────────────────────────────────────────────────────────
// lib/theme.js already set html[data-theme] + the hljs link before first paint
// (avoids a flash of the wrong theme); this re-applies it so the toggle button
// icon/title stay in sync, and handles user-triggered toggles afterward.
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme)
  try {
    localStorage.setItem('endeavor-theme', theme)
  } catch (e) {
    console.warn('theme persistence failed:', e)
  }
  const hljsLink = document.getElementById('hljs-theme')
  if (hljsLink) {
    hljsLink.href = theme === 'light'
      ? './node_modules/@highlightjs/cdn-assets/styles/github.min.css'
      : './node_modules/@highlightjs/cdn-assets/styles/github-dark.min.css'
  }
  const btn = document.getElementById('btn-theme')
  if (btn) {
    btn.textContent = theme === 'light' ? '☀️' : '🌙'
    btn.title = theme === 'light' ? 'สลับเป็น Dark' : 'สลับเป็น Light'
  }
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark'
  applyTheme(current === 'light' ? 'dark' : 'light')
}

// ── Compact mode ───────────────────────────────────────────────────────────────
// Mini always-on-top widget, same idea as SERVER_MONITOR's Compact toggle: hide
// sidebar/context bar, shrink the window (main.js owns the actual resize/pin via
// the 'set-compact' IPC message), keep only chat + input visible.
function applyCompact(compact) {
  document.body.classList.toggle('compact', compact)
  const btn = document.getElementById('btn-compact')
  if (btn) {
    btn.textContent = compact ? '▤' : '◱'
    btn.title = compact ? 'กลับสู่หน้าต่างเต็ม' : 'ย่อเป็นหน้าต่างเล็ก'
  }
  if (window.electronAPI && window.electronAPI.setCompact) window.electronAPI.setCompact(compact)
  const box = document.getElementById('messages')
  if (box) box.scrollTop = box.scrollHeight
}

function toggleCompact() {
  applyCompact(!document.body.classList.contains('compact'))
}

// ── WebSocket ──────────────────────────────────────────────────────────────────

async function getToken() {
  if (wsToken !== null) return wsToken
  if (window.electronAPI && window.electronAPI.getToken) {
    try { wsToken = await window.electronAPI.getToken() } catch { wsToken = '' }
  } else {
    wsToken = ''
  }
  return wsToken
}

async function connect() {
  clearTimeout(reconnectTimer)
  const token = await getToken()
  // Token rides in Sec-WebSocket-Protocol instead of ?token= — keeps it out
  // of server access logs. Server echoes the subprotocol back on accept.
  ws = token ? new WebSocket(WS_URL, [token]) : new WebSocket(WS_URL)

  // Effects injected into applyOpen/applyClose (lib/busy_state.js) — keeps the
  // connection-lifecycle logic single-source + spy-testable (UI-R8-1).
  const _disableInput = d => {
    document.getElementById('send-btn').disabled = d
    document.getElementById('input').disabled = d
  }
  const _connFx = {
    setStreamingActive: v => { streamingActive = v },
    setBusy,
    showCancelButton,
    setRunId: v => { currentRunId = v },
    discardStreaming,
    removeThinking,
    disableInput: _disableInput,
    setPhase,
    scheduleReconnect: () => { reconnectTimer = setTimeout(connect, 2000) },
    hideLabelNotice,
  }

  ws.addEventListener('open', () => {
    wsConnected = true
    console.log('[ws] connected')
    applyOpen({ isBusy }, _connFx)
  })

  ws.addEventListener('close', () => {
    wsConnected = false
    console.log('[ws] disconnected')
    applyClose({ isBusy }, _connFx)
  })

  ws.addEventListener('error', () => {
    // Fires when the TCP connection itself fails (e.g. port 8765 not open yet).
    // The 'close' event always follows, so scheduleReconnect is handled there.
    // Surface a visible hint so the user isn't stuck watching a silent spinner.
    setPhase('⚠️ เชื่อมต่อ agent server ไม่ได้ — กำลังลองใหม่…')
  })

  ws.addEventListener('message', e => {
    try {
      handleEvent(JSON.parse(e.data))
    } catch (err) {
      console.error('[ws] parse error', err)
    }
  })
}

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj))
  }
}

// ── Event handler ──────────────────────────────────────────────────────────────

function handleEvent(ev) {
  switch (ev.type) {
    case 'status':
      applyStatus(ev)
      break
    case 'files':
      if (ev.root) workspaceRoot = ev.root
      if (ev.path) currentDirPath = ev.path
      renderFiles(ev.files || [], ev.path || '', ev.root || '')
      break
    case 'file_content':
      openModal(ev.path, ev.content)
      break
    case 'file_image':
      openImageModal(ev.path)
      break
    case 'start':
      currentRunId = ev.run_id || null
      cancelPending = false  // fresh turn — clear any leftover stop lock
      streamingActive = false
      currentToolName = ''
      firstAgentBubbleThisTurn = null  // reset so refs attach to this turn's first bubble, not a stale one
      addActivitySeparator()
      setPhase(normalizePhase('thinking…', ''), true)
      showThinking()
      updateThinkingLabel(normalizePhase('thinking…', ''))
      scheduleWaitingSummary()
      showCancelButton(true)
      break
    case 'phase':
      setPhase(normalizePhase(ev.label, currentToolName), true)
      updateThinkingLabel(normalizePhase(ev.label, currentToolName))
      break
    case 'progress':
      updateThinkingSub(friendlyProgress(currentToolName, ev.msg))
      break
    case 'tool':
      currentToolName = ev.name || ''
      clearWaitingSummary()
      {
        const ui = toolDisplay(ev.name, ev.detail || '')
        addActivity(ev.name, ui.activityDetail || '')
        updateThinkingLabel(ui.label)
        updateThinkingSub(ui.sub || '')
      }
      break
    case 'plan':
      addPlanBlock(ev.steps || [])
      break
    case 'ctx_update':
      renderCtxBar(ev.chars, ev.max_chars)
      break
    case 'token':
      clearWaitingSummary()
      if (!streamingActive) startStreamingBubble()
      appendStreamToken(ev.text)
      break
    case 'discard_stream':
      discardStreaming()
      break
    case 'response':
      clearWaitingSummary()
      currentToolName = ''
      pendingTurnQuery = ''
      if (streamingActive) {
        finalizeStreaming(ev.content)
      } else {
        removeThinking()
        addMessage('agent', ev.content)
      }
      break
    case 'error':
      streamingActive = false
      clearWaitingSummary()
      currentToolName = ''
      pendingTurnQuery = ''
      removeThinking()
      addMessage('error', ev.msg || 'Unknown error')
      setPhase('พร้อม', false)
      setBusy(false)
      showCancelButton(false)
      currentRunId = null
      break
    case 'refs':
      appendWebRefs(ev.refs || [])
      break
    case 'done':
      streamingActive = false
      clearWaitingSummary()
      currentToolName = ''
      pendingTurnQuery = ''
      removeThinking()
      setPhase('พร้อม', false)
      setBusy(false)
      showCancelButton(false)
      currentRunId = null
      break
    case 'cancelled':
      streamingActive = false
      clearWaitingSummary()
      currentToolName = ''
      pendingTurnQuery = ''
      removeThinking()
      addSystem('ยกเลิกการทำงาน')
      setPhase('พร้อม', false)
      setBusy(false)
      showCancelButton(false)
      currentRunId = null
      break
    case 'clear_ok':
      clearMessages()
      addSystem('เคลียร์ context แล้ว — เริ่ม session ใหม่')
      break
    case 'compact_result':
      removeThinking()
      setPhase('พร้อม', false)
      setBusy(false)
      if (ev.error) {
        addSystem('⚠ compact: ' + ev.error)
      } else {
        const fmt = v => v >= 1000 ? (v / 1000).toFixed(1) + 'k' : String(v)
        addSystem(`บีบอัด context: ${fmt(ev.before)} → ${fmt(ev.after)} chars (ตัด ${ev.cut} messages)`)
      }
      break
    case 'memory_ok':
      removeThinking()
      setBusy(false)
      addSystem(`โหลด history แล้ว — ${ev.loaded_pairs ?? ev.turns ?? 0}/${ev.total_pairs ?? ev.turns ?? 0} pairs (${(ev.loaded_chars ?? 0).toLocaleString()} chars)\n\nหัวข้อใน history:\n${ev.topics ?? ''}`)
      break
    case 'history_list':
      renderHistory(ev.pairs || [], ev.total || 0)
      break
    case 'rag_rebuild_ok': {
      removeThinking()
      setBusy(false)
      const issues = ev.health_issues || []
      const ghostCount = ev.ghost_count || 0
      let healthLine = '\n\n✓ self-check: no anomalies'
      if (issues.length || ghostCount) {
        const lines = issues.map(i => `  - ${i}`)
        if (ghostCount) lines.push(`  - ${ghostCount} registered file(s) have zero chunks (dedup ghosts, informational)`)
        healthLine = `\n\n⚠ self-check found anomalies:\n${lines.join('\n')}`
      }
      if (ev.total === 0) {
        addSystem(`build_index — folder: ${ev.data_dir || 'knowledge'}\nไม่มีไฟล์ให้ index${healthLine}`)
      } else {
        const topicList = (ev.topics || []).join(' | ')
        addSystem(`build_index — folder: ${ev.data_dir || 'knowledge'}\nfound: ${ev.total} file(s)\nTopics: ${topicList || '(ไม่มี topic)'}${healthLine}`)
      }
      break
    }
    case 'skill_change':
      // A handoff frame carries `handoff`/`label`; a skill frame carries `skill`.
      // They are mutually exclusive — entering one clears the other.
      if (ev.handoff) {
        currentHandoffLabel = ev.label || ''
        currentSkill = ''
        updateSkillBadge(currentHandoffLabel)
      } else {
        currentSkill = ev.skill || ''
        currentHandoffLabel = ''
        updateSkillBadge(currentSkill)
      }
      if (ev.msg) {
        if (ev.error) addMessage('error', ev.msg)
        else addSystem(ev.msg)
      }
      buildSkillMenu()
      break
  }
}

// ── Status ─────────────────────────────────────────────────────────────────────

function applyStatus(s) {
  setDot('dot-mlx', 'lbl-mlx', s.server_up, mlxLabel(s.mlx_url))
  setDot('dot-net', 'lbl-net', s.online, 'Internet')
  setModelLabel(s.model)
  // Restore handoff label first (takes precedence); else fall back to skill.
  currentHandoffLabel = s.handoff_label || ''
  currentSkill = currentHandoffLabel ? '' : (s.skill || '')
  updateSkillBadge(currentHandoffLabel || currentSkill)
  if (s.builtin_cmds) builtinCmds = s.builtin_cmds
  if (s.skills) {
    availableSkills = s.skills
    buildSkillMenu()
  }
}

function setDot(dotId, lblId, up, label) {
  const dot = document.getElementById(dotId)
  dot.className = 'dot ' + (up ? 'green' : 'red')
  document.getElementById(lblId).textContent = label
}

// mlxLabel(url) and shortModelName(model) are defined in lib/format.js (loaded
// as a global via <script> before this file).

// The local model the agent actually calls (config.MODEL, sent in the status payload).
// Show the short name in the header; full id on hover. Empty model → label stays blank
// (CSS hides it).
function setModelLabel(model) {
  const el = document.getElementById('lbl-model')
  if (!el) return
  if (!model) { el.textContent = ''; el.title = 'local model'; return }
  el.textContent = shortModelName(model)
  el.title = 'local model: ' + model
}

function updateSkillBadge(skill) {
  const b = document.getElementById('skill-badge')
  if (skill) { b.textContent = '/' + skill; b.style.display = 'inline' }
  else b.style.display = 'none'
}

// ── Phase bar ──────────────────────────────────────────────────────────────────

function setPhase(label, active) {
  // While a cancel is pending, keep the "stopping…" notice pinned — don't let late
  // phase events from the still-draining turn overwrite it. A terminal transition
  // (active === false: done/error/cancelled) clears the lock and updates normally.
  if (cancelPending && active) return
  if (!active) cancelPending = false
  const bar = document.getElementById('phase-bar')
  document.getElementById('phase-text').textContent = label
  bar.classList.toggle('active', active)
}

// ── Activity ───────────────────────────────────────────────────────────────────

function clearActivity() {
  document.getElementById('activity-list').innerHTML = ''
}

// actTimestamp(date) is defined in lib/format.js (loaded as a global via <script>
// before this file).
function _actTs() {
  return actTimestamp(new Date())
}

function addActivitySeparator() {
  const list = document.getElementById('activity-list')
  // Skip if list is empty (first turn)
  if (!list.children.length) return
  const el = document.createElement('div')
  el.className = 'act-sep'
  el.textContent = `─── ${_actTs()} ───`
  list.appendChild(el)
  list.scrollTop = list.scrollHeight
}

const TOOL_ICONS = {
  web_search: '🔍', browse_url: '🌐', browser_use: '🌐', recall_web: '📋',
  write_file: '✏️', read_file: '📖', edit: '✏️', bash: '⚡',
  python_exec: '🐍', grep: '🔎', create_plan: '📋', workspace_ls: '📁',
  rag_search: '🗂', remember: '🧠', research_orchestrator: '🔬',
  fetch_sitemap: '🗺', batch_browse: '📦',
  plot: '📊', scrape_table: '📊', read_image: '🖼', tool_loop: '🔁',
  security_scan: '🛡', rag_ls: '🗂', rag_list_knowledge: '🗂',
  rag_rebuild_index: '🗂',
}

function addActivity(name, detail) {
  const list = document.getElementById('activity-list')
  const el = document.createElement('div')
  el.className = 'act-item'
  const icon = TOOL_ICONS[name] || '⚙'
  el.innerHTML = `
    <div class="act-name">${icon} ${esc(name)}<span class="act-ts">${_actTs()}</span></div>
    ${detail ? `<div class="act-detail">⎿ ${esc(detail)}</div>` : ''}
  `
  list.appendChild(el)
  while (list.children.length > MAX_ACTIVITY) list.removeChild(list.firstChild)
  list.scrollTop = list.scrollHeight
}

function addPlanBlock(steps) {
  const list = document.getElementById('activity-list')
  const el = document.createElement('div')
  el.className = 'act-item'
  const stepsHtml = steps.map((s, i) => `<div class="act-detail">⎿ ${i + 1}. ${esc(s)}</div>`).join('')
  el.innerHTML = `
    <div class="act-name">📋 แผน · ${steps.length} ขั้นตอน<span class="act-ts">${_actTs()}</span></div>
    ${stepsHtml}
  `
  list.appendChild(el)
  while (list.children.length > MAX_ACTIVITY) list.removeChild(list.firstChild)
  list.scrollTop = list.scrollHeight
}

// ── Messages ───────────────────────────────────────────────────────────────────

function agentLabel() {
  // Handoff persona (e.g. "Agent_Invest") takes precedence over the default
  // 'Agent' / 'Agent|<skill>' label.
  return currentHandoffLabel || (currentSkill ? 'Agent|' + currentSkill : 'Agent')
}

function addMessage(role, content) {
  const box = document.getElementById('messages')
  const labels = { user: 'You', agent: agentLabel(), error: '⚠ Error' }
  const wrap = document.createElement('div')
  wrap.className = `msg ${role}`
  const body = role === 'agent' ? renderMarkdown(content) : esc(content)
  wrap.innerHTML = `
    <div class="msg-label">${labels[role] || role}</div>
    <div class="msg-bubble">${body}</div>
  `
  box.appendChild(wrap)
  if (role === 'agent') {
    highlightChildren(wrap)  // syntax-highlight fenced code blocks
    if (!firstAgentBubbleThisTurn) firstAgentBubbleThisTurn = wrap.querySelector('.msg-bubble')
  }
  box.scrollTop = box.scrollHeight
}

function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text))
}

function addSystem(text) {
  const box = document.getElementById('messages')
  const el = document.createElement('div')
  el.className = 'msg system'
  el.innerHTML = `<div class="msg-bubble">— ${esc(text)} —</div>`
  box.appendChild(el)
  box.scrollTop = box.scrollHeight
}

function clearMessages() {
  document.getElementById('messages').innerHTML = ''
  thinkingEl = null
}

// ── Streaming response helpers ─────────────────────────────────────────────────

function startStreamingBubble() {
  if (!thinkingEl) showThinking()
  clearWaitingSummary()
  stopSpin()  // stop AFTER ensuring thinkingEl exists (showThinking calls startSpin)
  thinkingEl.classList.add('streaming')
  const bubble = thinkingEl.querySelector('.msg-bubble')
  bubble.className = 'msg-bubble stream-bubble'
  bubble.innerHTML = '<span class="stream-text"></span><span class="stream-cursor">▋</span>'
  streamingActive = true
}

function appendStreamToken(text) {
  if (!thinkingEl) return
  const span = thinkingEl.querySelector('.stream-text')
  if (span) {
    // Skip leading whitespace until real content arrives
    if (!span.textContent && !text.trim()) return
    span.textContent += text
  }
  const box = document.getElementById('messages')
  box.scrollTop = box.scrollHeight
}

function appendWebRefs(refs) {
  if (!refs || refs.length === 0) return
  const messages = document.getElementById('messages')
  // Prefer this turn's first/main answer bubble — a turn can render several
  // bubbles now ('(ต่อ)' continuations), and refs always arrive last, so
  // falling back to the last bubble in the DOM would misattach citations.
  let targetBubble = firstAgentBubbleThisTurn
  if (!targetBubble) {
    const agentMsgs = messages.querySelectorAll('.msg.agent')
    if (!agentMsgs.length) return
    targetBubble = agentMsgs[agentMsgs.length - 1].querySelector('.msg-bubble')
  }
  if (!targetBubble) return
  const items = refs.map(r => `<li><a href="${safeHref(r.url)}" target="_blank">${esc(r.title || r.url)}</a></li>`).join('')
  const block = document.createElement('details')
  block.className = 'web-refs'
  block.innerHTML = `<summary>📎 แหล่งอ้างอิง (${refs.length} ลิงก์)</summary><ul>${items}</ul>`
  targetBubble.appendChild(block)
  messages.scrollTop = messages.scrollHeight
}

function finalizeStreaming(content) {
  if (!thinkingEl) return
  stopSpin()
  thinkingEl.classList.remove('streaming')
  const bubble = thinkingEl.querySelector('.msg-bubble')
  bubble.className = 'msg-bubble'
  bubble.innerHTML = renderMarkdown(content)
  highlightChildren(bubble)  // syntax-highlight fenced code blocks (full content is in now)
  if (!firstAgentBubbleThisTurn) firstAgentBubbleThisTurn = bubble
  thinkingEl = null  // detach — element stays in DOM as the final message
  streamingActive = false
  document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight
}

function discardStreaming() {
  streamingActive = false
  clearWaitingSummary()
  removeThinking()
  showThinking()
}

function showThinking() {
  removeThinking()
  const box = document.getElementById('messages')
  thinkingEl = document.createElement('div')
  thinkingEl.className = 'msg agent'
  thinkingEl.innerHTML = `
    <div class="msg-label">${agentLabel()}</div>
    <div class="msg-bubble thinking-bubble">
      <div class="ts-main">
        <span class="ts-frame">${FRAMES[0]}</span>
        <span class="ts-label">thinking…</span>
      </div>
      <div class="ts-sub" style="display:none">
        <span class="ts-arrow">⎿</span>
        <span class="ts-detail"></span>
      </div>
    </div>
  `
  box.appendChild(thinkingEl)
  box.scrollTop = box.scrollHeight
  startSpin()
}

function removeThinking() {
  stopSpin()
  clearWaitingSummary()
  if (thinkingEl) { thinkingEl.remove(); thinkingEl = null }
}

function scheduleWaitingSummary() {
  clearWaitingSummary()
  const q = pendingTurnQuery
  if (!q) return
  waitingSummaryTimer = setTimeout(() => {
    if (!thinkingEl || streamingActive || currentToolName) return
    updateThinkingSub(summarizeWaitingQuery(q))
  }, THINKING_SUMMARY_DELAY_MS)
}

function clearWaitingSummary() {
  if (waitingSummaryTimer) {
    clearTimeout(waitingSummaryTimer)
    waitingSummaryTimer = null
  }
}

function updateThinkingLabel(label) {
  if (cancelPending || !thinkingEl) return  // keep "stopping…" pinned during cancel drain
  const el = thinkingEl.querySelector('.ts-label')
  if (el) el.textContent = label
}

function updateThinkingSub(msg) {
  if (cancelPending || !thinkingEl) return  // keep "stopping…" pinned during cancel drain
  const el = thinkingEl.querySelector('.ts-detail')
  const sub = thinkingEl.querySelector('.ts-sub')
  if (el) el.textContent = msg || ''
  if (sub) sub.style.display = msg ? '' : 'none'
}

function startSpin() {
  stopSpin()
  frameIdx = 0
  spinTimer = setInterval(() => {
    if (!thinkingEl) return
    const el = thinkingEl.querySelector('.ts-frame')
    if (el) { el.textContent = FRAMES[frameIdx % FRAMES.length]; frameIdx++ }
  }, 80)
}

function stopSpin() {
  if (spinTimer) { clearInterval(spinTimer); spinTimer = null }
}

// ── Send ───────────────────────────────────────────────────────────────────────

function sendMessage() {
  const inp = document.getElementById('input')
  const q = inp.value.trim()
  hideSuggestions()
  if ((!q && !attachedFilePath) || !wsConnected || isBusy) return

  if (q.startsWith('/')) {
    const cmd = q.toLowerCase()
    if (cmd === '/clear') { sendClear(); inp.value = ''; autoResize(inp); return }
    if (cmd === '/exit' || cmd === '/quit') {
      // Progressive back-out: inside a skill OR a handoff persona → leave that
      // mode (backend's toggle_skill('exit') already closes either, see
      // agent_server.py); only quit the app once neither is active.
      if (currentSkill || currentHandoffLabel) {
        addMessage('user', q)
        wsSend({ type: 'command', cmd: '/exit' })
      } else if (window.electronAPI && window.electronAPI.requestExit) {
        window.electronAPI.requestExit()
      }
      inp.value = ''; autoResize(inp); return
    }
    addMessage('user', q)
    wsSend({ type: 'command', cmd: q })
    inp.value = ''; autoResize(inp)
    // Slow blocking commands hold the server's _busy lock — lock the input too, or
    // the user can fire a query that the server rejects with a confusing
    // "agent กำลังทำงานอยู่" while the UI shows no busy state. The matching response
    // events (compact_result / memory_ok / rag_rebuild_ok) and 'error' all clear it.
    if (cmd === '/compact') { setBusy(true); showThinking(); updateThinkingLabel('กำลังบีบอัด context…') }
    else if (cmd === '/history') { setBusy(true); showThinking(); updateThinkingLabel('กำลังโหลดความจำ…') }
    else if (cmd === '/build_index') { setBusy(true); showThinking(); updateThinkingLabel('กำลังสร้าง RAG index…') }
    return
  }

  const content = attachedFilePath
    ? (q ? `${q}\n\n${_fileHint(attachedFilePath)}` : _fileHint(attachedFilePath))
    : q
  pendingTurnQuery = content
  addMessage('user', attachedFilePath
    ? (q ? `${q}  📎 ${attachedFilePath.split('/').pop()}` : `📎 ${attachedFilePath.split('/').pop()}`)
    : q)
  inp.value = ''; autoResize(inp)
  setAttachment(null)
  setBusy(true)
  wsSend({ type: 'query', content })
}

function sendClear() {
  if (!wsConnected) return
  wsSend({ type: 'command', cmd: '/clear' })
}

// ── File attachment ────────────────────────────────────────────────────────────

function setAttachment(filePath) {
  attachedFilePath = filePath
  const badge = document.getElementById('file-badge')
  const name  = document.getElementById('file-badge-name')
  if (filePath) {
    name.textContent = filePath.split('/').pop()
    name.title = filePath
    badge.style.display = 'flex'
  } else {
    badge.style.display = 'none'
    name.textContent = ''
    name.title = ''
  }
}

async function handleUpload() {
  if (!window.electronAPI) return
  const filePath = await window.electronAPI.showOpenDialog()
  if (filePath) setAttachment(filePath)
}

function sendCancel() {
  if (!wsConnected || !currentRunId) return
  wsSend({ type: 'command', cmd: 'cancel', run_id: currentRunId })
  // Cancel is not instant — the running turn only stops once it reaches the next
  // checkpoint, which can take a moment (mid token-gen or mid web_search). Without
  // feedback the click feels like nothing happened. Acknowledge it immediately:
  // the phase bar is always visible, the thinking label updates when present, and
  // the button is disabled so a frustrated user can't spam more cancels.
  setPhase('กำลังหยุด… รอสักครู่', true)
  clearWaitingSummary()
  updateThinkingLabel('กำลังหยุด…')
  updateThinkingSub('รอสักครู่ กำลังหยุดการทำงาน')
  cancelPending = true  // set AFTER the calls above so they aren't blocked by their own guard
  const btn = document.getElementById('cancel-btn')
  if (btn) btn.disabled = true
}

function showCancelButton(visible) {
  const btn = document.getElementById('cancel-btn')
  if (!btn) return
  btn.style.display = visible ? '' : 'none'
  // Re-enable on each new turn — a prior cancel leaves it disabled (see sendCancel).
  if (visible) btn.disabled = false
}

function setBusy(busy) {
  isBusy = busy
  document.getElementById('send-btn').disabled = busy
  document.getElementById('input').disabled = busy
}

// ── Files ──────────────────────────────────────────────────────────────────────

const FILE_ICONS = {
  md: '📄', txt: '📄', json: '📋', py: '🐍', js: '📜',
  csv: '📊', png: '🖼', jpg: '🖼', jpeg: '🖼', gif: '🖼',
  webp: '🖼', svg: '🖼', pdf: '📕', sh: '⚡',
}

function fileIcon(entry) {
  if (entry.type === 'dir') return '📂'
  const ext = entry.name.split('.').pop().toLowerCase()
  return FILE_ICONS[ext] || '📄'
}

function renderFiles(files, currentPath, root) {
  const list = document.getElementById('file-list')
  list.innerHTML = ''

  // Update breadcrumb
  const crumb = document.getElementById('files-crumb')
  const backBtn = document.getElementById('files-back')
  if (root && currentPath) {
    const rel = currentPath === root ? 'workspace' : currentPath.replace(root, '').replace(/^\//, '')
    crumb.textContent = rel || 'workspace'
    backBtn.style.display = currentPath !== root ? 'inline' : 'none'
  }

  if (!files.length) {
    list.innerHTML = '<div class="files-empty">โฟลเดอร์ว่างเปล่า</div>'
    return
  }
  for (const f of files) {
    const el = document.createElement('div')
    el.className = 'file-item'
    const sizeStr = f.type === 'dir' ? '' : fmtSize(f.size)
    const actions = f.type === 'dir' ? '' : `
      <span class="file-actions">
        <button class="file-act" data-act="edit" title="แก้ไขด้วยแอปของระบบ">✏️</button>
        <button class="file-act" data-act="del" title="ลบไฟล์">🗑</button>
      </span>`
    el.innerHTML = `
      <span class="file-icon">${fileIcon(f)}</span>
      <span class="file-name" title="${esc(f.path)}">${esc(f.name)}</span>
      ${actions}
      <span class="file-size">${sizeStr}</span>
    `
    el.onclick = () => wsSend({ type: 'open_file', path: f.path })
    // stopPropagation so the action click doesn't also trigger the row's open_file
    const editBtn = el.querySelector('[data-act="edit"]')
    const delBtn = el.querySelector('[data-act="del"]')
    if (editBtn) editBtn.onclick = (e) => { e.stopPropagation(); editWorkspaceFile(f.path) }
    if (delBtn) delBtn.onclick = (e) => { e.stopPropagation(); deleteWorkspaceFile(f.path, f.name) }
    list.appendChild(el)
  }
}

async function editWorkspaceFile(filePath) {
  if (!window.electronAPI || !window.electronAPI.editFile) return
  const err = await window.electronAPI.editFile(filePath)
  if (err) addSystem('เปิดไฟล์เพื่อแก้ไขไม่ได้: ' + err)
}

async function deleteWorkspaceFile(filePath, name) {
  if (!window.electronAPI || !window.electronAPI.deleteFile) return
  const res = await window.electronAPI.deleteFile(filePath)  // native confirm happens in main
  if (res && res.deleted) {
    addSystem('ลบไฟล์แล้ว: ' + name)
    wsSend({ type: 'get_files', path: currentDirPath || workspaceRoot })  // refresh list from disk
  } else if (res && res.error) {
    addSystem('ลบไม่ได้: ' + res.error)
  }
}

function navigateUp() {
  if (!currentDirPath || currentDirPath === workspaceRoot) return
  const parent = currentDirPath.split('/').slice(0, -1).join('/') || workspaceRoot
  const target = parent.startsWith(workspaceRoot) ? parent : workspaceRoot
  wsSend({ type: 'get_files', path: target })
}

// ── File modal ─────────────────────────────────────────────────────────────────

let _modalRaw = null  // original file text — rendered HTML loses the source, so copy uses this

// Extension → highlight.js language. Anything here is syntax-highlighted; .md is
// rendered as markdown (tables/headings); .csv becomes a table; the rest stay plain.
const _CODE_LANG = {
  py:'python', pyw:'python', js:'javascript', mjs:'javascript', cjs:'javascript',
  ts:'typescript', jsx:'javascript', tsx:'typescript', json:'json', jsonl:'json',
  html:'xml', htm:'xml', xml:'xml', svg:'xml', css:'css', scss:'scss', less:'less',
  sh:'bash', bash:'bash', zsh:'bash', yml:'yaml', yaml:'yaml', toml:'ini', ini:'ini',
  cfg:'ini', conf:'ini', env:'ini', sql:'sql', java:'java', c:'c', h:'c', cpp:'cpp',
  cc:'cpp', hpp:'cpp', go:'go', rs:'rust', rb:'ruby', php:'php', swift:'swift',
  kt:'kotlin', r:'r', lua:'lua', pl:'perl', dockerfile:'dockerfile', makefile:'makefile',
}
const _MD_EXT = new Set(['md', 'markdown', 'mdown', 'mkd'])

function highlightChildren(root) {
  if (typeof hljs === 'undefined') return
  root.querySelectorAll('pre code').forEach(el => { try { hljs.highlightElement(el) } catch (e) {} })
}

// Quote-aware CSV parse (handles commas + newlines inside "..." and "" escapes).
function parseCSV(text) {
  const rows = []
  let row = [], field = '', inQ = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (inQ) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++ } else inQ = false }
      else field += c
    } else if (c === '"') inQ = true
    else if (c === ',') { row.push(field); field = '' }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = '' }
    else if (c !== '\r') field += c
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row) }
  return rows.filter(r => !(r.length === 1 && r[0] === ''))
}

// Build an HTML table directly (not via markdown) so cell content with | or # is safe.
function csvToTable(text) {
  const rows = parseCSV(text)
  if (!rows.length) return null
  const head = rows[0].map(c => `<th>${esc(c)}</th>`).join('')
  const tbody = rows.slice(1).map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')
  return `<table><thead><tr>${head}</tr></thead><tbody>${tbody}</tbody></table>`
}

function openModal(filePath, content) {
  const name = filePath.split('/').pop()
  const ext = name.includes('.') ? name.split('.').pop().toLowerCase() : name.toLowerCase()
  const lines = content.split('\n').length
  document.getElementById('modal-title').textContent = name
  document.getElementById('modal-lines').textContent = `${lines} lines`
  const body = document.getElementById('modal-body')
  _modalRaw = content
  body.className = ''
  body.style.cssText = ''
  body.innerHTML = ''

  if (_MD_EXT.has(ext)) {
    body.classList.add('md')
    body.innerHTML = renderMarkdown(content)
    highlightChildren(body)  // highlight fenced code blocks inside the markdown
  } else if (ext === 'csv') {
    const table = csvToTable(content)
    if (table) { body.classList.add('md'); body.innerHTML = table }
    else { body.style.whiteSpace = 'pre-wrap'; body.textContent = content }
  } else if (_CODE_LANG[ext]) {
    body.classList.add('code')
    const pre = document.createElement('pre')
    const code = document.createElement('code')
    code.className = 'language-' + _CODE_LANG[ext]
    code.textContent = content  // textContent in → hljs escapes safely on highlight
    pre.appendChild(code)
    body.appendChild(pre)
    try { hljs.highlightElement(code) } catch (e) {}
  } else {
    body.style.whiteSpace = 'pre-wrap'  // unknown type → plain text, as before
    body.textContent = content
  }
  document.getElementById('modal-overlay').classList.add('open')
}

function openImageModal(filePath) {
  const name = filePath.split('/').pop()
  _modalRaw = null  // image has no text source to copy
  document.getElementById('modal-title').textContent = name
  document.getElementById('modal-lines').textContent = 'image'
  const body = document.getElementById('modal-body')
  body.className = ''  // clear any code/md class left from a previous file view
  body.style.cssText = 'display:flex;align-items:center;justify-content:center;padding:16px;'
  const img = document.createElement('img')
  img.src = 'file://' + filePath
  img.style.cssText = 'max-width:100%;max-height:60vh;border-radius:4px;object-fit:contain;'
  img.addEventListener('error', () => {
    const err = document.createElement('div')
    err.className = 'modal-img-error'
    err.textContent = 'ไม่สามารถโหลดรูปได้: ' + name
    img.replaceWith(err)
  })
  body.innerHTML = ''
  body.appendChild(img)
  document.getElementById('modal-overlay').classList.add('open')
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open')
}

function copyFile() {
  // Copy the original file text, not the rendered HTML's textContent (which would
  // drop markdown syntax / reflow the table).
  const text = _modalRaw != null ? _modalRaw : document.getElementById('modal-body').textContent
  navigator.clipboard.writeText(text).catch(() => {})
}

// ── Autocomplete ───────────────────────────────────────────────────────────────

function getAllCommands() {
  const builtinNames = new Set(builtinCmds.map(c => c.name))
  const skills = availableSkills
    .map(s => ({ name: s.name || s, desc: s.desc || 'skill mode' }))
    .filter(s => !builtinNames.has(s.name))
  return [...skills, ...builtinCmds]
}

function updateSuggestions(val) {
  const suggestList = document.getElementById('suggest-list')
  if (!val.startsWith('/')) { hideSuggestions(); return }
  const query = val.slice(1).toLowerCase()
  const all = getAllCommands()
  suggestItems = query === '' ? all : all.filter(c => c.name.toLowerCase().startsWith(query))
  suggestIdx = -1
  if (!suggestItems.length) { hideSuggestions(); return }
  suggestList.innerHTML = ''
  for (let i = 0; i < suggestItems.length; i++) {
    const c = suggestItems[i]
    const el = document.createElement('div')
    el.className = 'suggest-item'
    el.dataset.idx = i
    el.innerHTML = `<span class="suggest-cmd">/${esc(c.name)}</span><span class="suggest-desc">${esc(c.desc)}</span>`
    el.addEventListener('mousedown', ev => { ev.preventDefault(); applySuggestion(c.name) })
    suggestList.appendChild(el)
  }
  suggestList.classList.add('open')
}

function hideSuggestions() {
  document.getElementById('suggest-list').classList.remove('open')
  suggestIdx = -1
  suggestItems = []
}

function applySuggestion(name) {
  const inp = document.getElementById('input')
  inp.value = '/' + name + ' '
  inp.focus()
  hideSuggestions()
  autoResize(inp)
}

function moveSuggest(dir) {
  if (!suggestItems.length) return
  const items = document.getElementById('suggest-list').querySelectorAll('.suggest-item')
  if (suggestIdx >= 0 && suggestIdx < items.length) items[suggestIdx].classList.remove('selected')
  suggestIdx = (suggestIdx + dir + suggestItems.length) % suggestItems.length
  items[suggestIdx].classList.add('selected')
  items[suggestIdx].scrollIntoView({ block: 'nearest' })
}

// ── Skill menu ─────────────────────────────────────────────────────────────────

function buildSkillMenu() {
  const container = document.getElementById('skill-menu-items')
  container.innerHTML = ''

  for (const s of availableSkills) {
    const sName = s.name || s
    const el = document.createElement('div')
    el.className = 'skill-menu-item' + (sName === currentSkill ? ' active' : '')
    el.textContent = (sName === currentSkill ? '✓ ' : '  ') + sName
    el.onclick = () => {
      wsSend({ type: 'command', cmd: '/' + sName })
      closeSkillMenu()
    }
    container.appendChild(el)
  }

  if (currentSkill || currentHandoffLabel) {
    const sep = document.createElement('div')
    sep.className = 'skill-menu-sep'
    container.appendChild(sep)
    const exit = document.createElement('div')
    exit.className = 'skill-menu-item exit'
    exit.textContent = currentHandoffLabel ? `  ออกจาก ${currentHandoffLabel} mode` : '  ออกจาก skill mode'
    exit.onclick = () => {
      wsSend({ type: 'command', cmd: '/exit' })
      closeSkillMenu()
    }
    container.appendChild(exit)
  }
}

function toggleSkillMenu(e) {
  const menu = document.getElementById('skill-menu')
  if (menu.classList.contains('open')) { closeSkillMenu(); return }
  const rect = e.target.getBoundingClientRect()
  menu.style.top = (rect.bottom + 4) + 'px'
  menu.style.right = (window.innerWidth - rect.right) + 'px'
  menu.classList.add('open')
  buildSkillMenu()
  e.stopPropagation()
}

function closeSkillMenu() {
  document.getElementById('skill-menu').classList.remove('open')
}

// ── Loading screen ─────────────────────────────────────────────────────────────

function hideLabelNotice() {
  // called on first WS connect
}

function hideLoading() {
  const el = document.getElementById('loading')
  el.classList.add('fade-out')
  setTimeout(() => el.classList.add('hidden'), 420)
}

function setLoadingLog(msg) {
  document.getElementById('loading-log').textContent = msg
}

// ── Workspace ──────────────────────────────────────────────────────────────────

function renderCtxBar(chars, maxChars) {
  if (!maxChars) return
  const pct = Math.min(100, chars / maxChars * 100)
  const filled = Math.round(pct / 100 * 18)
  const graph = '▓'.repeat(filled) + '░'.repeat(18 - filled)
  const fmt = v => v >= 1000 ? (v / 1000).toFixed(1) + 'k' : String(v)

  const pctEl = document.getElementById('ctx-pct')
  document.getElementById('ctx-graph').textContent = graph
  document.getElementById('ctx-chars').textContent = `${fmt(chars)} / ${fmt(maxChars)} chars`

  pctEl.textContent = `${Math.round(pct)}%`
  pctEl.className = pct >= 90 ? 'danger' : pct >= 70 ? 'warn' : ''
}

function openWorkspace() {
  if (window.electronAPI) window.electronAPI.openWorkspace()
}

function togglePanel(name) {
  const panel = document.getElementById('left-panel')
  if (activePanel === name) {
    // Same icon clicked again → collapse
    activePanel = null
    panel.classList.remove('open')
    document.querySelectorAll('.icon-btn').forEach(b => b.classList.remove('active'))
  } else {
    activePanel = name
    panel.classList.add('open')
    document.querySelectorAll('.icon-btn').forEach(b => b.classList.remove('active'))
    const btnEl = document.getElementById('btn-' + name)
    if (btnEl) btnEl.classList.add('active')
    document.querySelectorAll('.left-view').forEach(v => v.classList.remove('active'))
    const viewEl = document.getElementById('left-' + name)
    if (viewEl) viewEl.classList.add('active')
    if (name === 'workspace') {
      wsSend({ type: 'get_files', path: workspaceRoot || '' })
    }
    if (name === 'history') {
      wsSend({ type: 'get_history' })
    }
  }
}

function fmtTs(ts) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  const now = new Date()
  const diffMs = now - d
  const diffMin = Math.floor(diffMs / 60000)
  const diffH = Math.floor(diffMin / 60)
  const diffD = Math.floor(diffH / 24)
  if (diffMin < 1) return 'เมื่อกี้'
  if (diffMin < 60) return `${diffMin} นาทีที่แล้ว`
  if (diffH < 24) return `${diffH} ชม.ที่แล้ว`
  if (diffD < 7) return `${diffD} วันที่แล้ว`
  const dd = String(d.getDate()).padStart(2, '0')
  const mm = String(d.getMonth() + 1).padStart(2, '0')
  const hh = String(d.getHours()).padStart(2, '0')
  const min = String(d.getMinutes()).padStart(2, '0')
  return `${dd}/${mm} ${hh}:${min}`
}

function renderHistory(pairs, total) {
  const list = document.getElementById('history-list')
  const countEl = document.getElementById('history-count')
  if (countEl) countEl.textContent = total ? `${total} รายการ` : ''
  list.innerHTML = ''
  if (!pairs.length) {
    list.innerHTML = '<div class="hist-empty">ยังไม่มีประวัติการสนทนา</div>'
    return
  }
  // Show newest first
  for (let i = pairs.length - 1; i >= 0; i--) {
    const p = pairs[i]
    const el = document.createElement('div')
    el.className = 'hist-item'
    const aPreview = p.a ? p.a.replace(/\n+/g, ' ').slice(0, 80) : '—'
    const tsStr = fmtTs(p.ts)
    el.innerHTML = `
      <div class="hist-idx">#${i + 1}${tsStr ? `<span style="float:right;color:var(--text-faint);font-weight:400">${esc(tsStr)}</span>` : ''}</div>
      <div class="hist-q">${esc(p.q)}</div>
      <div class="hist-a">${esc(aPreview)}</div>
    `
    el.addEventListener('click', () => {
      const shortQ = p.q.length > 60 ? p.q.slice(0, 60) + '…' : p.q
      const content = `**คำถาม**\n\n${p.q}\n\n---\n\n**คำตอบ**\n\n${p.a || '_(ไม่มีคำตอบ)_'}`
      openModal('_history.md', content)
      // Override the auto-generated title/subtitle from the fake filename
      document.getElementById('modal-title').textContent = `#${i + 1} — ${shortQ}`
      document.getElementById('modal-lines').textContent = tsStr || ''
    })
    list.appendChild(el)
  }
}

// ── Utilities ──────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

// safeHref blocks javascript:/data:/vbscript: protocol URLs before they reach an href.
// esc() only HTML-entity-escapes — it does NOT prevent javascript: from executing.
function safeHref(url) {
  const s = String(url || '').trim()
  return /^(javascript|data|vbscript):/i.test(s) ? '#' : esc(s)
}

function fmtSize(n) {
  if (n < 1024) return n + ' B'
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB'
  return (n / 1048576).toFixed(1) + ' MB'
}

function autoResize(el) {
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 120) + 'px'
}

// ── Init ───────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  applyTheme(document.documentElement.getAttribute('data-theme') || 'dark')

  const inp = document.getElementById('input')

  inp.addEventListener('keydown', e => {
    const suggestOpen = document.getElementById('suggest-list').classList.contains('open')
    if (suggestOpen) {
      if (e.key === 'ArrowDown')  { e.preventDefault(); moveSuggest(1); return }
      if (e.key === 'ArrowUp')    { e.preventDefault(); moveSuggest(-1); return }
      if (e.key === 'Escape')     { e.preventDefault(); hideSuggestions(); return }
      if (e.key === 'Tab') {
        e.preventDefault()
        const pick = suggestIdx >= 0 ? suggestItems[suggestIdx] : suggestItems[0]
        if (pick) applySuggestion(pick.name)
        return
      }
      if (e.key === 'Enter' && suggestIdx >= 0) {
        e.preventDefault()
        applySuggestion(suggestItems[suggestIdx].name)
        return
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  })
  inp.addEventListener('input', () => {
    autoResize(inp)
    updateSuggestions(inp.value)
  })

  // CSP: script-src 'self' (no 'unsafe-inline') — all buttons bound here,
  // never via inline onclick attributes in index.html
  const on = (sel, fn) => document.querySelectorAll(sel).forEach(el => el.addEventListener('click', fn))
  on('#skill-badge', toggleSkillMenu)
  on('#btn-skill', toggleSkillMenu)
  on('#btn-theme', toggleTheme)
  on('#btn-compact', toggleCompact)
  on('#btn-workspace', () => togglePanel('workspace'))
  on('#btn-activity', () => togglePanel('activity'))
  on('#btn-history', () => togglePanel('history'))
  on('#send-btn', () => sendMessage())
  on('#cancel-btn', () => sendCancel())
  on('#btn-clear', () => sendClear())
  on('#btn-upload', () => handleUpload())
  on('#file-badge-remove', () => setAttachment(null))
  on('#files-back', () => navigateUp())
  on('.files-open-btn', () => openWorkspace())
  on('.modal-copy', () => copyFile())
  on('.modal-close', () => closeModal())

  document.getElementById('modal-overlay').addEventListener('click', e => {
    if (e.target === document.getElementById('modal-overlay')) closeModal()
  })

  document.addEventListener('click', e => {
    const menu = document.getElementById('skill-menu')
    if (!menu.contains(e.target)) closeSkillMenu()
  })

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeModal(); closeSkillMenu() }
    // Ctrl+C while agent is running → cancel turn (only when no text is selected,
    // so copy still works normally during streaming)
    if (e.ctrlKey && e.key === 'c' && isBusy && currentRunId) {
      const sel = window.getSelection()
      if (!sel || !sel.toString()) {
        e.preventDefault()
        sendCancel()
      }
    }
  })

  // Electron IPC
  if (window.electronAPI) {
    window.electronAPI.onStartupStatus(({ msg, phase }) => {
      setLoadingLog(msg)
    })
    window.electronAPI.onStartupDone(() => {
      // Server processes started — start connecting WebSocket
      connect()
      // Hide loading once WS connects
      const checkHide = setInterval(() => {
        if (wsConnected) { hideLoading(); clearInterval(checkHide) }
      }, 500)
      setTimeout(() => clearInterval(checkHide), 15000)
    })
  } else {
    // Dev mode: no Electron, connect directly
    connect()
    setTimeout(() => { if (wsConnected) hideLoading() }, 1000)
    const checkHide = setInterval(() => {
      if (wsConnected) { hideLoading(); clearInterval(checkHide) }
    }, 500)
    setTimeout(() => clearInterval(checkHide), 15000)
  }
})
