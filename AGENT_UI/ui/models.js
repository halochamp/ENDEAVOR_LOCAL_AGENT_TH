// models.js — the Model page (Agent TH desktop app). Lists the agent's tested models (tools/models.py via the Rust core), their
// size, the RAM they need next to this Mac's RAM, and downloads them in the background. A downloaded model is what
// Settings -> Model offers (renderer.js asks downloadedIds()). Built with textContent only: no data reaches innerHTML.
;(function () {
  const RESERVE_GB = 8 // macOS and the open apps keep this much RAM for themselves
  let data = null
  let timer = null
  let busy = new Set()
  let inFlight = false // a slow answer (offline, a big cache) must not let polls pile up

  const api = () => window.electronAPI && window.electronAPI.models ? window.electronAPI : null
  const el = (tag, cls, text) => {
    const n = document.createElement(tag)
    if (cls) n.className = cls
    if (text != null) n.textContent = text
    return n
  }
  const gb = n => (n == null ? '—' : `${Number.isInteger(n) ? n : n.toFixed(1)} GB`)

  function fit(ram, machine) {
    if (ram == null) return { cls: '', label: 'ยังไม่ได้วัด RAM' }
    const usable = machine - RESERVE_GB
    if (ram <= usable) return { cls: 'fit-ok', label: 'เครื่องนี้พอ', bar: '' }
    if (ram <= machine) return { cls: 'fit-tight', label: 'ตึง: ปิดแอปอื่นก่อน', bar: 'tight' }
    return { cls: 'fit-no', label: 'RAM ไม่พอ', bar: 'no' }
  }

  const ACCEL = { off: 'AR', dflash: 'DFlash', mtp: 'MTP' }

  function statusCell(m) {
    const box = el('div', 'model-status')
    const current = typeof runtimeSettings !== 'undefined' && runtimeSettings && runtimeSettings.model === m.id
    if (m.state === 'installed') {
      box.append(el('span', 'ok', current ? 'ใช้อยู่' : 'ดาวน์โหลดแล้ว'))
      return box
    }
    if (m.state === 'downloading' || m.state === 'queued') {
      const bar = el('div', 'model-bar')
      bar.style.width = '100%'
      const fill = el('span', 'progress')
      fill.style.width = `${m.state === 'queued' ? 0 : Math.min(100, m.percent || 0)}%`
      bar.append(fill)
      const line = el('div', 'muted', m.state === 'queued' ? 'รอคิว (โหลดทีละตัว)' : `${Math.floor(m.percent || 0)}% · ${gb(m.done_gb)} / ${gb(m.size_gb)}`)
      const stop = el('button', 'settings-link', 'หยุด')
      stop.type = 'button'
      stop.disabled = busy.has(m.id)
      stop.addEventListener('click', () => act('cancel', m))
      box.append(bar, line, stop)
      return box
    }
    const btn = el('button', 'btn btn-primary', `${m.state === 'partial' || (m.download && ['cancelled', 'stopped', 'error'].includes(m.download.state)) ? 'โหลดต่อ' : 'ดาวน์โหลด'}${m.size_gb ? ' ' + gb(m.size_gb) : ''}`)
    btn.type = 'button'
    btn.disabled = busy.has(m.id) || (m.size_gb != null && data && data.machine.disk_free_gb < m.size_gb + 2)
    btn.addEventListener('click', () => act('download', m))
    box.append(btn)
    if (m.state === 'partial') box.append(el('span', 'muted', 'ตัวเร่งบางแบบยังไม่ครบ'))
    if (m.download && m.download.state === 'error' && m.download.error) box.append(el('span', 'err', m.download.error))
    if (m.download && m.download.state === 'stopped') box.append(el('span', 'muted', 'หยุดกลางทาง กดโหลดต่อได้'))
    return box
  }

  function render() {
    const list = document.getElementById('model-list')
    const machine = document.getElementById('model-machine')
    if (!list || !machine) return
    if (!data) return
    const m = data.machine
    machine.replaceChildren(
      el('span', null, `เครื่องนี้: ${m.chip || 'Mac'}`),
      el('span', null, `RAM ${m.ram_gb} GB`),
      el('span', null, `ดิสก์ว่าง ${gb(m.disk_free_gb)}`),
    )
    const head = el('div', 'model-row head')
    head.append(el('div', null, 'โมเดล'), el('div', null, 'ขนาด'), el('div', null, `RAM ที่ใช้ (เครื่องนี้ ${m.ram_gb} GB)`), el('div', null, 'สถานะ'))
    head.lastChild.style.textAlign = 'right'
    const rows = data.models.map(model => {
      const row = el('div', 'model-row')
      const name = el('div')
      name.append(el('div', 'model-name', model.name), el('div', 'model-sub', model.vision ? 'อ่านภาพได้' : 'ข้อความอย่างเดียว'))
      const chips = el('div', 'model-variants')
      for (const v of model.variants) {
        const chip = el('span', `model-chip${v.ready ? ' ready' : ''}`, ACCEL[v.acceleration] || v.acceleration)
        chip.title = v.ready ? v.label : `${v.label}: ${v.reason || 'ยังไม่ได้ดาวน์โหลด'}`
        chips.append(chip)
      }
      name.append(chips)
      const f = fit(model.ram_gb, m.ram_gb)
      const ram = el('div')
      const bar = el('div', 'model-bar')
      if (model.ram_gb != null) {
        const fill = el('span', f.bar)
        fill.style.width = `${Math.min(100, (100 * model.ram_gb) / Math.max(1, m.ram_gb))}%`
        bar.append(fill)
      }
      const label = el('div', 'model-ram')
      label.append(el('span', null, model.ram_gb != null ? `~${model.ram_gb} GB · ` : ''), el('span', f.cls, f.label))
      ram.append(bar, label)
      if (model.ram_note) ram.append(el('div', 'model-sub', model.ram_note))
      row.append(name, el('div', 'model-sub', gb(model.size_gb)), ram, statusCell(model))
      return row
    })
    list.replaceChildren(head, ...rows)
  }

  // check = also ask Hugging Face for sizes not known yet: only when the page opens or the user presses Refresh
  // (never on a timer). While a download runs the page re-reads the disk every 3 s, then stops on its own.
  function pollWhileDownloading() {
    const downloading = data.models.some(m => m.state === 'downloading' || m.state === 'queued')
    if (downloading && !timer && pageOpen) timer = setInterval(() => refresh(false), 3000)
    if (!downloading && timer) {
      clearInterval(timer)
      timer = null
    }
  }

  async function refresh(check) {
    const a = api()
    if (!a) return
    if (inFlight) return
    inFlight = true
    const button = document.getElementById('model-refresh')
    if (button && check) button.disabled = true
    try {
      const r = await a.models(!!check)
      if (r && r.ok) {
        const before = downloadedKey()
        data = r
        render()
        pollWhileDownloading()
        // a model that just finished downloading must appear in Settings -> Model without a restart
        if (before !== downloadedKey() && typeof wsSend === 'function') wsSend({ type: 'get_runtime_settings' })
      } else {
        const list = document.getElementById('model-list')
        if (list) list.replaceChildren(el('div', 'settings-help', (r && r.error) || 'อ่านรายการโมเดลไม่ได้'))
      }
    } catch (e) {
      const list = document.getElementById('model-list')
      if (list) list.replaceChildren(el('div', 'settings-help', String(e)))
    } finally {
      inFlight = false
      if (button) button.disabled = false
    }
  }

  function downloadedKey() {
    return data ? data.models.filter(m => m.state === 'installed' || m.state === 'partial').map(m => m.id).sort().join('|') : ''
  }

  async function act(action, model) {
    const a = api()
    if (!a) return
    busy.add(model.id)
    render()
    try {
      const r = await a.modelAction(action, model.id)
      if (r && !r.ok && r.error) {
        model.download = { state: 'error', error: r.error }
      }
    } finally {
      busy.delete(model.id)
      await refresh(false)
    }
  }

  let pageOpen = false
  window.ThModels = {
    open() {
      refresh(true)
      pageOpen = true
    },
    close() {
      pageOpen = false
      if (timer) clearInterval(timer)
      timer = null
    },
    // ids of models whose target is downloaded; null until the first answer (then Settings shows everything)
    downloadedIds() {
      if (!data) return null
      return new Set(data.models.filter(m => m.state === 'installed' || m.state === 'partial').map(m => m.id))
    },
  }

  // learn what is downloaded once at start, so Settings filters its list even before the Model page is opened
  const start = () => {
    refresh(false)
    const button = document.getElementById('model-refresh')
    if (button) button.addEventListener('click', () => refresh(true))
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start)
  else start()
})()
