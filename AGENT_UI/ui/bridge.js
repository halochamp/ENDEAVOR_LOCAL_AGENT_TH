// bridge.js — the renderer's `window.electronAPI` (the Agent TH preload contract), served by the Rust core.
//
// renderer.js and lib/*.js are the proven Agent TH code; keeping this one object identical means every feature keeps
// working while only the shell changes (Electron -> Tauri 2). Each method maps 1:1 to a #[tauri::command] in
// src-tauri/src/commands.rs; argument clamping happens there, exactly as the old main.js did.
;(function () {
  var T = window.__TAURI__
  if (!T || !T.core) return // opened outside the app (tests, a browser): renderer.js already tolerates a missing API
  var invoke = T.core.invoke
  var listen = T.event.listen
  var id = function (jobId) { return String(jobId || '') }

  window.electronAPI = {
    onStartupStatus: function (cb) { listen('startup-status', function (e) { cb(e.payload) }) },
    onStartupDone: function (cb) {
      var done = false
      listen('startup-done', function () { if (!done) { done = true; cb() } })
      // startup may already be over when this page (re)loads
      invoke('startup_state').then(function (s) { if (s && s.done && !done) { done = true; cb() } })
    },
    openWorkspace: function () { invoke('open_workspace') },
    getToken: function () { return invoke('get_token') },
    setRuntimeSettings: function (payload) { return invoke('set_runtime_settings', { payload: payload || {} }) },
    requestExit: function () { invoke('request_exit') },
    confirm: function (message) { return invoke('confirm_dialog', { message: String(message || '') }) },
    editFile: function (p) { return invoke('edit_file', { path: String(p || '') }) },
    deleteFile: function (p) { return invoke('delete_file', { path: String(p || '') }) },
    showOpenDialog: function () { return invoke('show_open_dialog') },
    showPinDialog: function () { return invoke('show_pin_dialog') },
    showApprovedEditFolderDialog: function () { return invoke('show_approved_edit_folder_dialog') },
    pdfToTextStart: function (rewriteThai) { return invoke('pdf_to_text_start', { rewriteThai: !!rewriteThai }) },
    pdfToTextStatus: function (jobId) { return invoke('agent_job', { route: '/pdf-to-text/status', jobId: id(jobId), post: false }) },
    translateStart: function () { return invoke('translate_start') },
    translateStatus: function (jobId) { return invoke('agent_job', { route: '/translate/status', jobId: id(jobId), post: false }) },
    translatePause: function (jobId) { return invoke('agent_job', { route: '/translate/pause', jobId: id(jobId), post: true }) },
    translateResume: function (jobId) { return invoke('agent_job', { route: '/translate/resume', jobId: id(jobId), post: true }) },
    translateCancel: function (jobId) { return invoke('agent_job', { route: '/translate/cancel', jobId: id(jobId), post: true }) },
    ttsVoices: function () { return invoke('agent_get', { route: '/tts/voices' }) },
    ttsStart: function (voice, speed) { return invoke('tts_start', { voice: String(voice || ''), speed: Number(speed || 1) }) },
    ttsStatus: function (jobId) { return invoke('agent_job', { route: '/tts/status', jobId: id(jobId), post: false }) },
    ttsPause: function (jobId) { return invoke('agent_job', { route: '/tts/pause', jobId: id(jobId), post: true }) },
    ttsResume: function (jobId) { return invoke('agent_job', { route: '/tts/resume', jobId: id(jobId), post: true }) },
    ttsCancel: function (jobId) { return invoke('agent_job', { route: '/tts/cancel', jobId: id(jobId), post: true }) },
    setCompact: function (compact) { invoke('set_compact', { compact: !!compact }) },
    platform: 'darwin',
    // Model page (new in the Rust UI)
    models: function (check) { return invoke('models_list', { check: !!check }) },
    modelAction: function (action, modelId) { return invoke('model_action', { action: String(action), model: String(modelId || '') }) },
  }

  // Web links in answers (target=_blank, sources lists) open in the browser, never inside the app window.
  document.addEventListener('click', function (e) {
    var a = e.target && e.target.closest ? e.target.closest('a[href]') : null
    if (!a) return
    var href = a.getAttribute('href') || ''
    if (/^https?:\/\//i.test(href)) {
      e.preventDefault()
      invoke('open_url', { url: href })
    }
  }, true)
})()
