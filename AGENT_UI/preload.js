const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  onStartupStatus: (cb) => ipcRenderer.on('startup-status', (_, data) => cb(data)),
  onStartupDone: (cb) => ipcRenderer.once('startup-done', () => cb()),
  openWorkspace: () => ipcRenderer.send('open-workspace'),
  getToken: () => ipcRenderer.invoke('get-token'),
  setRuntimeSettings: (payload) => ipcRenderer.invoke('set-runtime-settings', payload),
  requestExit: () => ipcRenderer.send('request-exit'),
  editFile: (p) => ipcRenderer.invoke('edit-file', p),
  deleteFile: (p) => ipcRenderer.invoke('delete-file', p),
  showOpenDialog: () => ipcRenderer.invoke('show-open-dialog'),
  showPinDialog: () => ipcRenderer.invoke('show-pin-dialog'),
  showApprovedEditFolderDialog: () => ipcRenderer.invoke('show-approved-edit-folder-dialog'),
  pdfToTextStart: (rewriteThai) => ipcRenderer.invoke('pdf-to-text-start', !!rewriteThai),
  pdfToTextStatus: (jobId) => ipcRenderer.invoke('pdf-to-text-status', String(jobId || '')),
  translateStart: () => ipcRenderer.invoke('translate-start'),
  translateStatus: (jobId) => ipcRenderer.invoke('translate-status', String(jobId || '')),
  translatePause: (jobId) => ipcRenderer.invoke('translate-pause', String(jobId || '')),
  translateResume: (jobId) => ipcRenderer.invoke('translate-resume', String(jobId || '')),
  translateCancel: (jobId) => ipcRenderer.invoke('translate-cancel', String(jobId || '')),
  setCompact: (compact) => ipcRenderer.send('set-compact', compact),
  platform: process.platform,
})
