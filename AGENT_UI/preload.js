const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  onStartupStatus: (cb) => ipcRenderer.on('startup-status', (_, data) => cb(data)),
  onStartupDone: (cb) => ipcRenderer.once('startup-done', () => cb()),
  openWorkspace: () => ipcRenderer.send('open-workspace'),
  getToken: () => ipcRenderer.invoke('get-token'),
  requestExit: () => ipcRenderer.send('request-exit'),
  editFile: (p) => ipcRenderer.invoke('edit-file', p),
  deleteFile: (p) => ipcRenderer.invoke('delete-file', p),
  showOpenDialog: () => ipcRenderer.invoke('show-open-dialog'),
  setCompact: (compact) => ipcRenderer.send('set-compact', compact),
  platform: process.platform,
})
