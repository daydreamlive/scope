const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Server status
  checkServerHealth: () => ipcRenderer.invoke('check-server-health'),
  onServerReady: (callback) => {
    ipcRenderer.on('server-ready', callback);
    return () => ipcRenderer.removeAllListeners('server-ready');
  },

  // Navigation
  onNavigateToSettings: (callback) => {
    ipcRenderer.on('navigate-to-settings', callback);
    return () => ipcRenderer.removeAllListeners('navigate-to-settings');
  },

  // Logs
  getLogsDir: () => ipcRenderer.invoke('get-logs-dir'),

  // Platform info
  platform: process.platform,
  isElectron: true
});
