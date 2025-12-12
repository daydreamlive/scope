// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts

import { contextBridge, ipcRenderer } from 'electron';
import { IPC_CHANNELS } from './types/ipc';

/**
 * Expose protected methods that allow the renderer process to use
 * the ipcRenderer without exposing the entire object
 */
contextBridge.exposeInMainWorld('scope', {
  onSetupStatus: (callback: (status: string) => void) => {
    // Validate callback
    if (typeof callback !== 'function') {
      throw new Error('Callback must be a function');
    }

    const handler = (_event: Electron.IpcRendererEvent, status: string) => {
      // Validate status is a string
      if (typeof status !== 'string') {
        console.error('Invalid setup status type:', typeof status);
        return;
      }
      callback(status);
    };

    ipcRenderer.on(IPC_CHANNELS.SETUP_STATUS, handler);

    // Return cleanup function
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.SETUP_STATUS, handler);
    };
  },

  onServerStatus: (callback: (isRunning: boolean) => void) => {
    // Validate callback
    if (typeof callback !== 'function') {
      throw new Error('Callback must be a function');
    }

    const handler = (_event: Electron.IpcRendererEvent, isRunning: boolean) => {
      // Validate isRunning is a boolean
      if (typeof isRunning !== 'boolean') {
        console.error('Invalid server status type:', typeof isRunning);
        return;
      }
      callback(isRunning);
    };

    ipcRenderer.on(IPC_CHANNELS.SERVER_STATUS, handler);

    // Return cleanup function
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.SERVER_STATUS, handler);
    };
  },

  onServerError: (callback: (error: string) => void) => {
    // Validate callback
    if (typeof callback !== 'function') {
      throw new Error('Callback must be a function');
    }

    const handler = (_event: Electron.IpcRendererEvent, error: string) => {
      // Validate error is a string
      if (typeof error !== 'string') {
        console.error('Invalid server error type:', typeof error);
        return;
      }
      callback(error);
    };

    ipcRenderer.on(IPC_CHANNELS.SERVER_ERROR, handler);

    // Return cleanup function
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.SERVER_ERROR, handler);
    };
  },

  getSetupState: () => ipcRenderer.invoke(IPC_CHANNELS.GET_SETUP_STATE),

  getSetupStatus: () => ipcRenderer.invoke(IPC_CHANNELS.GET_SETUP_STATUS),

  getServerStatus: () => ipcRenderer.invoke(IPC_CHANNELS.GET_SERVER_STATUS),

  showContextMenu: () => ipcRenderer.invoke(IPC_CHANNELS.SHOW_CONTEXT_MENU),

  getLogs: () => ipcRenderer.invoke(IPC_CHANNELS.GET_LOGS),
});
