import { app, BrowserWindow, ipcMain, Tray, Menu, nativeImage } from 'electron';
import path from 'path';
import fs from 'fs';
import { AppState } from '../types/services';
import { IPC_CHANNELS, SETUP_STATUS } from '../types/ipc';
import { getPaths, SERVER_CONFIG } from '../utils/config';
import { logger } from '../utils/logger';

export class ScopeElectronAppService {
  private appState: AppState;
  private tray: Tray | null = null;

  constructor(appState: AppState) {
    this.appState = appState;
  }

  createMainWindow(): BrowserWindow {
    const paths = getPaths();

    // Determine preload path
    // With Electron Forge + Vite, preload.js is built to the same directory as main.js
    // In both dev and production, __dirname points to .vite/build/
    const preloadPath = path.join(__dirname, 'preload.js');

    // Determine icon path
    let iconPath: string;
    if (app.isPackaged) {
      iconPath = path.join(process.resourcesPath, 'app', 'assets', 'icon.png');
    } else {
      iconPath = path.join(__dirname, '../../assets/icon.png');
    }

    const icon = nativeImage.createFromPath(iconPath);
    const windowIcon = icon.isEmpty() ? undefined : icon;

    const mainWindow = new BrowserWindow({
      width: 1400,
      height: 900,
      title: 'Daydream Scope',
      icon: windowIcon,
      backgroundColor: '#0f0f0f', // Dark background to match theme
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default', // Dark title bar on macOS
      webPreferences: {
        preload: preloadPath,
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: false, // Disabled because we need Node.js APIs in preload for IPC
        webSecurity: true,
        allowRunningInsecureContent: false,
        experimentalFeatures: false,
      },
      show: false, // Don't show until ready
    });

    // Security: Prevent navigation to external URLs
    mainWindow.webContents.on('will-navigate', (event, navigationUrl) => {
      try {
        const parsedUrl = new URL(navigationUrl);
        const allowedHosts = ['127.0.0.1', 'localhost', SERVER_CONFIG.host];

        if (!allowedHosts.includes(parsedUrl.hostname)) {
          event.preventDefault();
          logger.warn(`Blocked navigation to external URL: ${navigationUrl}`);
        }
      } catch (err) {
        // Invalid URL, prevent navigation
        event.preventDefault();
        logger.warn(`Blocked navigation to invalid URL: ${navigationUrl}`);
      }
    });

    // Security: Prevent new window creation (open external links)
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
      try {
        const parsedUrl = new URL(url);
        const allowedHosts = ['127.0.0.1', 'localhost', SERVER_CONFIG.host];

        if (!allowedHosts.includes(parsedUrl.hostname)) {
          logger.warn(`Blocked new window to external URL: ${url}`);
          return { action: 'deny' };
        }

        return { action: 'allow' };
      } catch (err) {
        // Invalid URL, deny
        logger.warn(`Blocked new window to invalid URL: ${url}`);
        return { action: 'deny' };
      }
    });

    // Load the Electron renderer (setup screen) initially
    // The main process will check server status and load frontend if needed
    if (app.isPackaged) {
      // In production, load from the built renderer
      const indexPath = path.join(process.resourcesPath, 'app', 'index.html');
      logger.info(`Loading from file: ${indexPath}`);
      mainWindow.loadFile(indexPath);
    } else {
      // In development, load from Vite dev server for renderer
      const devUrl = 'http://localhost:5173';
      logger.info(`Loading from dev server: ${devUrl}`);
      mainWindow.loadURL(devUrl).catch((err) => {
        logger.error(`Failed to load ${devUrl}:`, err);
      });
    }

    mainWindow.once('ready-to-show', () => {
      logger.info('Window ready to show');
      mainWindow.show();
    });

    // Add event listeners for debugging
    mainWindow.webContents.on('did-start-loading', () => {
      logger.info('Window started loading');
    });

    mainWindow.webContents.on('did-finish-load', () => {
      logger.info('Window finished loading');
    });

    mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
      logger.error(`Window failed to load: ${errorCode} - ${errorDescription}`);
    });

    mainWindow.on('closed', () => {
      this.appState.mainWindow = null;
    });

    this.appState.mainWindow = mainWindow;
    return mainWindow;
  }

  loadFrontend(): void {
    if (this.appState.mainWindow && !this.appState.mainWindow.isDestroyed()) {
      try {
        // Once server is running, load the actual frontend from Python server
        this.appState.mainWindow.loadURL(SERVER_CONFIG.url);
      } catch (err) {
        logger.error('Failed to load frontend:', err);
      }
    }
  }

  createTray(): void {
    // Create a simple tray icon
    let iconPath: string;
    if (app.isPackaged) {
      // Try tray-icon.png first, fall back to icon.png
      const trayIconPath = path.join(process.resourcesPath, 'app', 'assets', 'tray-icon.png');
      const fallbackIconPath = path.join(process.resourcesPath, 'app', 'assets', 'icon.png');
      iconPath = fs.existsSync(trayIconPath) ? trayIconPath : fallbackIconPath;
    } else {
      // Try tray-icon.png first, fall back to icon.png
      const trayIconPath = path.join(__dirname, '../../assets/tray-icon.png');
      const fallbackIconPath = path.join(__dirname, '../../assets/icon.png');
      iconPath = fs.existsSync(trayIconPath) ? trayIconPath : fallbackIconPath;
    }

    const icon = nativeImage.createFromPath(iconPath);
    this.tray = new Tray(icon.isEmpty() ? nativeImage.createEmpty() : icon);

    const contextMenu = Menu.buildFromTemplate([
      {
        label: 'Show Window',
        click: () => {
          if (this.appState.mainWindow) {
            this.appState.mainWindow.show();
          }
        },
      },
      {
        label: 'Quit',
        click: () => {
          app.quit();
        },
      },
    ]);

    this.tray.setToolTip('Daydream Scope');
    this.tray.setContextMenu(contextMenu);

    this.tray.on('click', () => {
      if (this.appState.mainWindow) {
        this.appState.mainWindow.show();
      }
    });
  }

  sendSetupStatus(status: string): void {
    if (this.appState.mainWindow && !this.appState.mainWindow.isDestroyed()) {
      try {
        this.appState.mainWindow.webContents.send(IPC_CHANNELS.SETUP_STATUS, status);
      } catch (err) {
        logger.error('Failed to send setup status:', err);
      }
    }
  }

  sendServerStatus(isRunning: boolean): void {
    if (this.appState.mainWindow && !this.appState.mainWindow.isDestroyed()) {
      try {
        this.appState.mainWindow.webContents.send(IPC_CHANNELS.SERVER_STATUS, isRunning);
      } catch (err) {
        logger.error('Failed to send server status:', err);
      }
    }
  }

  sendServerError(error: string): void {
    if (this.appState.mainWindow && !this.appState.mainWindow.isDestroyed()) {
      try {
        this.appState.mainWindow.webContents.send(IPC_CHANNELS.SERVER_ERROR, error);
      } catch (err) {
        logger.error('Failed to send server error:', err);
      }
    }
  }

  waitForMainWindowLoad(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.appState.mainWindow) {
        resolve();
        return;
      }

      // Add timeout to prevent hanging forever
      const timeout = setTimeout(() => {
        logger.warn('Window load timeout, proceeding anyway');
        resolve();
      }, 10000); // 10 second timeout

      // Check if already loaded
      if (this.appState.mainWindow.webContents.isLoading() === false) {
        clearTimeout(timeout);
        resolve();
        return;
      }

      this.appState.mainWindow.webContents.once('did-finish-load', () => {
        clearTimeout(timeout);
        resolve();
      });

      // Also handle navigation failures
      this.appState.mainWindow.webContents.once('did-fail-load', () => {
        clearTimeout(timeout);
        logger.warn('Window failed to load, proceeding anyway');
        resolve();
      });
    });
  }

  cleanup(): void {
    if (this.tray) {
      this.tray.destroy();
      this.tray = null;
    }
  }
}
