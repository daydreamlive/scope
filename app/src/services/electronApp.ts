import { app, BrowserWindow, Tray, Menu, nativeImage, nativeTheme, globalShortcut, session } from 'electron';
import path from 'path';
import fs from 'fs';
import { AppState } from '../types/services';
import { IPC_CHANNELS } from '../types/ipc';
import { SERVER_CONFIG } from '../utils/config';
import { logger, getLogPath } from '../utils/logger';

export class ScopeElectronAppService {
  private appState: AppState;
  private tray: Tray | null = null;
  private logsWindow: BrowserWindow | null = null;

  constructor(appState: AppState) {
    this.appState = appState;
    this.setupWindowsJumpList();
  }

  /**
   * Set up Windows Jump List (right-click menu on taskbar icon)
   * Note: Linux uses desktop actions in the .desktop file instead
   */
  private setupWindowsJumpList(): void {
    if (process.platform !== 'win32') return;

    app.setUserTasks([
      {
        program: process.execPath,
        arguments: '--show-logs',
        iconPath: process.execPath,
        iconIndex: 0,
        title: 'Logs',
        description: 'Open server logs',
      },
    ]);
  }

  /**
   * Check if app was launched with --show-logs argument
   */
  checkLaunchArgs(): void {
    if (process.argv.includes('--show-logs')) {
      // Delay to ensure window is ready
      setTimeout(() => {
        this.showLogsWindow();
      }, 1000);
    }
  }

  createMainWindow(): BrowserWindow {
    // Force dark mode for native UI elements (title bar, scrollbars, etc.)
    nativeTheme.themeSource = 'dark';

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
      darkTheme: true, // Force dark theme for window frame (Linux/Windows)
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : (process.platform === 'win32' ? 'hidden' : 'default'),
      titleBarOverlay: process.platform === 'win32' ? {
        color: '#0f0f0f',
        symbolColor: '#ffffff',
        height: 32,
      } : undefined,
      autoHideMenuBar: true, // Hide menu bar (can still access with Alt key)
      webPreferences: {
        preload: preloadPath,
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true, // Enable sandboxing for security
        webSecurity: true,
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
      const indexPath = path.join(process.resourcesPath, 'app', '.vite', 'build', 'renderer', 'index.html');
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

      // Security: Disable DevTools in production
      if (app.isPackaged) {
        mainWindow.webContents.closeDevTools();

        // Prevent DevTools from being opened in production
        mainWindow.webContents.on('devtools-opened', () => {
          logger.warn('DevTools attempted to open in production - closing');
          mainWindow.webContents.closeDevTools();
        });
      }
    });

    // Add event listeners for debugging
    mainWindow.webContents.on('did-start-loading', () => {
      logger.info('Window started loading');
    });

    // Inject CSS to hide scrollbar as early as possible (before page renders)
    mainWindow.webContents.on('dom-ready', () => {
      const currentUrl = mainWindow.webContents.getURL();
      // Only hide scrollbar on loading/setup screens (not on the Python server frontend)
      if (!currentUrl.includes(SERVER_CONFIG.host) && !currentUrl.includes('127.0.0.1:') && !currentUrl.includes('localhost:')) {
        mainWindow.webContents.insertCSS(`
          ::-webkit-scrollbar {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
          }
          html, body, #root {
            overflow: hidden !important;
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
          }
        `);
      }
    });

    mainWindow.webContents.on('did-finish-load', () => {
      logger.info('Window finished loading');
    });

    mainWindow.webContents.on('did-fail-load', (_event, errorCode, errorDescription) => {
      logger.error(`Window failed to load: ${errorCode} - ${errorDescription}`);
    });

    mainWindow.on('closed', () => {
      this.appState.mainWindow = null;
    });

    // Add right-click context menu for the title bar area
    // This creates an app-level menu that appears on right-click in the title bar region
    this.setupWindowContextMenu(mainWindow);

    this.appState.mainWindow = mainWindow;
    return mainWindow;
  }

  private setupWindowContextMenu(_window: BrowserWindow): void {
    // Set up the application menu with context menu items (including keyboard shortcuts)
    const menu = Menu.buildFromTemplate([
      {
        label: 'File',
        submenu: [
          {
            label: 'Logs',
            accelerator: 'CmdOrCtrl+Shift+L',
            click: () => {
              this.showLogsWindow();
            },
          },
          { type: 'separator' },
          {
            label: 'Quit',
            accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Alt+F4',
            click: () => {
              app.quit();
            },
          },
        ],
      },
    ]);

    // Set this as the application menu (accessible via Alt key)
    Menu.setApplicationMenu(menu);

    // Register global shortcut for logs (works even when menu is hidden)
    globalShortcut.register('CmdOrCtrl+Shift+L', () => {
      this.showLogsWindow();
    });
  }

  loadFrontend(): void {
    if (this.appState.mainWindow && !this.appState.mainWindow.isDestroyed()) {
      try {
        // Once server is running, load the actual frontend from Python server
        this.appState.mainWindow.loadURL(SERVER_CONFIG.url);

        // Inject dark scrollbar styling and draggable title bar once the page loads
        this.appState.mainWindow.webContents.once('did-finish-load', () => {
          if (this.appState.mainWindow && !this.appState.mainWindow.isDestroyed()) {
            this.appState.mainWindow.webContents.insertCSS(`
              ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
              }
              ::-webkit-scrollbar-track {
                background: hsl(0, 0%, 10%);
              }
              ::-webkit-scrollbar-thumb {
                background: hsl(0, 0%, 25%);
                border-radius: 5px;
              }
              ::-webkit-scrollbar-thumb:hover {
                background: hsl(0, 0%, 35%);
              }
              ::-webkit-scrollbar-corner {
                background: hsl(0, 0%, 10%);
              }
              /* Draggable title bar region for Windows/macOS with hidden title bar */
              body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 32px;
                -webkit-app-region: drag;
                z-index: 10000;
                pointer-events: auto;
              }
              /* Make interactive elements non-draggable */
              button, input, textarea, select, a,
              [role="button"], [role="textbox"], [role="combobox"],
              [contenteditable="true"] {
                -webkit-app-region: no-drag;
              }
            `);
          }
        });
      } catch (err) {
        logger.error('Failed to load frontend:', err);
      }
    }
  }

  createTray(): void {
    // Create a simple tray icon
    let iconPath: string;
    const isWindows = process.platform === 'win32';

    if (app.isPackaged) {
      const assetsPath = path.join(process.resourcesPath, 'app', 'assets');
      if (isWindows) {
        // Windows: prefer .ico for tray icons
        iconPath = path.join(assetsPath, 'icon.ico');
      } else {
        // macOS/Linux: use PNG
        const trayIconPath = path.join(assetsPath, 'tray-icon.png');
        const fallbackIconPath = path.join(assetsPath, 'icon.png');
        iconPath = fs.existsSync(trayIconPath) ? trayIconPath : fallbackIconPath;
      }
    } else {
      const assetsPath = path.join(__dirname, '../../assets');
      if (isWindows) {
        // Windows: prefer .ico for tray icons
        iconPath = path.join(assetsPath, 'icon.ico');
      } else {
        // macOS/Linux: use PNG
        const trayIconPath = path.join(assetsPath, 'tray-icon.png');
        const fallbackIconPath = path.join(assetsPath, 'icon.png');
        iconPath = fs.existsSync(trayIconPath) ? trayIconPath : fallbackIconPath;
      }
    }

    logger.info(`Creating tray with icon: ${iconPath}, exists: ${fs.existsSync(iconPath)}`);
    const icon = nativeImage.createFromPath(iconPath);
    logger.info(`Tray icon isEmpty: ${icon.isEmpty()}, size: ${icon.getSize().width}x${icon.getSize().height}`);
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
        label: 'Logs',
        click: () => {
          this.showLogsWindow();
        },
      },
      { type: 'separator' },
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
    // Unregister global shortcuts
    globalShortcut.unregisterAll();

    if (this.tray) {
      this.tray.destroy();
      this.tray = null;
    }
    if (this.logsWindow) {
      this.logsWindow.close();
      this.logsWindow = null;
    }
  }

  showContextMenu(): void {
    const contextMenu = Menu.buildFromTemplate([
      {
        label: 'Logs',
        click: () => {
          this.showLogsWindow();
        },
      },
      { type: 'separator' },
      {
        label: 'Quit',
        click: () => {
          app.quit();
        },
      },
    ]);

    contextMenu.popup();
  }

  showLogsWindow(): void {
    // If logs window already exists, focus it
    if (this.logsWindow && !this.logsWindow.isDestroyed()) {
      this.logsWindow.focus();
      return;
    }

    const logPath = getLogPath();

    // Create isolated session for logs window (session partitioning)
    const logsSession = session.fromPartition('persist:logs-viewer');

    // Create logs viewer window with isolated session
    this.logsWindow = new BrowserWindow({
      width: 900,
      height: 600,
      title: 'Server Logs',
      backgroundColor: '#1a1a1a',
      autoHideMenuBar: true,
      webPreferences: {
        session: logsSession, // Use isolated session
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
      },
    });

    // Security: Disable DevTools in production for logs window
    if (app.isPackaged) {
      this.logsWindow.webContents.closeDevTools();
      this.logsWindow.webContents.on('devtools-opened', () => {
        logger.warn('DevTools attempted to open in logs window (production) - closing');
        this.logsWindow?.webContents.closeDevTools();
      });
    }

    // Read log file and display it
    let logContent = '';
    try {
      if (fs.existsSync(logPath)) {
        logContent = fs.readFileSync(logPath, 'utf-8');
        // Get the last 500 lines for better performance
        const lines = logContent.split('\n');
        if (lines.length > 500) {
          logContent = lines.slice(-500).join('\n');
        }
      } else {
        logContent = 'No logs found.';
      }
    } catch (err) {
      logContent = `Error reading logs: ${err}`;
    }

    // Load the log viewer HTML file with query parameters
    // This is more secure than using data: URLs
    let logViewerPath: string;
    if (app.isPackaged) {
      logViewerPath = path.join(process.resourcesPath, 'app', '.vite', 'build', 'renderer', 'LogViewer.html');
    } else {
      logViewerPath = path.join(__dirname, '../../src/components/LogViewer.html');
    }

    // Pass log content and path as URL parameters
    // Note: loadFile's query option handles URL encoding automatically
    this.logsWindow.loadFile(logViewerPath, {
      query: {
        content: logContent,
        path: logPath,
      },
    }).catch((err) => {
      logger.error('Failed to load log viewer:', err);
    });

    this.logsWindow.on('closed', () => {
      this.logsWindow = null;
    });
  }

  getLogs(): string {
    const logPath = getLogPath();
    try {
      if (fs.existsSync(logPath)) {
        const content = fs.readFileSync(logPath, 'utf-8');
        // Return last 500 lines
        const lines = content.split('\n');
        if (lines.length > 500) {
          return lines.slice(-500).join('\n');
        }
        return content;
      }
      return 'No logs found.';
    } catch (err) {
      return `Error reading logs: ${err}`;
    }
  }
}
