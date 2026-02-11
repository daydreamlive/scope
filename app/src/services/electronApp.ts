import { app, BrowserWindow, Tray, Menu, nativeImage, nativeTheme, globalShortcut, session, shell } from 'electron';
import path from 'path';
import fs from 'fs';
import { AppState } from '../types/services';
import { IPC_CHANNELS } from '../types/ipc';
import { SERVER_CONFIG } from '../utils/config';
import { logger, getLogPath } from '../utils/logger';

// Constants
const WINDOW_CONFIG = {
  MAIN: { width: 1400, height: 900 },
  LOGS: { width: 900, height: 600 },
} as const;

const TIMEOUTS = {
  LAUNCH_ARGS_DELAY: 1000,
  WINDOW_LOAD: 30000,
} as const;

const COLORS = {
  BACKGROUND: '#0f0f0f',
  LOGS_BACKGROUND: '#1a1a1a',
} as const;

const LOG_LINE_LIMIT = 500;

const ALLOWED_HOSTS = ['127.0.0.1', 'localhost', SERVER_CONFIG.host];

export class ScopeElectronAppService {
  private appState: AppState;
  private tray: Tray | null = null;
  private logsWindow: BrowserWindow | null = null;
  private logsRefreshInterval: NodeJS.Timeout | null = null;

  constructor(appState: AppState) {
    this.appState = appState;
    this.setupWindowsJumpList();
  }

  // Helper methods
  private isAllowedUrl(url: string): boolean {
    try {
      const parsedUrl = new URL(url);
      return ALLOWED_HOSTS.includes(parsedUrl.hostname);
    } catch {
      return false;
    }
  }

  private getIconPath(iconName: string = 'icon.png'): string {
    if (app.isPackaged) {
      return path.join(process.resourcesPath, 'app', 'assets', iconName);
    }
    return path.join(__dirname, '../../assets', iconName);
  }

  private getTrayIconPath(): string {
    const isWindows = process.platform === 'win32';
    if (isWindows) {
      return this.getIconPath('icon.ico');
    }
    const trayIconPath = this.getIconPath('tray-icon.png');
    const fallbackIconPath = this.getIconPath('icon.png');
    return fs.existsSync(trayIconPath) ? trayIconPath : fallbackIconPath;
  }

  private readLogFile(maxLines: number = LOG_LINE_LIMIT): string {
    const logPath = getLogPath();
    try {
      if (fs.existsSync(logPath)) {
        const content = fs.readFileSync(logPath, 'utf-8');
        const lines = content.split('\n');
        if (lines.length > maxLines) {
          return lines.slice(-maxLines).join('\n');
        }
        return content;
      }
      return 'No logs found.';
    } catch (err) {
      return `Error reading logs: ${err}`;
    }
  }

  private setupDevToolsSecurity(window: BrowserWindow): void {
    if (!app.isPackaged) return;

    window.webContents.closeDevTools();
    window.webContents.on('devtools-opened', () => {
      logger.warn('DevTools attempted to open in production - closing');
      window.webContents.closeDevTools();
    });
  }

  /**
   * Injects draggable title bar CSS for the Python server frontend.
   * Required because the window has a hidden title bar on Windows/macOS,
   * so users need a draggable region to move the window.
   *
   * Note: Scrollbar styling is handled by the frontend itself.
   * Uses !important to prevent Radix UI portals/scroll-locks from overriding styles.
   */
  private injectDraggableTitleBarCSS(window: BrowserWindow): void {
    const css = `
      /* Draggable title bar region for Windows/macOS with hidden title bar */
      body::before {
        content: '' !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 32px !important;
        -webkit-app-region: drag;
        z-index: 10000 !important;
        pointer-events: auto !important;
      }
      /* Add padding to body to prevent content from overlapping with title bar */
      /* Prevent overflow by ensuring html and body don't exceed viewport height */
      /* Uses !important to prevent Radix UI scroll-lock from overriding padding */
      html {
        height: 100% !important;
        overflow: hidden !important;
      }
      body {
        padding-top: 32px !important;
        height: 100% !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
        position: relative !important;
        margin-right: 0 !important;
      }
      /* Adjust #root to account for the title bar padding */
      #root {
        height: 100% !important;
        max-height: 100% !important;
        overflow: hidden !important;
      }
      /* Adjust the main content container to fit within available space */
      #root > div {
        height: 100% !important;
        max-height: 100% !important;
      }
      /* Make interactive elements non-draggable */
      button, input, textarea, select, a,
      [role="button"], [role="textbox"], [role="combobox"],
      [contenteditable="true"] {
        -webkit-app-region: no-drag;
      }
    `;
    window.webContents.insertCSS(css);
  }

  private sendIPC(channel: string, data: unknown): void {
    if (this.appState.mainWindow && !this.appState.mainWindow.isDestroyed()) {
      try {
        this.appState.mainWindow.webContents.send(channel, data);
      } catch (err) {
        logger.error(`Failed to send ${channel}:`, err);
      }
    }
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

  checkLaunchArgs(): void {
    if (process.argv.includes('--show-logs')) {
      setTimeout(() => {
        this.showLogsWindow();
      }, TIMEOUTS.LAUNCH_ARGS_DELAY);
    }
  }

  createMainWindow(): BrowserWindow {
    // Force dark mode for native UI elements (title bar, scrollbars, etc.)
    nativeTheme.themeSource = 'dark';

    const preloadPath = path.join(__dirname, 'preload.js');
    const icon = nativeImage.createFromPath(this.getIconPath());
    const windowIcon = icon.isEmpty() ? undefined : icon;

    const mainWindow = new BrowserWindow({
      width: WINDOW_CONFIG.MAIN.width,
      height: WINDOW_CONFIG.MAIN.height,
      title: 'Daydream Scope',
      icon: windowIcon,
      backgroundColor: COLORS.BACKGROUND,
      darkTheme: true,
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : (process.platform === 'win32' ? 'hidden' : 'default'),
      titleBarOverlay: process.platform === 'win32' ? {
        color: COLORS.BACKGROUND,
        symbolColor: '#ffffff',
        height: 32,
      } : undefined,
      autoHideMenuBar: true,
      webPreferences: {
        preload: preloadPath,
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        webSecurity: true,
        experimentalFeatures: false,
      },
      show: false,
    });

    // Security: Prevent navigation to external URLs
    mainWindow.webContents.on('will-navigate', (event, navigationUrl) => {
      if (!this.isAllowedUrl(navigationUrl)) {
        event.preventDefault();
        logger.warn(`Blocked navigation to external URL: ${navigationUrl}`);
      }
    });

    // Security: Open external links in default browser
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
      if (!this.isAllowedUrl(url)) {
        // Open external URLs in the default browser
        shell.openExternal(url).catch((err) => {
          logger.error(`Failed to open external URL: ${url}`, err);
        });
        return { action: 'deny' };
      }
      return { action: 'allow' };
    });

    // Load the Electron renderer (setup screen) initially
    if (app.isPackaged) {
      const indexPath = path.join(process.resourcesPath, 'app', '.vite', 'build', 'renderer', 'index.html');
      logger.info(`Loading from file: ${indexPath}`);
      mainWindow.loadFile(indexPath);
    } else {
      const devUrl = 'http://localhost:5173';
      logger.info(`Loading from dev server: ${devUrl}`);
      mainWindow.loadURL(devUrl).catch((err) => {
        logger.error(`Failed to load ${devUrl}:`, err);
      });
    }

    mainWindow.once('ready-to-show', () => {
      logger.info('Window ready to show');
      // Ensure window is visible, not minimized, and focused
      if (mainWindow.isMinimized()) {
        mainWindow.restore();
      }
      mainWindow.show();
      mainWindow.focus();
      this.setupDevToolsSecurity(mainWindow);
    });

    mainWindow.webContents.on('did-start-loading', () => {
      logger.info('Window started loading');
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
      {
        label: 'Edit',
        submenu: [
          { role: 'undo' },
          { role: 'redo' },
          { type: 'separator' },
          { role: 'cut' },
          { role: 'copy' },
          { role: 'paste' },
          { role: 'selectAll' },
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
        this.appState.mainWindow.loadURL(SERVER_CONFIG.url);
        this.appState.mainWindow.webContents.once('did-finish-load', () => {
          if (this.appState.mainWindow && !this.appState.mainWindow.isDestroyed()) {
            // Inject draggable title bar CSS (required for hidden title bar)
            this.injectDraggableTitleBarCSS(this.appState.mainWindow);
          }
        });
      } catch (err) {
        logger.error('Failed to load frontend:', err);
      }
    }
  }

  createTray(): void {
    const iconPath = this.getTrayIconPath();
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
    this.sendIPC(IPC_CHANNELS.SETUP_STATUS, status);
  }

  sendServerStatus(isRunning: boolean): void {
    this.sendIPC(IPC_CHANNELS.SERVER_STATUS, isRunning);
  }

  sendServerError(error: string): void {
    this.sendIPC(IPC_CHANNELS.SERVER_ERROR, error);
  }

  sendDeepLinkAction(data: { action: string; package: string }): void {
    this.sendIPC(IPC_CHANNELS.DEEP_LINK_ACTION, data);
  }

  sendAuthCallback(data: { token: string; userId: string | null; state: string | null }): void {
    this.sendIPC(IPC_CHANNELS.AUTH_CALLBACK, data);
  }

  waitForMainWindowLoad(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.appState.mainWindow) {
        resolve();
        return;
      }

      const timeout = setTimeout(() => {
        logger.warn('Window load timeout, proceeding anyway');
        resolve();
      }, TIMEOUTS.WINDOW_LOAD);

      if (this.appState.mainWindow.webContents.isLoading() === false) {
        clearTimeout(timeout);
        resolve();
        return;
      }

      this.appState.mainWindow.webContents.once('did-finish-load', () => {
        clearTimeout(timeout);
        resolve();
      });

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
    if (this.logsRefreshInterval) {
      clearInterval(this.logsRefreshInterval);
      this.logsRefreshInterval = null;
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

  private updateLogsContent(): void {
    if (!this.logsWindow || this.logsWindow.isDestroyed()) {
      return;
    }

    const logPath = getLogPath();
    const logContent = this.readLogFile();

    // Escape the content for safe injection into JavaScript
    const escapedPath = JSON.stringify(logPath);
    const escapedContent = JSON.stringify(logContent);

    // Update the logs content via executeJavaScript instead of reloading the page
    // This avoids URL length limits when log content is large
    this.logsWindow.webContents.executeJavaScript(
      `if (window.updateLogs) { window.updateLogs(${escapedPath}, ${escapedContent}); }`
    ).catch(() => {
      // Silently ignore errors - expected when window is closed or page is still loading
    });
  }

  showLogsWindow(): void {
    if (this.logsWindow && !this.logsWindow.isDestroyed()) {
      this.logsWindow.focus();
      return;
    }

    const logsSession = session.fromPartition('persist:logs-viewer');
    this.logsWindow = new BrowserWindow({
      width: WINDOW_CONFIG.LOGS.width,
      height: WINDOW_CONFIG.LOGS.height,
      title: 'Server Logs',
      backgroundColor: COLORS.LOGS_BACKGROUND,
      autoHideMenuBar: true,
      webPreferences: {
        session: logsSession,
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
      },
    });

    this.setupDevToolsSecurity(this.logsWindow);

    // Load the log viewer HTML file (without query params to avoid URL length limits)
    const logViewerPath = app.isPackaged
      ? path.join(process.resourcesPath, 'app', '.vite', 'build', 'renderer', 'LogViewer.html')
      : path.join(__dirname, '../../src/components/LogViewer.html');

    this.logsWindow.loadFile(logViewerPath).catch((err) => {
      logger.error('Failed to load log viewer:', err);
    });

    // Once the page is loaded, start updating the content
    this.logsWindow.webContents.once('did-finish-load', () => {
      // Initial content update
      this.updateLogsContent();

      // Set up auto-refresh: update logs every 2 seconds
      this.logsRefreshInterval = setInterval(() => {
        this.updateLogsContent();
      }, 2000);
    });

    this.logsWindow.on('closed', () => {
      if (this.logsRefreshInterval) {
        clearInterval(this.logsRefreshInterval);
        this.logsRefreshInterval = null;
      }
      this.logsWindow = null;
    });
  }

  getLogs(): string {
    return this.readLogFile();
  }
}
