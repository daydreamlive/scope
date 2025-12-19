import { app, ipcMain, nativeImage, dialog, session, shell } from 'electron';
import path from 'path';
import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import { AppState } from './types/services';
import { IPC_CHANNELS, SETUP_STATUS } from './types/ipc';
import { ScopeSetupService } from './services/setup';
import { ScopePythonProcessService } from './services/pythonProcess';
import { ScopeElectronAppService } from './services/electronApp';
import { logger } from './utils/logger';
import { SERVER_CONFIG, validateConfig } from './utils/config';

// Configure electron-log for auto-updater
log.transports.file.level = 'info';
autoUpdater.logger = log;

// Configure auto-updater
autoUpdater.autoDownload = false; // Don't download automatically, ask user first
autoUpdater.autoInstallOnAppQuit = true; // Install updates when app quits
autoUpdater.disableDifferentialDownload = true; // Disable differential downloads (no blockmaps)

// Auto-updater event handlers
let updateDownloaded = false;

autoUpdater.on('checking-for-update', () => {
  logger.info('Checking for updates...');
});

autoUpdater.on('update-available', (info) => {
  logger.info('Update available:', info);

  // Notify user about available update
  dialog.showMessageBox({
    type: 'info',
    title: 'Update Available',
    message: `A new version ${info.version} is available!`,
    detail: 'Would you like to download it now? The app will restart after the update is installed.',
    buttons: ['Download', 'Later'],
    defaultId: 0,
    cancelId: 1
  }).then(result => {
    if (result.response === 0) {
      autoUpdater.downloadUpdate();
    }
  });
});

autoUpdater.on('update-not-available', (info) => {
  logger.info('Update not available:', info);
});

autoUpdater.on('error', (err) => {
  logger.error('Auto-updater error:', err);
});

autoUpdater.on('download-progress', (progressObj) => {
  logger.info(`Download speed: ${progressObj.bytesPerSecond} - Downloaded ${progressObj.percent}%`);
});

autoUpdater.on('update-downloaded', (info) => {
  logger.info('Update downloaded:', info);
  updateDownloaded = true;

  // Notify user that update is ready
  dialog.showMessageBox({
    type: 'info',
    title: 'Update Ready',
    message: 'Update has been downloaded.',
    detail: 'The update will be installed when you quit the application. You can also restart now to install it.',
    buttons: ['Restart Now', 'Later'],
    defaultId: 0,
    cancelId: 1
  }).then(result => {
    if (result.response === 0) {
      // Quit and install
      autoUpdater.quitAndInstall(false, true);
    }
  });
});

// IPC Rate Limiting Configuration
const IPC_RATE_LIMITS = {
  windowMs: 1000, // 1 second window
  maxCalls: 100,  // Maximum 100 calls per second per channel
  maxCallsPerMinute: 1000 // Maximum 1000 calls per minute per channel
};

const ipcCallCounts = new Map<string, { count: number; minuteCount: number; resetTime: number; minuteResetTime: number }>();

/**
 * Validate IPC rate limits to prevent DoS attacks
 */
function validateIPCRateLimit(channel: string): boolean {
  const now = Date.now();
  const current = ipcCallCounts.get(channel);

  // Initialize or reset counters if needed
  if (!current || now > current.resetTime) {
    ipcCallCounts.set(channel, {
      count: 1,
      minuteCount: 1,
      resetTime: now + IPC_RATE_LIMITS.windowMs,
      minuteResetTime: now + 60000 // 1 minute
    });
    return true;
  }

  // Reset minute counter if minute window has passed
  if (now > current.minuteResetTime) {
    current.minuteCount = 0;
    current.minuteResetTime = now + 60000;
  }

  // Check per-second rate limit
  if (current.count >= IPC_RATE_LIMITS.maxCalls) {
    logger.warn(`IPC rate limit exceeded for channel: ${channel} (${current.count} calls in ${IPC_RATE_LIMITS.windowMs}ms)`);
    return false;
  }

  // Check per-minute rate limit
  if (current.minuteCount >= IPC_RATE_LIMITS.maxCallsPerMinute) {
    logger.warn(`IPC minute rate limit exceeded for channel: ${channel} (${current.minuteCount} calls in 1 minute)`);
    return false;
  }

  // Increment counters
  current.count++;
  current.minuteCount++;

  return true;
}

// Setup logging early
logger.info('Application starting...');

// Validate configuration early to catch issues at startup
try {
  validateConfig();
  logger.info('Configuration validated successfully');
} catch (err) {
  logger.error('Configuration validation failed:', err);
  dialog.showErrorBox(
    'Configuration Error',
    `The application configuration is invalid:\n\n${err instanceof Error ? err.message : String(err)}\n\nThe application will now exit.`
  );
  app.quit();
  process.exit(1);
}

/**
 * Main application state
 */
const appState: AppState = {
  mainWindow: null,
  isSettingUp: false,
  needsSetup: false,
  currentSetupStatus: SETUP_STATUS.INITIALIZING,
  serverProcess: null,
  isServerRunning: false,
};

// Initialize services
let setupService: ScopeSetupService;
let pythonProcessService: ScopePythonProcessService;
let electronAppService: ScopeElectronAppService;

// Request single instance lock for Windows Jump List functionality
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  // Another instance is already running, quit this one
  app.quit();
} else {
  // Handle second instance (e.g., when user clicks Jump List item while app is running)
  app.on('second-instance', (_event, commandLine) => {
    // Someone tried to run a second instance, focus our window and handle args
    if (appState.mainWindow) {
      if (appState.mainWindow.isMinimized()) {
        appState.mainWindow.restore();
      }
      appState.mainWindow.focus();
    }

    // Check if launched with --show-logs
    if (commandLine.includes('--show-logs')) {
      if (electronAppService) {
        electronAppService.showLogsWindow();
      }
    }
  });
}

/**
 * IPC Validation Wrapper
 * Wraps IPC handlers with error handling and validation
 */
function validateIPC<T extends (...args: any[]) => Promise<any>>(
  handler: T,
  handlerName: string
): T {
  return (async (...args: any[]) => {
    try {
      // Apply rate limiting
      if (!validateIPCRateLimit(handlerName)) {
        throw new Error(`IPC rate limit exceeded for ${handlerName}`);
      }

      // Log IPC call for debugging (only in development)
      if (!app.isPackaged) {
        logger.debug(`IPC call: ${handlerName}`, args);
      }

      // Validate that handler exists
      if (typeof handler !== 'function') {
        throw new Error(`IPC handler ${handlerName} is not a function`);
      }

      // Execute handler
      const result = await handler(...args);

      // Validate result is serializable (basic check)
      if (result !== undefined && typeof result === 'object') {
        try {
          JSON.stringify(result);
        } catch (err) {
          logger.warn(`IPC handler ${handlerName} returned non-serializable result`);
        }
      }

      return result;
    } catch (error) {
      logger.error(`IPC handler ${handlerName} error:`, error);
      // Re-throw to let renderer handle it
      throw error;
    }
  }) as T;
}

/**
 * IPC Handlers with validation - register early so they're available when renderer loads
 */
ipcMain.handle(IPC_CHANNELS.GET_SETUP_STATE, validateIPC(async () => {
  return { needsSetup: appState.needsSetup };
}, IPC_CHANNELS.GET_SETUP_STATE));

ipcMain.handle(IPC_CHANNELS.GET_SETUP_STATUS, validateIPC(async () => {
  return { status: appState.currentSetupStatus };
}, IPC_CHANNELS.GET_SETUP_STATUS));

ipcMain.handle(IPC_CHANNELS.GET_SERVER_STATUS, validateIPC(async () => {
  return { isRunning: appState.isServerRunning };
}, IPC_CHANNELS.GET_SERVER_STATUS));

ipcMain.handle(IPC_CHANNELS.SHOW_CONTEXT_MENU, validateIPC(async () => {
  if (electronAppService) {
    electronAppService.showContextMenu();
  } else {
    logger.warn('showContextMenu called but electronAppService not initialized');
  }
}, IPC_CHANNELS.SHOW_CONTEXT_MENU));

ipcMain.handle(IPC_CHANNELS.GET_LOGS, validateIPC(async () => {
  if (electronAppService) {
    return electronAppService.getLogs();
  }
  logger.warn('getLogs called but electronAppService not initialized');
  return 'Service not initialized';
}, IPC_CHANNELS.GET_LOGS));

// Setup error callback for Python process
function setupPythonProcessErrorHandler(): void {
  pythonProcessService.setErrorCallback((error: string) => {
    logger.error('Python process error:', error);
    appState.isServerRunning = false;
    electronAppService.sendServerStatus(false);
    electronAppService.sendServerError(error);
  });
}

/**
 * Send setup status to renderer
 */
function sendSetupStatus(status: string): void {
  appState.currentSetupStatus = status;
  electronAppService.sendSetupStatus(status);
}

/**
 * Run the complete setup process
 */
async function runSetup(): Promise<void> {
  logger.info('runSetup() called - START');
  appState.isSettingUp = true;
  logger.info('About to send INITIALIZING status...');
  sendSetupStatus(SETUP_STATUS.INITIALIZING);
  logger.info('Setup status sent: INITIALIZING');

  // Window is already loaded, no need to wait again
  logger.info('Starting setup process...');

  try {
    logger.info('Checking if uv is installed...');
    sendSetupStatus(SETUP_STATUS.CHECKING_UV);
    logger.info('Setup status sent: CHECKING_UV');

    // Check if uv is installed
    const uvInstalled = await setupService.checkUvInstalled();
    logger.info(`UV installed: ${uvInstalled}`);

    if (!uvInstalled) {
      logger.info('UV not installed, downloading...');
      sendSetupStatus(SETUP_STATUS.DOWNLOADING_UV);
      logger.info('Setup status sent: DOWNLOADING_UV');
      await setupService.downloadAndInstallUv();
      logger.info('UV downloaded and installed');
      sendSetupStatus(SETUP_STATUS.INSTALLING_UV);
      logger.info('Setup status sent: INSTALLING_UV');
    }

    logger.info('Running uv sync...');
    sendSetupStatus(SETUP_STATUS.RUNNING_UV_SYNC);
    logger.info('Setup status sent: RUNNING_UV_SYNC');
    await setupService.runUvSync();
    logger.info('UV sync completed');

    appState.needsSetup = false;
    sendSetupStatus(SETUP_STATUS.SETUP_DONE);
    logger.info('Setup status sent: SETUP_DONE');
  } catch (err) {
    logger.error('Setup failed:', err);
    sendSetupStatus(SETUP_STATUS.SETUP_ERROR);
    throw err;
  } finally {
    // Wait a moment to show completion message
    setTimeout(() => {
      appState.isSettingUp = false;
    }, 1500);
  }
}

/**
 * Check if server is already running and accessible
 */
async function checkServerRunning(): Promise<boolean> {
  logger.info('Checking server status...');
  const http = await import('http');

  try {
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        req.destroy();
        reject(new Error('Request timeout'));
      }, 2000); // 2 second timeout

      const req = http.get(`${SERVER_CONFIG.url}/api/v1/health`, (res) => {
        clearTimeout(timeout);
        if (res.statusCode === 200) {
          resolve();
        } else {
          reject(new Error(`Server returned status ${res.statusCode}`));
        }
      });
      req.on('error', (err) => {
        clearTimeout(timeout);
        reject(err);
      });
    });
    logger.info('Server is already running and accessible');
    return true;
  } catch (err) {
    logger.info(`Server check failed (expected if not running): ${err instanceof Error ? err.message : String(err)}`);
    return false;
  }
}

/**
 * Wait for server to be available on port 8000
 * Default timeout is 10 minutes (600 attempts × 1 second) to allow for first-run dependency downloads
 */
async function waitForServer(maxAttempts: number = 600, intervalMs: number = 1000): Promise<boolean> {
  const http = await import('http');

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      await new Promise<void>((resolve, reject) => {
        const req = http.get(`${SERVER_CONFIG.url}/api/v1/health`, (res) => {
          if (res.statusCode === 200) {
            resolve();
          } else {
            reject(new Error(`Server returned status ${res.statusCode}`));
          }
        });
        req.on('error', reject);
        req.setTimeout(1000, () => {
          req.destroy();
          reject(new Error('Request timeout'));
        });
      });
      logger.info('Server is ready and accessible');
      return true;
    } catch (err) {
      // Server not ready yet, continue waiting
      if (attempt < maxAttempts - 1) {
        await new Promise(resolve => setTimeout(resolve, intervalMs));
      }
    }
  }

  return false;
}

/**
 * Start the Python server
 */
async function startServer(): Promise<void> {
  if (appState.isServerRunning) {
    logger.warn('Server is already running');
    return;
  }

  try {
    await pythonProcessService.startServer();
    // Don't mark server as running yet - wait until it's actually ready
    // This allows the renderer to show the ServerLoading screen

    // Wait for server to be available
    const serverReady = await waitForServer();

    if (serverReady) {
      logger.info('Server is ready, loading frontend...');
      appState.isServerRunning = true;
      electronAppService.sendServerStatus(true);
      electronAppService.loadFrontend();
    } else {
      throw new Error('Server failed to start within timeout period');
    }
  } catch (err) {
    logger.error('Failed to start server:', err);
    appState.isServerRunning = false;
    electronAppService.sendServerStatus(false);

    const errorMessage = err instanceof Error ? err.message : String(err);
    electronAppService.sendServerError(errorMessage);

    throw err;
  }
}

/**
 * Configure session permissions
 * Shows user consent dialogs for permissions instead of denying all
 */
async function configureSessionPermissions(): Promise<void> {
  // Set permission request handler - ask user for consent
  session.defaultSession.setPermissionRequestHandler(async (webContents, permission, callback, details) => {
    try {
      // Check if main window exists
      if (!appState.mainWindow || appState.mainWindow.isDestroyed()) {
        logger.warn(`Permission denied (no window): ${permission}`);
        callback(false);
        return;
      }

      // Define user-friendly permission descriptions
      const permissionDescriptions: Record<string, string> = {
        'media': 'Access to camera and microphone for video/audio features',
        'geolocation': 'Access to your location',
        'notifications': 'Show desktop notifications',
        'midi': 'Access to MIDI devices',
        'midiSysex': 'Access to MIDI devices with system exclusive messages',
        'pointerLock': 'Lock the mouse pointer (for games or 3D applications)',
        'fullscreen': 'Enter fullscreen mode',
        'openExternal': 'Open links in external applications',
        'clipboard-read': 'Read from clipboard',
        'clipboard-sanitized-write': 'Write to clipboard',
        'keyboardLock': 'Lock keyboard input',
        'unknown': 'Unknown permission request'
      };

      const description = permissionDescriptions[permission] || permissionDescriptions.unknown;
      const requestingUrl = details.requestingUrl || 'Unknown source';

      // Show permission dialog to user
      const result = await dialog.showMessageBox(appState.mainWindow, {
        type: 'question',
        title: 'Permission Request',
        message: `Allow "${permission}" permission?`,
        detail: `${description}\n\nRequested by: ${requestingUrl}`,
        buttons: ['Allow', 'Deny'],
        defaultId: 0,
        cancelId: 1,
        checkboxLabel: 'Remember this decision',
        checkboxChecked: false
      });

      const granted = result.response === 0;
      const remember = result.checkboxChecked;

      if (granted) {
        logger.info(`Permission granted: ${permission} (remember: ${remember})`);
        if (remember) {
          // TODO: Store permission decision persistently
          logger.info(`Would persist permission grant for: ${permission}`);
        }
      } else {
        logger.info(`Permission denied: ${permission} (remember: ${remember})`);
        if (remember) {
          // TODO: Store permission denial persistently
          logger.info(`Would persist permission denial for: ${permission}`);
        }
      }

      callback(granted);
    } catch (error) {
      logger.error('Error handling permission request:', error);
      // Deny permission on error for security
      callback(false);
    }
  });

  // Set permission check handler - deny all by default for security
  session.defaultSession.setPermissionCheckHandler(() => {
    // Permission checks are handled by the request handler above
    // This ensures no permissions are granted without user consent
    return false;
  });

  // Disable cache in development mode to ensure fresh frontend code
  if (!app.isPackaged) {
    session.defaultSession.webRequest.onBeforeSendHeaders((details, callback) => {
      // Add cache-busting headers for all requests in development
      callback({
        requestHeaders: {
          ...details.requestHeaders,
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0',
        },
      });
    });
    logger.info('Cache disabled in development mode');
  }

  logger.info('Session permissions configured with user consent dialogs');
}

/**
 * Configure process-level security hardening
 * Adds webContents event handlers and security measures
 */
function configureProcessSecurity(): void {
  // Global webContents security handlers
  app.on('web-contents-created', (_event, contents) => {
    // Set up window open handler for all windows to open external links in default browser
    contents.setWindowOpenHandler(({ url }) => {
      try {
        const parsedUrl = new URL(url);
        // Check if it's an external URL (not localhost or the server)
        const allowedOrigins = [SERVER_CONFIG.url];
        if (!app.isPackaged) {
          allowedOrigins.push('http://localhost:5173');
        }

        const isExternal = !allowedOrigins.some(origin => {
          try {
            return new URL(origin).origin === parsedUrl.origin;
          } catch {
            return false;
          }
        });

        if (isExternal) {
          // Open external URLs in the default browser
          shell.openExternal(url).catch((err) => {
            logger.error(`Failed to open external URL: ${url}`, err);
          });
          return { action: 'deny' };
        }
        return { action: 'allow' };
      } catch (err) {
        // Invalid URL, block it
        logger.warn(`Blocked invalid URL: ${url}`);
        return { action: 'deny' };
      }
    });

    // Enhanced navigation protection
    contents.on('will-navigate', (event, navigationUrl) => {
      try {
        const parsedUrl = new URL(navigationUrl);

        // Allow localhost/dev server in development
        const allowedOrigins = [SERVER_CONFIG.url];
        if (!app.isPackaged) {
          allowedOrigins.push('http://localhost:5173');
        }

        const isAllowed = allowedOrigins.some(origin => parsedUrl.origin === origin);

        if (!isAllowed) {
          logger.warn(`Blocked navigation to external URL: ${navigationUrl}`);
          event.preventDefault();
        }
      } catch (err) {
        logger.warn(`Blocked navigation to invalid URL: ${navigationUrl}`);
        event.preventDefault();
      }
    });

    // Prevent navigation to about:blank and other schemes
    contents.on('will-navigate', (event, navigationUrl) => {
      if (navigationUrl.startsWith('about:') ||
          navigationUrl.startsWith('javascript:') ||
          navigationUrl.startsWith('data:') ||
          navigationUrl.startsWith('file:')) {
        logger.warn(`Blocked navigation to dangerous scheme: ${navigationUrl}`);
        event.preventDefault();
      }
    });

    // Monitor for DevTools opening in production
    if (app.isPackaged) {
      contents.on('devtools-opened', () => {
        logger.warn('DevTools opened in production - this should be monitored');
        // Note: We don't auto-close DevTools in production as it might be needed for debugging
        // But we log it for security monitoring
      });
    }

    // Prevent context menu in production (optional - can be enabled per window)
    if (app.isPackaged) {
      contents.on('context-menu', (event) => {
        // Allow context menu for text input fields only
        const target = event.params;
        if (!target || !target.isEditable) {
          event.preventDefault();
        }
      });
    }
  });

  logger.info('Process-level security hardening configured');
}

/**
 * Application ready handler
 */
app.on('ready', async () => {
  // Configure session permissions early
  await configureSessionPermissions();

  // Configure process-level security hardening
  configureProcessSecurity();

  // Check for updates (only in production)
  if (app.isPackaged) {
    // Check for updates after 3 seconds to let app initialize first
    setTimeout(() => {
      autoUpdater.checkForUpdates().catch(err => {
        logger.error('Failed to check for updates:', err);
      });
    }, 3000);

    // Check for updates every 4 hours
    setInterval(() => {
      autoUpdater.checkForUpdates().catch(err => {
        logger.error('Failed to check for updates:', err);
      });
    }, 4 * 60 * 60 * 1000);
  }

  // Set app icon (especially important for macOS dock icon in development)
  if (process.platform === 'darwin') {
    let iconPath: string;
    if (app.isPackaged) {
      iconPath = path.join(process.resourcesPath, 'app', 'assets', 'icon.png');
    } else {
      iconPath = path.join(__dirname, '../../assets/icon.png');
    }
    const icon = nativeImage.createFromPath(iconPath);
    if (!icon.isEmpty()) {
      app.dock.setIcon(icon);
    }
  }

  // Initialize services
  setupService = new ScopeSetupService();
  pythonProcessService = new ScopePythonProcessService();
  electronAppService = new ScopeElectronAppService(appState);

  // Setup error handler for Python process
  setupPythonProcessErrorHandler();

  // Initialize app state BEFORE creating window so IPC handlers can respond immediately
  logger.info('Checking if setup is needed...');
  appState.needsSetup = setupService.isSetupNeeded();
  logger.info(`Setup needed: ${appState.needsSetup}`);

  // Create main window
  electronAppService.createMainWindow();

  // Create system tray
  electronAppService.createTray();

  // Check if launched with special arguments (e.g., --show-logs from Jump List)
  electronAppService.checkLaunchArgs();

  // Wait for window to load before proceeding (with timeout)
  logger.info('Waiting for window to load...');
  await electronAppService.waitForMainWindowLoad();
  logger.info('Window loaded, proceeding with setup check...');

  if (appState.needsSetup) {
    logger.info('Setup needed, running setup...');
    try {
      // Run setup (download uv, run uv sync, etc.)
      // Note: Source code stays in resources (read-only), only .venv goes to userData
      await runSetup();
      logger.info('Setup completed');
    } catch (err) {
      logger.error('Setup error caught:', err);
      sendSetupStatus(SETUP_STATUS.SETUP_ERROR);
      throw err;
    }
  } else {
    logger.info('No setup needed');
  }

  // Check if server is already running before showing loading screen
  logger.info('Checking if server is already running...');
  const serverAlreadyRunning = await checkServerRunning();
  logger.info(`Server already running: ${serverAlreadyRunning}`);
  if (serverAlreadyRunning) {
    // Server is already running, update state and load frontend directly
    logger.info('Server is already running, loading frontend directly');
    appState.isServerRunning = true;
    electronAppService.sendServerStatus(true);
    // Small delay to ensure renderer has shown the loading state
    await new Promise(resolve => setTimeout(resolve, 500));
    electronAppService.loadFrontend();
  } else {
    // Start the Python server (it will wait for server to be ready and load frontend)
    // The renderer will show ServerLoading screen while we wait
    try {
      await startServer();
    } catch (err) {
      logger.error('Server startup failed:', err);
      // Error is already sent to renderer via sendServerError
    }
  }

  logger.info('Application ready');
});

/**
 * Application lifecycle handlers
 */
app.on('window-all-closed', () => {
  // Don't quit on macOS when all windows are closed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', async () => {
  // Recreate window on macOS when dock icon is clicked
  if (appState.mainWindow === null) {
    electronAppService.createMainWindow();

    // Check if server is already running
    const serverAlreadyRunning = await checkServerRunning();
    if (serverAlreadyRunning) {
      // Server is already running, update state and load frontend directly
      logger.info('Server is already running, loading frontend directly');
      appState.isServerRunning = true;
      electronAppService.sendServerStatus(true);
      // Wait a moment for window to be ready, then load frontend
      await electronAppService.waitForMainWindowLoad();
      electronAppService.loadFrontend();
    } else {
      // Check if setup is needed
      appState.needsSetup = setupService.isSetupNeeded();

      if (appState.needsSetup) {
        // Run setup (download uv, run uv sync, etc.)
        await runSetup();
      }

      // Start the Python server (it will wait for server to be ready and load frontend)
      try {
        await startServer();
      } catch (err) {
        logger.error('Server startup failed:', err);
        // Error is already sent to renderer via sendServerError
      }
    }
  } else {
    // Window exists, ensure it's visible and focused
    if (appState.mainWindow.isMinimized()) {
      appState.mainWindow.restore();
    }
    appState.mainWindow.show();
    appState.mainWindow.focus();
  }
});

/**
 * Cleanup function
 */
const cleanup = () => {
  logger.info('Cleaning up...');
  try {
    pythonProcessService?.stopServer();
    electronAppService?.cleanup();
    // Remove all IPC listeners
    ipcMain.removeAllListeners();
    logger.info('Application cleanup completed');
  } catch (err) {
    logger.error('Error during cleanup:', err);
  }
};

// Handle application lifecycle
app.on('before-quit', () => {
  cleanup();
});

app.on('will-quit', () => {
  cleanup();
});

// Handle process termination
process.on('exit', () => {
  cleanup();
});

process.on('SIGINT', () => {
  logger.info('Received SIGINT, shutting down gracefully...');
  cleanup();
  process.exit(0);
});

process.on('SIGTERM', () => {
  logger.info('Received SIGTERM, shutting down gracefully...');
  cleanup();
  process.exit(0);
});

process.on('uncaughtException', (err) => {
  logger.error('Uncaught Exception:', err);
  cleanup();
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // Don't exit on unhandled rejection, but log it
});
