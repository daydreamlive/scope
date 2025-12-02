import { app, ipcMain, nativeImage } from 'electron';
import path from 'path';
import started from 'electron-squirrel-startup';
import { AppState } from './types/services';
import { IPC_CHANNELS, SETUP_STATUS } from './types/ipc';
import { ScopeSetupService } from './services/setup';
import { ScopePythonProcessService } from './services/pythonProcess';
import { ScopeElectronAppService } from './services/electronApp';
import { logger } from './utils/logger';
import { SERVER_CONFIG } from './utils/config';

// Handle creating/removing shortcuts on Windows when installing/uninstalling
if (started) {
  app.quit();
}

// Setup logging early
logger.info('Application starting...');

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

/**
 * IPC Handlers with validation - register early so they're available when renderer loads
 */
ipcMain.handle(IPC_CHANNELS.GET_SETUP_STATE, async () => {
  return { needsSetup: appState.needsSetup };
});

ipcMain.handle(IPC_CHANNELS.GET_SETUP_STATUS, async () => {
  return { status: appState.currentSetupStatus };
});

ipcMain.handle(IPC_CHANNELS.GET_SERVER_STATUS, async () => {
  return { isRunning: appState.isServerRunning };
});

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
 */
async function waitForServer(maxAttempts: number = 60, intervalMs: number = 1000): Promise<boolean> {
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
 * Application ready handler
 */
app.on('ready', async () => {
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

  // Wait for window to load before proceeding (with timeout)
  logger.info('Waiting for window to load...');
  await electronAppService.waitForMainWindowLoad();
  logger.info('Window loaded, proceeding with setup check...');

  if (appState.needsSetup) {
    logger.info('Setup needed, running setup...');
    try {
      await runSetup();
      logger.info('Setup completed');
    } catch (err) {
      logger.error('Setup error caught:', err);
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
