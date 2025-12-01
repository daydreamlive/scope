const { app, BrowserWindow, Tray, Menu, shell, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const http = require('http');
const pythonDepsInstaller = require('./install-python-deps');

// Initialize the Python dependency installer with Electron app instance
pythonDepsInstaller.init(app, dialog);
const { ensurePythonDependencies } = pythonDepsInstaller;

// Keep a global reference of the window and tray objects
let mainWindow = null;
let tray = null;
let pythonProcess = null;
let serverReady = false;
const SERVER_PORT = 8000;
const SERVER_HOST = '127.0.0.1';

// Paths
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
const appPath = app.getAppPath();
const userDataPath = app.getPath('userData');

// Python path - use bundled Python environment if available
function getPythonPath() {
  if (isDev) {
    // Development: use project .venv
    const devPythonPath = path.join(appPath, '..', '.venv', 'bin', 'python');
    if (fs.existsSync(devPythonPath)) {
      return devPythonPath;
    }
    // Fallback to system Python
    return 'python3';
  } else {
    // Production: use bundled Python environment
    const platform = process.platform;
    let bundledPythonPath;
    if (platform === 'win32') {
      bundledPythonPath = path.join(process.resourcesPath, 'python-env', '.venv', 'Scripts', 'python.exe');
    } else {
      bundledPythonPath = path.join(process.resourcesPath, 'python-env', '.venv', 'bin', 'python');
    }

    if (fs.existsSync(bundledPythonPath)) {
      return bundledPythonPath;
    }
    // Fallback to system Python (shouldn't happen if build was successful)
    return 'python3';
  }
}

const pythonPath = getPythonPath();
const serverScriptPath = isDev
  ? path.join(appPath, '..', 'src', 'scope', 'server', 'app.py')
  : path.join(process.resourcesPath, 'python-env', 'src', 'scope', 'server', 'app.py');

// Logs directory (matches Python server's default)
const logsDir = path.join(os.homedir(), '.daydream-scope', 'logs');

/**
 * Check if Python server is running
 */
function checkServerHealth() {
  return new Promise((resolve) => {
    const req = http.get(`http://${SERVER_HOST}:${SERVER_PORT}/health`, (res) => {
      resolve(res.statusCode === 200);
    });

    req.on('error', () => {
      resolve(false);
    });

    req.setTimeout(2000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

/**
 * Start the Python server
 */
function startPythonServer() {
  if (pythonProcess) {
    console.log('Python server already running');
    return;
  }

  console.log('Starting Python server...');
  console.log('Python path:', pythonPath);
  console.log('Server script:', serverScriptPath);

  // Check if Python executable exists
  if (pythonPath !== 'python3' && !fs.existsSync(pythonPath)) {
    console.error('Python executable not found at:', pythonPath);
    dialog.showErrorBox(
      'Python Not Found',
      `Python executable not found at: ${pythonPath}\n\nPlease rebuild the app to include the Python environment.`
    );
    return;
  }

  // Check if server script exists
  if (!fs.existsSync(serverScriptPath)) {
    console.error('Server script not found at:', serverScriptPath);
    dialog.showErrorBox(
      'Server Script Not Found',
      `Server script not found at: ${serverScriptPath}`
    );
    return;
  }

  // Use the bundled Python environment directly
  // All dependencies are pre-installed, so we just run Python directly
  let pythonCommand, pythonArgs;

  if (pythonPath !== 'python3' && fs.existsSync(pythonPath)) {
    // Use bundled Python
    pythonCommand = pythonPath;
    pythonArgs = [
      '-m', 'scope.server.app',
      '--host', SERVER_HOST,
      '--port', SERVER_PORT.toString(),
      '--no-browser'
    ];
    console.log(`Using bundled Python at: ${pythonPath}`);
  } else {
    // Fallback to system Python (shouldn't happen in production)
    pythonCommand = 'python3';
    pythonArgs = [
      '-m', 'scope.server.app',
      '--host', SERVER_HOST,
      '--port', SERVER_PORT.toString(),
      '--no-browser'
    ];
    console.warn('Using system Python (bundled Python not found)');
  }

  // Start Python server
  const projectDir = isDev
    ? path.join(appPath, '..')
    : path.join(process.resourcesPath, 'python-env');

  pythonProcess = spawn(pythonCommand, pythonArgs, {
    cwd: projectDir,
    env: {
      ...process.env,
      // Ensure Python can find the scope package
      PYTHONPATH: isDev
        ? path.join(appPath, '..', 'src')
        : path.join(process.resourcesPath, 'python-env', 'src')
    },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  pythonProcess.stdout.on('data', (data) => {
    const output = data.toString();
    console.log('[Python Server]', output);

    // Check for server ready indicators
    if (output.includes('Uvicorn running') || output.includes('Application startup complete')) {
      serverReady = true;
      if (mainWindow) {
        mainWindow.webContents.send('server-ready');
      }
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    const output = data.toString();
    console.error('[Python Server Error]', output);
    // In production, also show critical errors to user
    if (!isDev && output.includes('Error') || output.includes('Traceback')) {
      // Only show dialog for critical errors, not warnings
      if (output.includes('ModuleNotFoundError') || output.includes('ImportError') ||
          output.includes('FileNotFoundError') || output.includes('PermissionError')) {
        dialog.showErrorBox(
          'Python Server Error',
          `Python server encountered an error:\n\n${output.substring(0, 500)}`
        );
      }
    }
  });

  pythonProcess.on('error', (error) => {
    console.error('Failed to start Python server:', error);
    const errorMsg = `Failed to start Python server: ${error.message}\n\n` +
      `Command: ${pythonCommand} ${pythonArgs.join(' ')}\n` +
      `Working directory: ${isDev ? path.join(appPath, '..') : path.join(process.resourcesPath, 'app')}\n\n` +
      `Please ensure Python 3.10+ and uv are installed and available in PATH.`;
    dialog.showErrorBox('Server Start Error', errorMsg);
    pythonProcess = null;
  });

  pythonProcess.on('exit', (code, signal) => {
    console.log(`Python server exited with code ${code} and signal ${signal}`);
    pythonProcess = null;
    serverReady = false;

    if (code !== 0 && code !== null) {
      // Server crashed, try to restart after a delay
      setTimeout(() => {
        if (!pythonProcess) {
          console.log('Attempting to restart Python server...');
          startPythonServer();
        }
      }, 5000);
    }
  });
}

/**
 * Stop the Python server
 */
function stopPythonServer() {
  if (pythonProcess) {
    console.log('Stopping Python server...');
    pythonProcess.kill('SIGTERM');

    // Force kill after 5 seconds if still running
    setTimeout(() => {
      if (pythonProcess) {
        console.log('Force killing Python server...');
        pythonProcess.kill('SIGKILL');
      }
    }, 5000);

    pythonProcess = null;
    serverReady = false;
  }
}

/**
 * Create the main application window
 */
function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 600,
    icon: path.join(__dirname, 'assets', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true
    },
    show: false // Don't show until ready
  });

  // Load the app - always load from Python server (which serves the frontend)
  // The Python server will serve the frontend from frontend/dist if it exists
  const serverURL = `http://${SERVER_HOST}:${SERVER_PORT}`;
  mainWindow.loadURL(serverURL);

  // Handle page load failures
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
    if (errorCode === -106 || errorCode === -105) { // ERR_CONNECTION_REFUSED or ERR_NAME_NOT_RESOLVED
      console.error('Failed to load page:', errorCode, errorDescription);
      if (!isDev) {
        dialog.showErrorBox(
          'Connection Error',
          `Failed to connect to the Python server at ${serverURL}.\n\n` +
          `Error: ${errorDescription}\n\n` +
          `Please check:\n` +
          `1. The Python server is starting correctly\n` +
          `2. Port ${SERVER_PORT} is not in use by another application\n` +
          `3. Check the logs in ${logsDir}`
        );
      }
    }
  });

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();

    // Check if server is ready, if not wait a bit
    checkServerHealth().then(ready => {
      if (ready) {
        mainWindow.webContents.send('server-ready');
      } else {
        // Wait for server to be ready
        const checkInterval = setInterval(() => {
          checkServerHealth().then(ready => {
            if (ready) {
              clearInterval(checkInterval);
              mainWindow.webContents.send('server-ready');
            }
          });
        }, 1000);

        // Stop checking after 30 seconds
        setTimeout(() => clearInterval(checkInterval), 30000);
      }
    });
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

/**
 * Create system tray icon and menu
 */
function createTray() {
  try {
    // Use a placeholder icon for now (will be replaced with actual icon)
    const iconPath = path.join(__dirname, 'assets', 'tray-icon.png');
    const fallbackIconPath = path.join(__dirname, 'assets', 'icon.png');

    // Fallback to a default icon if custom icon doesn't exist
    let trayIcon;
    if (fs.existsSync(iconPath)) {
      trayIcon = iconPath;
    } else if (fs.existsSync(fallbackIconPath)) {
      trayIcon = fallbackIconPath;
    } else {
      // If no icon exists, log a warning and skip tray creation
      console.warn('Tray icon not found. Skipping tray creation.');
      console.warn('Expected paths:', iconPath, 'or', fallbackIconPath);
      return;
    }

    tray = new Tray(trayIcon);
  } catch (error) {
    console.error('Failed to create tray:', error);
    console.error('Icon path attempted:', error.path || 'unknown');
    // Don't throw - allow app to continue without tray
    return;
  }

  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Open Scope',
      click: () => {
        if (mainWindow) {
          if (mainWindow.isMinimized()) mainWindow.restore();
          mainWindow.focus();
        } else {
          createWindow();
        }
      }
    },
    {
      label: 'Settings',
      click: () => {
        if (mainWindow) {
          if (mainWindow.isMinimized()) mainWindow.restore();
          mainWindow.focus();
          // Navigate to settings - for now just focus the window
          // In the future, we could add a route or query param
          mainWindow.webContents.send('navigate-to-settings');
        } else {
          createWindow();
          mainWindow.webContents.once('did-finish-load', () => {
            mainWindow.webContents.send('navigate-to-settings');
          });
        }
      }
    },
    {
      label: 'Logs',
      click: () => {
        // Open the logs directory in the file manager
        shell.openPath(logsDir).catch(err => {
          console.error('Failed to open logs directory:', err);
          dialog.showErrorBox(
            'Error',
            `Failed to open logs directory: ${err.message}`
          );
        });
      }
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        stopPythonServer();
        app.quit();
      }
    }
  ]);

  tray.setToolTip('Daydream Scope');
  tray.setContextMenu(contextMenu);

  // Double-click to open/restore window
  tray.on('double-click', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    } else {
      createWindow();
    }
  });
}

// IPC handlers
ipcMain.handle('check-server-health', async () => {
  return await checkServerHealth();
});

ipcMain.handle('get-logs-dir', () => {
  return logsDir;
});

// App event handlers
app.whenReady().then(async () => {
  createTray();

  // Ensure Python dependencies are installed before starting server
  const depsReady = await ensurePythonDependencies();
  if (depsReady) {
    startPythonServer();
    // Create window after starting server (give it a moment to initialize)
    setTimeout(() => {
      createWindow();
    }, 1000);
  } else {
    // Show error and quit if dependencies can't be installed
    dialog.showErrorBox(
      'Setup Failed',
      'Failed to set up Python dependencies. The application will now exit.'
    );
    app.quit();
  }

  // Create window on macOS when dock icon is clicked
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    } else if (mainWindow) {
      mainWindow.focus();
    }
  });
});

// Quit when all windows are closed (except on macOS)
app.on('window-all-closed', () => {
  // On macOS, keep app running even when all windows are closed
  if (process.platform !== 'darwin') {
    // Don't quit - keep running in background with tray icon
    // app.quit();
  }
});

app.on('before-quit', () => {
  stopPythonServer();
});

app.on('will-quit', () => {
  stopPythonServer();
});

// Handle app termination
process.on('exit', () => {
  stopPythonServer();
});
