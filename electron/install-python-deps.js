/**
 * Python Dependency Installer for Electron App
 *
 * This script handles installing Python dependencies when the app is first launched.
 * It checks for Python/uv installation and installs the required dependencies.
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// These will be set when the module is imported
let dialog = null;
let app = null;
let isDev = false;
let appPath = '';

/**
 * Initialize the module with Electron app and dialog instances
 */
function init(electronApp, electronDialog) {
  app = electronApp;
  dialog = electronDialog;
  isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
  appPath = app.getAppPath();
}

/**
 * Check if a command exists in PATH
 */
function commandExists(command) {
  return new Promise((resolve) => {
    const checkCommand = os.platform() === 'win32' ? 'where' : 'which';
    const process = spawn(checkCommand, [command]);

    process.on('close', (code) => {
      resolve(code === 0);
    });

    process.on('error', () => {
      resolve(false);
    });
  });
}

/**
 * Check if uv is installed
 */
async function checkUvInstalled() {
  return await commandExists('uv');
}

/**
 * Check if Python is installed
 */
async function checkPythonInstalled() {
  return await commandExists('python3') || await commandExists('python');
}

/**
 * Install uv if not present
 */
async function installUv() {
  return new Promise((resolve, reject) => {
    console.log('Installing uv...');

    const installScript = os.platform() === 'win32'
      ? 'powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"'
      : 'curl -LsSf https://astral.sh/uv/install.sh | sh';

    const process = spawn('sh', ['-c', installScript], {
      shell: true,
      stdio: 'inherit'
    });

    process.on('close', (code) => {
      if (code === 0) {
        console.log('uv installed successfully');
        resolve();
      } else {
        reject(new Error(`Failed to install uv: exit code ${code}`));
      }
    });

    process.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * Install Python dependencies using uv
 */
async function installPythonDependencies() {
  return new Promise((resolve, reject) => {
    console.log('Installing Python dependencies...');

    const pyprojectPath = isDev
      ? path.join(appPath, '..', 'pyproject.toml')
      : path.join(process.resourcesPath, 'app', 'pyproject.toml');

    if (!fs.existsSync(pyprojectPath)) {
      reject(new Error(`pyproject.toml not found at: ${pyprojectPath}`));
      return;
    }

    const projectDir = isDev
      ? path.join(appPath, '..')
      : path.join(process.resourcesPath, 'app');

    // Use uv to sync dependencies
    const process = spawn('uv', ['sync', '--frozen'], {
      cwd: projectDir,
      stdio: 'inherit',
      env: {
        ...process.env,
        // Ensure uv uses the correct Python version
        UV_PYTHON: 'python3'
      }
    });

    process.on('close', (code) => {
      if (code === 0) {
        console.log('Python dependencies installed successfully');
        resolve();
      } else {
        reject(new Error(`Failed to install Python dependencies: exit code ${code}`));
      }
    });

    process.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * Main installation check and setup
 */
async function ensurePythonDependencies() {
  try {
    // Check for Python
    const pythonInstalled = await checkPythonInstalled();
    if (!pythonInstalled) {
      const result = await dialog.showMessageBox({
        type: 'error',
        title: 'Python Not Found',
        message: 'Python is required to run Daydream Scope',
        detail: 'Please install Python 3.10 or later and try again.\n\nYou can download Python from https://www.python.org/downloads/',
        buttons: ['OK', 'Open Python Website']
      });

      if (result.response === 1 && app) {
        require('electron').shell.openExternal('https://www.python.org/downloads/');
      }
      throw new Error('Python not installed');
    }

    // Check for uv
    const uvInstalled = await checkUvInstalled();
    if (!uvInstalled) {
      const result = await dialog.showMessageBox({
        type: 'question',
        title: 'uv Not Found',
        message: 'uv package manager is required',
        detail: 'uv is needed to manage Python dependencies. Would you like to install it automatically?',
        buttons: ['Cancel', 'Install']
      });

      if (result.response === 0) {
        throw new Error('uv not installed and user declined installation');
      }

      await installUv();
    }

    // Install Python dependencies
    await installPythonDependencies();

    return true;
  } catch (error) {
    console.error('Error ensuring Python dependencies:', error);
    dialog.showErrorBox(
      'Installation Error',
      `Failed to set up Python dependencies: ${error.message}\n\nPlease ensure Python 3.10+ and uv are installed.`
    );
    return false;
  }
}

module.exports = {
  ensurePythonDependencies,
  checkUvInstalled,
  checkPythonInstalled
};
