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
    const childProcess = spawn(checkCommand, [command]);

    childProcess.on('close', (code) => {
      resolve(code === 0);
    });

    childProcess.on('error', () => {
      resolve(false);
    });
  });
}

/**
 * Get the path to uv binary (bundled or system)
 */
function getUvPath() {
  if (!isDev && process.resourcesPath) {
    // Try to find bundled uv
    const platform = os.platform();
    const arch = os.arch();

    let bundledUvPath;
    if (platform === 'win32') {
      bundledUvPath = path.join(process.resourcesPath, 'uv-binaries', 'x86_64-pc-windows-msvc', 'uv.exe');
    } else if (platform === 'darwin') {
      if (arch === 'arm64') {
        bundledUvPath = path.join(process.resourcesPath, 'uv-binaries', 'aarch64-apple-darwin', 'uv');
      } else {
        bundledUvPath = path.join(process.resourcesPath, 'uv-binaries', 'x86_64-apple-darwin', 'uv');
      }
    } else if (platform === 'linux') {
      bundledUvPath = path.join(process.resourcesPath, 'uv-binaries', 'x86_64-unknown-linux-gnu', 'uv');
    }

    if (bundledUvPath && fs.existsSync(bundledUvPath)) {
      console.log(`Using bundled uv at: ${bundledUvPath}`);
      return bundledUvPath;
    }
  }

  // Fall back to system uv
  return 'uv';
}

/**
 * Check if uv is available (bundled or system)
 */
async function checkUvInstalled() {
  const uvPath = getUvPath();
  if (uvPath !== 'uv') {
    // Bundled uv exists
    return true;
  }
  // Check if system uv exists
  return await commandExists('uv');
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

    const childProcess = spawn('sh', ['-c', installScript], {
      shell: true,
      stdio: 'inherit'
    });

    childProcess.on('close', (code) => {
      if (code === 0) {
        console.log('uv installed successfully');
        resolve();
      } else {
        reject(new Error(`Failed to install uv: exit code ${code}`));
      }
    });

    childProcess.on('error', (error) => {
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

    // Verify required files exist (for debugging)
    const readmePath = path.join(projectDir, 'README.md');
    if (!fs.existsSync(readmePath)) {
      console.warn(`Warning: README.md not found at: ${readmePath}`);
      console.warn(`Project directory: ${projectDir}`);
      console.warn(`Files in project directory:`, fs.readdirSync(projectDir).join(', '));
    } else {
      console.log(`Verified README.md exists at: ${readmePath}`);
    }

    // Check if lockfile exists
    const lockfilePath = isDev
      ? path.join(appPath, '..', 'uv.lock')
      : path.join(process.resourcesPath, 'app', 'uv.lock');

    const hasLockfile = fs.existsSync(lockfilePath);

    // Use uv to sync dependencies
    // In production, don't use --frozen to avoid path resolution issues with file:// URLs
    // In development, use --frozen if lockfile exists for faster, reproducible installs
    const useFrozen = isDev && hasLockfile;
    const uvArgs = useFrozen ? ['sync', '--frozen'] : ['sync'];

    const uvPath = getUvPath();
    console.log(`Using uv sync${useFrozen ? ' --frozen' : ''} (lockfile ${hasLockfile ? 'found' : 'not found'}, mode: ${isDev ? 'dev' : 'production'})`);
    console.log(`uv path: ${uvPath}`);

    // uv will automatically download Python if needed
    // Set UV_PYTHON to let uv manage Python installation
    const env = {
      ...process.env,
      // Let uv download and manage Python automatically
      // UV_PYTHON can be set to a specific version like "3.12" or left unset for auto-detection
    };

    const childProcess = spawn(uvPath, uvArgs, {
      cwd: projectDir,
      stdio: 'inherit',
      env: env
    });

    childProcess.on('close', (code) => {
      if (code === 0) {
        console.log('Python dependencies installed successfully');
        resolve();
      } else {
        reject(new Error(`Failed to install Python dependencies: exit code ${code}`));
      }
    });

    childProcess.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * Check if bundled Python environment exists
 */
function checkBundledPython() {
  if (isDev) {
    return false; // Always install in dev mode
  }

  if (!process.resourcesPath) {
    return false;
  }

  const platform = process.platform;
  let pythonPath;
  if (platform === 'win32') {
    pythonPath = path.join(process.resourcesPath, 'python-env', '.venv', 'Scripts', 'python.exe');
  } else {
    pythonPath = path.join(process.resourcesPath, 'python-env', '.venv', 'bin', 'python');
  }

  return fs.existsSync(pythonPath);
}

/**
 * Main installation check and setup
 */
async function ensurePythonDependencies() {
  try {
    // In production, check if Python environment is already bundled
    if (!isDev && checkBundledPython()) {
      console.log('Bundled Python environment found - skipping installation');
      return true;
    }

    // In development or if bundled environment not found, install dependencies
    // Check for uv (bundled or system)
    const uvInstalled = await checkUvInstalled();
    if (!uvInstalled) {
      // If we have bundled uv, we should always have it
      // If not, try to install system uv
      if (isDev) {
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
      } else {
        throw new Error('uv not found. Please ensure the app is properly installed.');
      }
    }

    // Install Python dependencies
    // uv will automatically download Python if needed
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
  init,
  ensurePythonDependencies,
  checkUvInstalled
};
