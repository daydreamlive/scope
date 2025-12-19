import path from 'path';
import { app } from 'electron';

export const getPaths = () => {
  const userDataPath = app.getPath('userData');

  // Architecture:
  // - projectRoot: Where Python source code lives (read-only in packaged mode)
  // - venvPath: Where .venv is created (writable, in userData)
  //
  // This separation ensures:
  // 1. Updates automatically get new Python/frontend code (in resources)
  // 2. The .venv persists in userData (no re-download unless deps change)
  // 3. No need to copy source files on update
  let projectRoot: string;

  if (app.isPackaged) {
    // When packaged, source files are in resourcesPath (read-only)
    projectRoot = process.resourcesPath || path.dirname(app.getPath('exe'));
  } else {
    // In development, go up from app/src/utils to the project root
    projectRoot = path.resolve(__dirname, '../../..');
  }

  // Virtual environment always lives in userData (writable)
  const venvPath = path.join(userDataPath, '.venv');

  return {
    userData: userDataPath,
    uvBin: path.join(userDataPath, 'uv', process.platform === 'win32' ? 'uv.exe' : 'uv'),
    projectRoot,
    venvPath,
    frontendDist: path.join(projectRoot, 'frontend', 'dist'),
  };
};

export const UV_DOWNLOAD_URLS = {
  darwin: {
    x64: 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin.tar.gz',
    arm64: 'https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz',
  },
  win32: {
    x64: 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip',
  },
  linux: {
    x64: 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz',
    arm64: 'https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-unknown-linux-gnu.tar.gz',
  },
} as const;

// Default port - using 52178 as it's less commonly used than 8000
const DEFAULT_PORT = 52178;
const DEFAULT_HOST = '127.0.0.1';

// Mutable server config - port may change if default is busy
export const SERVER_CONFIG = {
  host: DEFAULT_HOST,
  port: DEFAULT_PORT,
  get url() {
    return `http://${this.host}:${this.port}`;
  },
};

/**
 * Update the server port (called when finding an available port)
 */
export const setServerPort = (port: number): void => {
  if (port < 1 || port > 65535) {
    throw new Error(`Invalid port number: ${port}. Must be between 1 and 65535.`);
  }
  SERVER_CONFIG.port = port;
};

/**
 * Validate critical configuration at startup
 * Throws an error if required configuration is missing or invalid
 */
export const validateConfig = (): void => {
  // Validate port is in valid range
  if (SERVER_CONFIG.port < 1 || SERVER_CONFIG.port > 65535) {
    throw new Error(`SERVER_CONFIG.port must be a valid port number (1-65535), got: ${SERVER_CONFIG.port}`);
  }

  // Validate UV download URLs are present for current platform
  const platform = process.platform as keyof typeof UV_DOWNLOAD_URLS;
  if (!UV_DOWNLOAD_URLS[platform]) {
    throw new Error(`Unsupported platform: ${platform}`);
  }
};

/**
 * Build an enhanced PATH that includes common installation locations.
 * This is crucial for when the app is launched by double-clicking (not from terminal),
 * as macOS apps don't inherit the user's shell PATH.
 */
export const getEnhancedPath = (): string => {
  const originalPath = process.env.PATH || '';
  const homeDir = process.env.HOME || '';

  const commonPaths = [
    '/usr/local/bin',
    '/opt/homebrew/bin', // Homebrew on Apple Silicon
    '/usr/bin',
    '/bin',
    '/usr/sbin',
    '/sbin',
    path.join(homeDir, '.local', 'bin'), // uv default install location
    path.join(homeDir, '.cargo', 'bin'), // Rust/cargo tools
  ];

  // Add original PATH components that aren't already in commonPaths
  const pathComponents = originalPath.split(':').filter(p => p && !commonPaths.includes(p));
  const enhancedPath = [...commonPaths, ...pathComponents].join(':');

  return enhancedPath;
};
