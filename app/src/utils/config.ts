import path from 'path';
import { app } from 'electron';

export const getPaths = () => {
  const userDataPath = app.getPath('userData');

  // In packaged mode, the Python project should be in resourcesPath (source files)
  // but we'll work in userData (writable location)
  // In dev mode, use the actual project root
  let projectRoot: string;
  let resourcesRoot: string;

  if (app.isPackaged) {
    // When packaged, source files are in resourcesPath
    resourcesRoot = process.resourcesPath || path.dirname(app.getPath('exe'));
    // But we work in userData for write permissions
    projectRoot = path.join(userDataPath, 'python-project');
  } else {
    // In development, go up from app/src/utils to the project root
    projectRoot = path.resolve(__dirname, '../../..');
    resourcesRoot = projectRoot;
  }

  return {
    userData: userDataPath,
    uvBin: path.join(userDataPath, 'uv', process.platform === 'win32' ? 'uv.exe' : 'uv'),
    projectRoot,
    resourcesRoot,
    frontendDist: app.isPackaged
      ? path.join(resourcesRoot, 'frontend', 'dist')
      : path.resolve(projectRoot, 'frontend/dist'),
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

// Mutable server config - port may change if default is busy
export const SERVER_CONFIG = {
  host: process.env.SCOPE_SERVER_HOST || '127.0.0.1',
  port: parseInt(process.env.SCOPE_SERVER_PORT || String(DEFAULT_PORT), 10),
  get url() {
    return process.env.SCOPE_SERVER_URL || `http://${this.host}:${this.port}`;
  },
};

/**
 * Update the server port (called when finding an available port)
 */
export const setServerPort = (port: number): void => {
  SERVER_CONFIG.port = port;
};

/**
 * Validate critical configuration at startup
 * Throws an error if required configuration is missing or invalid
 */
export const validateConfig = (): void => {
  const errors: string[] = [];

  // Validate server configuration
  if (!SERVER_CONFIG.host || SERVER_CONFIG.host.trim() === '') {
    errors.push('SERVER_CONFIG.host is required and cannot be empty');
  }

  if (isNaN(SERVER_CONFIG.port) || SERVER_CONFIG.port < 1 || SERVER_CONFIG.port > 65535) {
    errors.push(`SERVER_CONFIG.port must be a valid port number (1-65535), got: ${SERVER_CONFIG.port}`);
  }

  if (!SERVER_CONFIG.url || !SERVER_CONFIG.url.startsWith('http')) {
    errors.push(`SERVER_CONFIG.url must be a valid HTTP URL, got: ${SERVER_CONFIG.url}`);
  }

  // Validate UV download URLs are present
  const platform = process.platform as keyof typeof UV_DOWNLOAD_URLS;
  if (!UV_DOWNLOAD_URLS[platform]) {
    errors.push(`Unsupported platform: ${platform}`);
  }

  if (errors.length > 0) {
    throw new Error(`Configuration validation failed:\n${errors.join('\n')}`);
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
