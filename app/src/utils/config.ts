import path from 'node:path';
import { app } from 'electron';

export const getPaths = () => {
  const userDataPath = app.getPath('userData');

  // In packaged mode, the Python project should be in resourcesPath
  // In dev mode, use the actual project root
  let projectRoot: string;
  if (app.isPackaged) {
    // When packaged, the entire project (including Python code) should be in resourcesPath
    projectRoot = process.resourcesPath || path.dirname(app.getPath('exe'));
  } else {
    // In development, go up from app/src/utils to the project root
    projectRoot = path.resolve(__dirname, '../../..');
  }

  return {
    userData: userDataPath,
    uvBin: path.join(userDataPath, 'uv', process.platform === 'win32' ? 'uv.exe' : 'uv'),
    projectRoot,
    frontendDist: app.isPackaged
      ? path.join(projectRoot, 'frontend', 'dist')
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

export const SERVER_CONFIG = {
  host: '127.0.0.1',
  port: 8000,
  url: 'http://127.0.0.1:8000',
} as const;

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
