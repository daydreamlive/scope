import fs from 'fs';
import path from 'path';
import os from 'os';
import https from 'https';
import { spawn, execSync } from 'child_process';
import { SetupService } from '../types/services';
import { getPaths, UV_DOWNLOAD_URLS, getEnhancedPath } from '../utils/config';
import { logger } from '../utils/logger';

export class ScopeSetupService implements SetupService {
  isSetupNeeded(): boolean {
    const paths = getPaths();
    const uvExists = fs.existsSync(paths.uvBin);
    const projectExists = fs.existsSync(path.join(paths.projectRoot, 'pyproject.toml'));

    // We no longer need to check for venv existence here - uv sync will handle it
    // and will be fast if dependencies haven't changed
    return !uvExists || !projectExists;
  }

  async checkUvInstalled(): Promise<boolean> {
    try {
      // Check if uv is in PATH (using enhanced PATH for macOS app launches)
      execSync('uv --version', {
        stdio: 'ignore',
        env: {
          ...process.env,
          PATH: getEnhancedPath(),
        },
      });
      return true;
    } catch {
      // Check if uv is in our local directory
      const paths = getPaths();
      return fs.existsSync(paths.uvBin);
    }
  }

  async downloadAndInstallUv(): Promise<void> {
    const platform = process.platform;
    const arch = process.arch;
    const paths = getPaths();

    logger.info('Downloading uv...');

    // Determine download URL
    let downloadUrl: string;
    if (platform === 'darwin') {
      downloadUrl = arch === 'arm64'
        ? UV_DOWNLOAD_URLS.darwin.arm64
        : UV_DOWNLOAD_URLS.darwin.x64;
    } else if (platform === 'win32') {
      downloadUrl = UV_DOWNLOAD_URLS.win32.x64;
    } else if (platform === 'linux') {
      downloadUrl = arch === 'arm64'
        ? UV_DOWNLOAD_URLS.linux.arm64
        : UV_DOWNLOAD_URLS.linux.x64;
    } else {
      throw new Error(`Unsupported platform: ${platform}`);
    }

    // Create uv directory
    const uvDir = path.dirname(paths.uvBin);
    if (!fs.existsSync(uvDir)) {
      fs.mkdirSync(uvDir, { recursive: true });
    }

    // Download uv with retries
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'scope-uv-'));
    const archivePath = path.join(tmpDir, path.basename(downloadUrl));

    const maxRetries = 3;
    let lastError: Error | null = null;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        logger.info(`Downloading uv (attempt ${attempt}/${maxRetries})...`);
        await this.downloadFile(downloadUrl, archivePath);
        lastError = null;
        break;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
        logger.warn(`Download attempt ${attempt} failed:`, lastError.message);
        if (attempt < maxRetries) {
          const delay = attempt * 2000; // Exponential backoff: 2s, 4s
          logger.info(`Retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
          // Clean up failed download file if it exists
          try {
            if (fs.existsSync(archivePath)) {
              fs.unlinkSync(archivePath);
            }
          } catch {
            // ignore cleanup errors
          }
        }
      }
    }

    if (lastError) {
      throw new Error(`Failed to download uv after ${maxRetries} attempts: ${lastError.message}`);
    }

    // Extract and install
    if (platform === 'win32') {
      await this.extractZip(archivePath, uvDir);
      // On Windows, uv.exe might be in a subdirectory (e.g., uv-x86_64-pc-windows-msvc/uv.exe)
      const extractedUv = this.findFile(uvDir, 'uv.exe');
      if (extractedUv) {
        logger.info(`Found uv.exe at: ${extractedUv}`);
        fs.renameSync(extractedUv, paths.uvBin);
      } else {
        throw new Error('Could not find uv.exe in extracted archive');
      }
    } else {
      await this.extractTarGz(archivePath, uvDir);
      // On Unix, uv binary might be in a subdirectory
      const extractedUv = this.findFile(uvDir, 'uv');
      if (extractedUv) {
        logger.info(`Found uv at: ${extractedUv}`);
        fs.renameSync(extractedUv, paths.uvBin);
        // Make executable
        fs.chmodSync(paths.uvBin, 0o755);
      } else {
        throw new Error('Could not find uv in extracted archive');
      }
    }

    // Cleanup
    fs.rmSync(tmpDir, { recursive: true, force: true });

    logger.info('uv installed successfully');
  }

  async runUvSync(): Promise<void> {
    const paths = getPaths();
    const projectRoot = paths.projectRoot;
    const venvPath = paths.venvPath;

    // Use local uv if available, otherwise try system uv
    let uvCommand = paths.uvBin;
    if (!fs.existsSync(uvCommand)) {
      uvCommand = 'uv';
    }

    logger.info(`Running uv sync in ${projectRoot}...`);
    logger.info(`Virtual environment path: ${venvPath}`);

    return new Promise((resolve, reject) => {
      const proc = spawn(uvCommand, ['sync'], {
        cwd: projectRoot,
        stdio: 'pipe',
        shell: false,
        env: {
          ...process.env,
          PATH: getEnhancedPath(),
          // Use UV_PROJECT_ENVIRONMENT to place .venv in userData (writable)
          // while keeping source code in resources (read-only)
          UV_PROJECT_ENVIRONMENT: venvPath,
        },
      });

      proc.stdout?.on('data', (data) => {
        logger.info('[UV SYNC]', data.toString().trim());
      });

      proc.stderr?.on('data', (data) => {
        logger.warn('[UV SYNC]', data.toString().trim());
      });

      proc.on('close', (code) => {
        if (code === 0) {
          logger.info('uv sync completed successfully');
          resolve();
        } else {
          logger.error(`uv sync failed with code ${code}`);
          reject(new Error(`uv sync failed with code ${code}`));
        }
      });

      proc.on('error', (err) => {
        logger.error('uv sync error:', err);
        reject(err);
      });
    });
  }

  private async downloadFile(url: string, dest: string, maxRedirects: number = 10): Promise<void> {
    return new Promise((resolve, reject) => {
      const file = fs.createWriteStream(dest);
      let redirectCount = 0;

      const followRedirect = (currentUrl: string) => {
        https.get(currentUrl, (response) => {
          if (response.statusCode === 301 || response.statusCode === 302) {
            redirectCount++;
            if (redirectCount > maxRedirects) {
              file.close();
              reject(new Error(`Too many redirects for ${url}`));
              return;
            }
            const redirectUrl = response.headers.location;
            if (!redirectUrl) {
              file.close();
              reject(new Error(`Redirect with no location header for ${url}`));
              return;
            }
            // Follow the redirect
            logger.info(`Following redirect ${redirectCount}: ${redirectUrl}`);
            followRedirect(redirectUrl);
            return;
          }
          if (response.statusCode !== 200) {
            file.close();
            reject(new Error(`Failed to download ${url} (${response.statusCode})`));
            return;
          }
          response.pipe(file);
          file.on('finish', () => file.close(() => resolve()));
        }).on('error', (err) => {
          file.close();
          try {
            fs.unlinkSync(dest);
          } catch {
            // ignore
          }
          reject(err);
        });
      };

      followRedirect(url);
    });
  }

  /**
   * Find a file by name in a directory (searches recursively, max 2 levels deep)
   */
  private findFile(dir: string, filename: string, depth: number = 0): string | null {
    if (depth > 2) return null;

    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isFile() && entry.name === filename) {
          return fullPath;
        }
        if (entry.isDirectory()) {
          const found = this.findFile(fullPath, filename, depth + 1);
          if (found) return found;
        }
      }
    } catch {
      // ignore errors
    }
    return null;
  }

  private async extractTarGz(archivePath: string, destDir: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const tar = spawn('tar', ['-xzf', archivePath, '-C', destDir]);
      tar.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`tar exited with code ${code}`));
      });
      tar.on('error', reject);
    });
  }

  private async extractZip(archivePath: string, destDir: string): Promise<void> {
    return new Promise((resolve, reject) => {
      // Use PowerShell's Expand-Archive on Windows
      const powershellCmd = `Expand-Archive -Path "${archivePath}" -DestinationPath "${destDir}" -Force`;
      const proc = spawn('powershell', ['-NoProfile', '-Command', powershellCmd], {
        shell: false,
        stdio: 'pipe',
      });

      proc.stdout?.on('data', (data) => {
        logger.info('[EXTRACT]', data.toString().trim());
      });

      proc.stderr?.on('data', (data) => {
        logger.warn('[EXTRACT]', data.toString().trim());
      });

      proc.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`PowerShell Expand-Archive exited with code ${code}`));
      });
      proc.on('error', reject);
    });
  }
}
