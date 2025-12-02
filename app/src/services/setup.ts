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
    return !uvExists;
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

    // Download uv
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'scope-uv-'));
    const archivePath = path.join(tmpDir, path.basename(downloadUrl));

    await this.downloadFile(downloadUrl, archivePath);

    // Extract and install
    if (platform === 'win32') {
      await this.extractZip(archivePath, uvDir);
      // On Windows, uv.exe is in the extracted directory
      const extractedUv = path.join(uvDir, 'uv.exe');
      if (fs.existsSync(extractedUv)) {
        fs.renameSync(extractedUv, paths.uvBin);
      }
    } else {
      await this.extractTarGz(archivePath, uvDir);
      // On Unix, uv binary is in the extracted directory
      const extractedUv = path.join(uvDir, 'uv');
      if (fs.existsSync(extractedUv)) {
        fs.renameSync(extractedUv, paths.uvBin);
        // Make executable
        fs.chmodSync(paths.uvBin, 0o755);
      }
    }

    // Cleanup
    fs.rmSync(tmpDir, { recursive: true, force: true });

    logger.info('uv installed successfully');
  }

  async runUvSync(): Promise<void> {
    const paths = getPaths();
    const projectRoot = paths.projectRoot;

    // Use local uv if available, otherwise try system uv
    let uvCommand = paths.uvBin;
    if (!fs.existsSync(uvCommand)) {
      uvCommand = 'uv';
    }

    logger.info(`Running uv sync in ${projectRoot}...`);

    return new Promise((resolve, reject) => {
      const proc = spawn(uvCommand, ['sync'], {
        cwd: projectRoot,
        stdio: 'pipe',
        shell: false,
        env: {
          ...process.env,
          PATH: getEnhancedPath(),
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

  private async downloadFile(url: string, dest: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const file = fs.createWriteStream(dest);
      https.get(url, (response) => {
        if (response.statusCode === 301 || response.statusCode === 302) {
          // Handle redirect
          https.get(response.headers.location!, (redirectResponse) => {
            if (redirectResponse.statusCode !== 200) {
              reject(new Error(`Failed to download ${url} (${redirectResponse.statusCode})`));
              return;
            }
            redirectResponse.pipe(file);
            file.on('finish', () => file.close(() => resolve()));
          }).on('error', reject);
          return;
        }
        if (response.statusCode !== 200) {
          reject(new Error(`Failed to download ${url} (${response.statusCode})`));
          return;
        }
        response.pipe(file);
        file.on('finish', () => file.close(() => resolve()));
      }).on('error', (err) => {
        try {
          fs.unlinkSync(dest);
        } catch {
          // ignore
        }
        reject(err);
      });
    });
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
      const unzip = spawn('unzip', ['-q', archivePath, '-d', destDir]);
      unzip.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`unzip exited with code ${code}`));
      });
      unzip.on('error', reject);
    });
  }
}
