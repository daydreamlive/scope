import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import { app } from 'electron';
import path from 'path';
import fs from 'fs';
import { PythonProcessService } from '../types/services';
import { getPaths, SERVER_CONFIG, getEnhancedPath } from '../utils/config';
import { logger } from '../utils/logger';

export class ScopePythonProcessService implements PythonProcessService {
  private serverProcess: ChildProcessWithoutNullStreams | null = null;
  private onErrorCallback: ((error: string) => void) | null = null;

  setErrorCallback(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  async startServer(): Promise<void> {
    if (this.serverProcess) {
      logger.warn('Server process already running');
      return;
    }

    const paths = getPaths();
    const projectRoot = paths.projectRoot;

    // Try to use local uv first, then fall back to system uv
    let uvCommand = 'uv';
    if (fs.existsSync(paths.uvBin)) {
      uvCommand = paths.uvBin;
    } else {
      // Try to find uv in PATH (using enhanced PATH for macOS app launches)
      try {
        const { execSync } = await import('child_process');
        execSync('uv --version', {
          stdio: 'ignore',
          env: {
            ...process.env,
            PATH: getEnhancedPath(),
          },
        });
        uvCommand = 'uv';
      } catch {
        logger.error('uv not found. Please ensure uv is installed.');
        throw new Error('uv not found');
      }
    }

    logger.info(`Starting server with: ${uvCommand} run daydream-scope --host ${SERVER_CONFIG.host} --port ${SERVER_CONFIG.port} --no-browser`);
    logger.info(`Working directory: ${projectRoot}`);

    const enhancedPath = getEnhancedPath();
    logger.info(`Using PATH: ${enhancedPath}`);

    const child = spawn(uvCommand, [
      'run',
      'daydream-scope',
      '--host',
      SERVER_CONFIG.host,
      '--port',
      String(SERVER_CONFIG.port),
      '--no-browser',
    ], {
      cwd: projectRoot,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
      env: {
        ...process.env,
        PATH: enhancedPath,
      },
    });

    this.serverProcess = child;
    this.setupProcessHandlers(child);
  }

  private setupProcessHandlers(child: ChildProcessWithoutNullStreams): void {
    let stderrBuffer = '';

    child.stdout?.on('data', (data) => {
      logger.info('[SERVER]', data.toString().trim());
    });

    child.stderr?.on('data', (data) => {
      const output = data.toString();
      stderrBuffer += output;
      logger.error('[SERVER]', output.trim());
    });

    child.on('close', (code, signal) => {
      logger.info(`[SERVER] closed with code ${code}, signal ${signal}`);
      if (code !== 0 && code !== null) {
        const errorMsg = `Server process exited with code ${code}${signal ? ` (signal: ${signal})` : ''}${stderrBuffer ? `\n\nError output:\n${stderrBuffer}` : ''}`;
        if (this.onErrorCallback) {
          this.onErrorCallback(errorMsg);
        }
      }
      this.serverProcess = null;
    });

    child.on('exit', (code, signal) => {
      logger.info(`[SERVER] exited with code ${code}, signal ${signal}`);
      if (code !== 0 && code !== null) {
        const errorMsg = `Server process exited with code ${code}${signal ? ` (signal: ${signal})` : ''}${stderrBuffer ? `\n\nError output:\n${stderrBuffer}` : ''}`;
        if (this.onErrorCallback) {
          this.onErrorCallback(errorMsg);
        }
      }
      this.serverProcess = null;
    });

    child.on('error', (err) => {
      logger.error('[SERVER] process error:', err);
      const errorMsg = `Failed to start server process: ${err.message}`;
      if (this.onErrorCallback) {
        this.onErrorCallback(errorMsg);
      }
      this.serverProcess = null;
    });
  }

  stopServer(): void {
    if (this.serverProcess) {
      logger.info('Stopping server...');
      this.serverProcess.kill('SIGINT');
      this.serverProcess = null;
    }
  }

  isServerRunning(): boolean {
    return this.serverProcess !== null && !this.serverProcess.killed;
  }
}
