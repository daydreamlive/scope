import { spawn, ChildProcessWithoutNullStreams, execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { app } from 'electron';
import { PythonProcessService } from '../types/services';
import { getPaths, SERVER_CONFIG, getEnhancedPath, setServerPort } from '../utils/config';
import { logger } from '../utils/logger';
import { findAvailablePort } from '../utils/port';

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

    // Find an available port (use configured port as starting point)
    const desiredPort = SERVER_CONFIG.port;
    logger.info(`Finding available port starting from ${desiredPort}...`);

    // Find an available port
    const availablePort = await findAvailablePort(desiredPort, SERVER_CONFIG.host);

    // Update the server config with the actual port we're using
    if (availablePort !== desiredPort) {
      logger.info(`Port ${desiredPort} was busy, using port ${availablePort} instead`);
    }
    setServerPort(availablePort);

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
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: false,
      env: {
        ...process.env,
        PATH: enhancedPath,
        PYTHONUNBUFFERED: '1',
        // Use UV_PROJECT_ENVIRONMENT to use .venv from userData (writable)
        // while running source code from resources (read-only)
        UV_PROJECT_ENVIRONMENT: paths.venvPath,
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
      const pid = this.serverProcess.pid;

      if (process.platform === 'win32' && pid) {
        // On Windows, kill the entire process tree using taskkill
        // This ensures child processes (like the Python server spawned by uv) are also terminated
        try {
          execSync(`taskkill /PID ${pid} /T /F`, { stdio: 'ignore' });
          logger.info(`Killed process tree for PID ${pid}`);
        } catch (err) {
          logger.warn(`Failed to kill process tree: ${err}`);
          // Fallback to regular kill
          this.serverProcess.kill('SIGINT');
        }
      } else {
        // On Unix-like systems, SIGINT should propagate to child processes
        this.serverProcess.kill('SIGINT');
      }

      this.serverProcess = null;
    }
  }

  isServerRunning(): boolean {
    return this.serverProcess !== null && !this.serverProcess.killed;
  }
}
