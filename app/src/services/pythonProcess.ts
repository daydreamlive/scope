import { spawn, ChildProcessWithoutNullStreams, execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { PythonProcessService } from '../types/services';
import { getPaths, SERVER_CONFIG, getEnhancedPath, setServerPort } from '../utils/config';
import { logger } from '../utils/logger';
import { findAvailablePort } from '../utils/port';

export class ScopePythonProcessService implements PythonProcessService {
  private serverProcess: ChildProcessWithoutNullStreams | null = null;
  private onErrorCallback: ((error: string) => void) | null = null;
  private intentionalStop: boolean = false;
  private lastUsedPort: number | null = null;  // Track the port for respawns

  setErrorCallback(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  async startServer(isRespawn: boolean = false): Promise<void> {
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

    let portToUse: number;

    if (isRespawn && this.lastUsedPort !== null) {
      // On respawn, reuse the same port - the frontend expects it
      portToUse = this.lastUsedPort;
      logger.info(`Respawn: reusing port ${portToUse}`);
    } else {
      // Initial start: find an available port
      const desiredPort = SERVER_CONFIG.port;
      logger.info(`Finding available port starting from ${desiredPort}...`);
      portToUse = await findAvailablePort(desiredPort, SERVER_CONFIG.host);

      if (portToUse !== desiredPort) {
        logger.info(`Port ${desiredPort} was busy, using port ${portToUse} instead`);
      }
    }

    // Track the port for future respawns
    this.lastUsedPort = portToUse;
    setServerPort(portToUse);

    logger.info(`Starting server with: ${uvCommand} run daydream-scope --host ${SERVER_CONFIG.host} --port ${SERVER_CONFIG.port} --no-browser`);
    logger.info(`Working directory: ${projectRoot}`);

    // Build PATH with uv directory included so Python subprocess can find uv
    const uvDir = path.dirname(paths.uvBin);
    let pathEnv: string;
    if (process.platform === 'win32') {
      // On Windows, use semicolon separator and add uv directory
      pathEnv = [uvDir, process.env.PATH || ''].filter(Boolean).join(';');
    } else {
      // On Unix, include uv directory alongside enhanced paths
      pathEnv = [uvDir, getEnhancedPath()].filter(Boolean).join(':');
    }
    logger.info(`Using PATH: ${pathEnv}`);

    const child = spawn(uvCommand, [
      'run',
      '--no-sync',
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
        PATH: pathEnv,
        PYTHONUNBUFFERED: '1',
        // Use UV_PROJECT_ENVIRONMENT to use .venv from userData (writable)
        // while running source code from resources (read-only)
        UV_PROJECT_ENVIRONMENT: paths.venvPath,
        // Signal to the Python server that its lifecycle is managed externally.
        // When set, the server will exit with code 42 on restart instead of
        // using os.execv, allowing us to maintain PID tracking.
        DAYDREAM_SCOPE_MANAGED: '1',
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
      const wasIntentionalStop = this.intentionalStop;
      this.serverProcess = null;
      this.intentionalStop = false;

      // Exit code 42 = server requested restart, respawn unless intentionally stopped
      if (code === 42 && !wasIntentionalStop) {
        logger.info('[SERVER] Server requested restart (exit code 42), respawning after delay...');
        // Add a delay before respawning to ensure the port is fully released
        // Windows can keep ports in TIME_WAIT state briefly after process exit
        setTimeout(() => {
          logger.info('[SERVER] Delay complete, starting respawn...');
          this.startServer(true).catch((err) => {
            logger.error('[SERVER] Failed to respawn server:', err);
            if (this.onErrorCallback) {
              this.onErrorCallback(`Failed to restart server: ${err.message}`);
            }
          });
        }, 1000);  // 1 second delay
        return;
      }

      // Other non-zero exit codes are errors (unless intentionally stopped)
      if (code !== 0 && code !== null && !wasIntentionalStop) {
        const errorMsg = `Server process exited with code ${code}${signal ? ` (signal: ${signal})` : ''}${stderrBuffer ? `\n\nError output:\n${stderrBuffer}` : ''}`;
        if (this.onErrorCallback) {
          this.onErrorCallback(errorMsg);
        }
      }
    });

    child.on('exit', (code, signal) => {
      logger.info(`[SERVER] exited with code ${code}, signal ${signal}`);
      // Note: 'close' handler handles cleanup and respawn logic
      // Exit code 42 means server requested restart - not an error
      if (code !== 0 && code !== null && code !== 42 && !this.intentionalStop) {
        const errorMsg = `Server process exited with code ${code}${signal ? ` (signal: ${signal})` : ''}${stderrBuffer ? `\n\nError output:\n${stderrBuffer}` : ''}`;
        if (this.onErrorCallback) {
          this.onErrorCallback(errorMsg);
        }
      }
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
    this.intentionalStop = true;  // Mark as intentional to prevent respawn on exit code 42
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
