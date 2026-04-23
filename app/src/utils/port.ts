import { createServer } from 'net';
import { logger } from './logger';

/**
 * Check if a specific port is available
 */
export async function isPortAvailable(port: number, host: string = '127.0.0.1'): Promise<boolean> {
  return new Promise((resolve) => {
    const server = createServer();

    server.once('error', (_err: NodeJS.ErrnoException) => {
      // EADDRINUSE - port already in use
      // EACCES     - permission denied (e.g. privileged port)
      // ENOBUFS    - Windows: port is in an excluded/reserved range
      // Any other error also means the port is not usable
      resolve(false);
    });

    server.once('listening', () => {
      server.close(() => {
        resolve(true);
      });
    });

    server.listen(port, host);
  });
}

/**
 * Find an available port starting from the given port.
 *
 * On Windows 11, large contiguous TCP port ranges can be excluded by the OS
 * (visible via `netsh int ipv4 show excludedportrange protocol=tcp`).  A small
 * fixed `maxAttempts` may not be enough to escape such a block, so we use a
 * larger default and also apply a jump when many consecutive ports fail,
 * helping us skip over wide exclusion windows quickly.
 */
export async function findAvailablePort(
  startPort: number,
  host: string = '127.0.0.1',
  maxAttempts: number = 50
): Promise<number> {
  let candidate = startPort;
  let consecutiveFails = 0;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (candidate > 65535) {
      throw new Error('No available ports found in valid range');
    }

    if (await isPortAvailable(candidate, host)) {
      if (attempt > 0) {
        logger.info(`Port ${startPort} was busy, using port ${candidate} instead`);
      }
      return candidate;
    }

    consecutiveFails++;

    // After 10 consecutive failures, jump ahead by 200 ports to escape a
    // Windows excluded range block more quickly.
    if (consecutiveFails >= 10) {
      logger.warn(
        `${consecutiveFails} consecutive ports unavailable near ${candidate}; ` +
        `jumping ahead to escape possible OS-excluded port range`
      );
      candidate += 200;
      consecutiveFails = 0;
    } else {
      candidate++;
    }
  }

  throw new Error(`No available ports found after ${maxAttempts} attempts starting from ${startPort}`);
}
