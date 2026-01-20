import { createServer } from 'net';
import { logger } from './logger';

/**
 * Check if a specific port is available
 */
export async function isPortAvailable(port: number, host: string = '127.0.0.1'): Promise<boolean> {
  return new Promise((resolve) => {
    const server = createServer();

    server.once('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE' || err.code === 'EACCES') {
        resolve(false);
      } else {
        // Other errors - assume port is not available
        resolve(false);
      }
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
 * Find an available port starting from the given port
 * Tries up to maxAttempts consecutive ports
 */
export async function findAvailablePort(
  startPort: number,
  host: string = '127.0.0.1',
  maxAttempts: number = 5
): Promise<number> {
  for (let i = 0; i < maxAttempts; i++) {
    const port = startPort + i;
    if (port > 65535) {
      throw new Error('No available ports found in valid range');
    }

    if (await isPortAvailable(port, host)) {
      if (i > 0) {
        logger.info(`Port ${startPort} was busy, using port ${port} instead`);
      }
      return port;
    }
  }

  throw new Error(`No available ports found after ${maxAttempts} attempts starting from ${startPort}`);
}
