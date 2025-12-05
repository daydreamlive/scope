import fs from 'fs';
import path from 'path';
import { app } from 'electron';

const LOG_DIR = path.join(app.getPath('userData'), 'logs');
const LOG_FILE = path.join(LOG_DIR, 'main.log');

export function getLogPath(): string {
  return LOG_FILE;
}

// Ensure log directory exists
if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
}

const logStream = fs.createWriteStream(LOG_FILE, { flags: 'a' });

export const logger = {
  info: (...args: any[]) => {
    const message = `[INFO] ${new Date().toISOString()} ${args.map(String).join(' ')}\n`;
    process.stdout.write(message);
    logStream.write(message);
  },
  error: (...args: any[]) => {
    const message = `[ERROR] ${new Date().toISOString()} ${args.map(String).join(' ')}\n`;
    process.stderr.write(message);
    logStream.write(message);
  },
  warn: (...args: any[]) => {
    const message = `[WARN] ${new Date().toISOString()} ${args.map(String).join(' ')}\n`;
    process.stdout.write(message);
    logStream.write(message);
  },
};
