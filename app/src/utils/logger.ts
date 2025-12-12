import fs from 'fs';
import path from 'path';
import { app } from 'electron';

const LOG_DIR = path.join(app.getPath('userData'), 'logs');
const LOG_FILE = path.join(LOG_DIR, 'main.log');
const PYTHON_LOGS_DIR = path.join(LOG_DIR, 'main');

/**
 * Get the path to the most recent Python server log file.
 * Python server writes logs as scope-logs-YYYY-MM-DD-HH-MM-SS.log
 */
function getMostRecentPythonLogFile(): string | null {
  if (!fs.existsSync(PYTHON_LOGS_DIR)) {
    return null;
  }

  try {
    const files = fs.readdirSync(PYTHON_LOGS_DIR)
      .filter(file => file.startsWith('scope-logs-') && file.endsWith('.log'))
      .map(file => ({
        name: file,
        path: path.join(PYTHON_LOGS_DIR, file),
        // Extract timestamp from filename for sorting (scope-logs-YYYY-MM-DD-HH-MM-SS.log)
        timestamp: file.replace('scope-logs-', '').replace('.log', ''),
      }))
      .filter(file => {
        // Validate timestamp format (YYYY-MM-DD-HH-MM-SS)
        const timestampRegex = /^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$/;
        return timestampRegex.test(file.timestamp);
      })
      .sort((a, b) => b.timestamp.localeCompare(a.timestamp)); // Sort descending (most recent first)

    if (files.length > 0) {
      return files[0].path;
    }
  } catch (err) {
    console.error('Error reading Python logs directory:', err);
  }

  return null;
}

export function getLogPath(): string {
  // Try to get the most recent Python server log file
  const pythonLogFile = getMostRecentPythonLogFile();
  if (pythonLogFile) {
    return pythonLogFile;
  }

  // Fallback to Electron app log file
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
