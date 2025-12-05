import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// Cleanup after each test case (e.g. clearing jsdom)
afterEach(() => {
  cleanup();
});

// Mock Electron APIs for testing
global.window = global.window || {};

// Mock the scope API exposed by preload script
(global.window as any).scope = {
  onSetupStatus: () => () => {},
  onServerStatus: () => () => {},
  onServerError: () => () => {},
  getSetupState: async () => ({ needsSetup: false }),
  getSetupStatus: async () => ({ status: 'SETUP_DONE' }),
  getServerStatus: async () => ({ isRunning: true }),
  showContextMenu: async () => {},
  getLogs: async () => 'Test logs',
};
