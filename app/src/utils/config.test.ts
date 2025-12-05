import { describe, it, expect, beforeEach, vi } from 'vitest';
import { validateConfig, SERVER_CONFIG, UV_DOWNLOAD_URLS } from './config';

// Mock electron app module
vi.mock('electron', () => ({
  app: {
    getPath: () => '/mock/path',
    isPackaged: false,
  },
}));

describe('Configuration Utilities', () => {
  describe('validateConfig', () => {
    it('should validate successfully with default configuration', () => {
      expect(() => validateConfig()).not.toThrow();
    });

    it('should have valid SERVER_CONFIG defaults', () => {
      expect(SERVER_CONFIG.host).toBe('127.0.0.1');
      expect(SERVER_CONFIG.port).toBeGreaterThan(0);
      expect(SERVER_CONFIG.port).toBeLessThanOrEqual(65535);
      expect(SERVER_CONFIG.url).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
    });

    it('should have UV download URLs for supported platforms', () => {
      expect(UV_DOWNLOAD_URLS.darwin).toBeDefined();
      expect(UV_DOWNLOAD_URLS.win32).toBeDefined();
      expect(UV_DOWNLOAD_URLS.linux).toBeDefined();

      // Verify URLs are valid
      expect(UV_DOWNLOAD_URLS.darwin.x64).toMatch(/^https:\/\//);
      expect(UV_DOWNLOAD_URLS.darwin.arm64).toMatch(/^https:\/\//);
    });
  });
});
