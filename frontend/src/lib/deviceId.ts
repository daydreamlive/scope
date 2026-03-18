const STORAGE_KEY = "daydream_device_id";

/**
 * Get or create a stable device identifier.
 * Persisted in localStorage. Falls back to generating a new UUID on first launch.
 */
export function getDeviceId(): string {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) return stored;
  } catch {
    // localStorage not available
  }

  const id = crypto.randomUUID();

  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // localStorage not available — device ID is ephemeral this session
  }

  return id;
}
