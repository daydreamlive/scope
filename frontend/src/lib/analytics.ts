/**
 * Analytics event emission layer
 *
 * Stub implementation that logs in dev mode. When PostHog (or another
 * provider) is integrated, replace the body of `trackEvent` with the
 * real call (e.g. `posthog.capture(name, properties)`).
 */

export function trackEvent(
  name: string,
  properties?: Record<string, unknown>
): void {
  if (import.meta.env.DEV) {
    console.debug("[analytics]", name, properties ?? {});
  }
  // TODO: integrate PostHog — posthog.capture(name, properties)
}

/**
 * Creates a debounced tracker that coalesces rapid-fire events (e.g. slider
 * drags, search keystrokes) into a single event after a quiet period.
 *
 * @param delayMs - Debounce window in milliseconds (default 2000)
 * @returns A function `(name, properties, key?)` where `key` scopes the timer
 *          (defaults to `name`). Only the last call within the window fires.
 */
export function createDebouncedTracker(delayMs = 2000) {
  const timers = new Map<string, ReturnType<typeof setTimeout>>();
  return (name: string, properties?: Record<string, unknown>, key?: string) => {
    const k = key ?? name;
    const prev = timers.get(k);
    if (prev) clearTimeout(prev);
    timers.set(
      k,
      setTimeout(() => {
        trackEvent(name, properties);
        timers.delete(k);
      }, delayMs)
    );
  };
}
