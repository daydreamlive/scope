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
