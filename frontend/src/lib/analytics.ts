/**
 * Analytics event emission layer
 *
 * Delegates to the telemetry module which manages provider lifecycle,
 * opt-in gating, and pre-disclosure queuing.
 */

import { track } from "./telemetry";

export function trackEvent(
  name: string,
  properties?: Record<string, unknown>
): void {
  if (import.meta.env.DEV) {
    console.debug("[analytics]", name, properties ?? {});
  }
  track(name, properties);
}
