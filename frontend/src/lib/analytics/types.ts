/**
 * Analytics provider interface.
 *
 * Any analytics backend (Mixpanel, PostHog, etc.) implements this interface.
 * The telemetry module delegates SDK-specific calls to the active provider.
 */

export interface AnalyticsProvider {
  /** Human-readable name for logging. */
  readonly name: string;

  /** Initialize the SDK. Called once on app start. */
  init(config: AnalyticsInitConfig): void;

  /** Track a named event with optional properties. */
  track(event: string, properties?: Record<string, unknown>): void;

  /** Register super properties attached to every subsequent event. */
  registerSuperProperties(properties: Record<string, unknown>): void;

  /** Associate future events with an authenticated user. */
  identify(
    userId: string,
    traits?: { displayName?: string | null; email?: string | null }
  ): void;

  /** Revert to anonymous tracking (on logout). */
  reset(): void;

  /** Opt the user in to tracking. */
  optIn(): void;

  /** Opt the user out of tracking. */
  optOut(): void;
}

export interface AnalyticsInitConfig {
  /** Project token / API key for the provider. */
  token: string;
  /** Optional API host override (PostHog self-hosted, etc.). */
  apiHost?: string;
}

export type AnalyticsProviderType = "mixpanel" | "posthog" | "noop";
