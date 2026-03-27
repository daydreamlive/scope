/**
 * PostHog analytics provider.
 *
 * Wraps the posthog-js SDK to conform to the AnalyticsProvider interface.
 */

import posthog from "posthog-js";
import type { AnalyticsProvider, AnalyticsInitConfig } from "./types";

export class PostHogProvider implements AnalyticsProvider {
  readonly name = "posthog";

  init(config: AnalyticsInitConfig): void {
    posthog.init(config.token, {
      api_host: config.apiHost || "https://us.i.posthog.com",
      persistence: "localStorage",
      autocapture: false,
      capture_pageview: false,
      capture_pageleave: false,
      ip: false,
      // Disable session recording — out of scope per spec
      disable_session_recording: true,
    });
  }

  track(event: string, properties?: Record<string, unknown>): void {
    posthog.capture(event, properties);
  }

  registerSuperProperties(properties: Record<string, unknown>): void {
    posthog.register(properties);
  }

  identify(
    userId: string,
    traits?: { displayName?: string | null; email?: string | null }
  ): void {
    const props: Record<string, unknown> = {
      daydream_user_id: userId,
    };
    if (traits?.displayName) props.name = traits.displayName;
    if (traits?.email) props.email = traits.email;

    posthog.identify(userId, props);
  }

  reset(): void {
    posthog.reset();
  }

  optIn(): void {
    posthog.opt_in_capturing();
  }

  optOut(): void {
    posthog.opt_out_capturing();
  }
}
