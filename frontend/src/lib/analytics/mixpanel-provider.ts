/**
 * Mixpanel analytics provider.
 *
 * Wraps the mixpanel-browser SDK to conform to the AnalyticsProvider interface.
 */

import mixpanel from "mixpanel-browser";
import type { AnalyticsProvider, AnalyticsInitConfig } from "./types";

export class MixpanelProvider implements AnalyticsProvider {
  readonly name = "mixpanel";

  init(config: AnalyticsInitConfig): void {
    mixpanel.init(config.token, {
      persistence: "localStorage",
      ip: false,
      property_blacklist: [
        "$current_url",
        "$initial_referrer",
        "$initial_referring_domain",
        "$referrer",
      ],
    });
  }

  track(event: string, properties?: Record<string, unknown>): void {
    mixpanel.track(event, properties);
  }

  trackBeacon(event: string, properties?: Record<string, unknown>): void {
    mixpanel.track(event, properties, { transport: "sendBeacon" });
  }

  registerSuperProperties(properties: Record<string, unknown>): void {
    mixpanel.register(properties);
  }

  identify(
    userId: string,
    traits?: { displayName?: string | null; email?: string | null }
  ): void {
    mixpanel.identify(userId);

    const props: Record<string, unknown> = {
      daydream_user_id: userId,
    };
    if (traits?.displayName) props.$name = traits.displayName;
    if (traits?.email) props.$email = traits.email;
    mixpanel.people.set(props);
  }

  reset(): void {
    mixpanel.reset();
  }

  optIn(): void {
    mixpanel.opt_in_tracking();
  }

  optOut(): void {
    mixpanel.opt_out_tracking();
  }
}
