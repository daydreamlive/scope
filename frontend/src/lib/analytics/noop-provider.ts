/**
 * No-op analytics provider.
 *
 * Used when no provider is configured or no token is available.
 * All methods are silent no-ops.
 */

import type { AnalyticsProvider } from "./types";

export class NoopProvider implements AnalyticsProvider {
  readonly name = "noop";
  init(): void {}
  track(): void {}
  registerSuperProperties(): void {}
  optIn(): void {}
  optOut(): void {}
}
