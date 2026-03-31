/**
 * Telemetry module — provider-agnostic analytics orchestrator.
 *
 * All tracking flows through this module. It manages:
 * - Analytics provider lifecycle (init, identify, reset)
 * - Provider selection via VITE_ANALYTICS_PROVIDER env var
 * - Opt-in preference (env vars > localStorage > default OFF)
 * - Pre-disclosure event queuing (queue until user sees disclosure)
 * - Super properties attached to every event
 *
 * Swap between providers by setting VITE_ANALYTICS_PROVIDER to "posthog",
 * "mixpanel", or "noop". When unset, production builds default to PostHog;
 * dev defaults to noop (no SDK).
 */

import type {
  AnalyticsProvider,
  AnalyticsProviderType,
} from "./analytics/types";
import { MixpanelProvider } from "./analytics/mixpanel-provider";
import { PostHogProvider } from "./analytics/posthog-provider";
import { NoopProvider } from "./analytics/noop-provider";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LS_TELEMETRY_ENABLED = "scope_telemetry_enabled";
const LS_TELEMETRY_DISCLOSED = "scope_telemetry_disclosed";
const LS_DEVICE_ID = "scope_device_id";
const EVENT_QUEUE_CAP = 500;

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

let _provider: AnalyticsProvider = new NoopProvider();
let _initialized = false;
let _eventQueue: Array<{ event: string; properties: Record<string, unknown> }> =
  [];
let _sessionId: string | null = null;
const _appStartTime: number =
  typeof performance !== "undefined" ? performance.now() : Date.now();

// ---------------------------------------------------------------------------
// Provider resolution
// ---------------------------------------------------------------------------

function getProviderType(): AnalyticsProviderType {
  const env =
    typeof import.meta !== "undefined"
      ? (import.meta.env?.VITE_ANALYTICS_PROVIDER as string | undefined)
      : undefined;
  const trimmed = env?.trim();
  if (trimmed === "posthog") return "posthog";
  if (trimmed === "mixpanel") return "mixpanel";
  if (trimmed === "noop") return "noop";
  // Preview / production builds without the env var should send to PostHog.
  if (typeof import.meta !== "undefined" && import.meta.env.PROD) {
    return "posthog";
  }
  return "noop";
}

function getToken(providerType: AnalyticsProviderType): string {
  if (providerType === "noop") return "";
  if (providerType === "posthog") {
    return (
      (typeof import.meta !== "undefined"
        ? (import.meta.env?.VITE_POSTHOG_KEY as string | undefined)
        : undefined) || ""
    );
  }
  return (
    (typeof import.meta !== "undefined"
      ? (import.meta.env?.VITE_MIXPANEL_TOKEN as string | undefined)
      : undefined) || ""
  );
}

function getApiHost(): string | undefined {
  if (typeof import.meta === "undefined") return undefined;
  return import.meta.env?.VITE_POSTHOG_HOST as string | undefined;
}

function createProvider(
  providerType: AnalyticsProviderType
): AnalyticsProvider {
  switch (providerType) {
    case "posthog":
      return new PostHogProvider();
    case "mixpanel":
      return new MixpanelProvider();
    case "noop":
      return new NoopProvider();
    default:
      return new NoopProvider();
  }
}

/** Get the name of the currently active analytics provider. */
export function getActiveProvider(): string {
  return _provider.name;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function generateUUID(): string {
  return crypto.randomUUID();
}

/** Get or create a persistent device ID. */
export function getDeviceId(): string {
  let id = localStorage.getItem(LS_DEVICE_ID);
  if (!id) {
    id = generateUUID();
    localStorage.setItem(LS_DEVICE_ID, id);
  }
  return id;
}

function getSessionId(): string {
  if (!_sessionId) {
    _sessionId = generateUUID();
  }
  return _sessionId;
}

// ---------------------------------------------------------------------------
// Preference resolution
// ---------------------------------------------------------------------------

/** Check if telemetry is disabled by an environment variable (Electron only). */
export function isEnvTelemetryDisabled(): boolean {
  try {
    return window.scope?.getEnvTelemetryDisabled?.() === true;
  } catch {
    return false;
  }
}

/**
 * Resolve whether telemetry is enabled.
 * Precedence: SCOPE_TELEMETRY_DISABLED > DO_NOT_TRACK > UI setting > default OFF (opt-in).
 */
export function getTelemetryEnabled(): boolean {
  if (isEnvTelemetryDisabled()) return false;
  const stored = localStorage.getItem(LS_TELEMETRY_ENABLED);
  if (stored === "true") return true;
  return false;
}

/** Persist the telemetry preference and update provider state. */
export function setTelemetryEnabled(enabled: boolean): void {
  localStorage.setItem(LS_TELEMETRY_ENABLED, String(enabled));
  if (!_initialized) return;
  if (enabled) {
    _provider.optIn();
  } else {
    _provider.optOut();
  }
}

// ---------------------------------------------------------------------------
// Disclosure state
// ---------------------------------------------------------------------------

export function isDisclosed(): boolean {
  return localStorage.getItem(LS_TELEMETRY_DISCLOSED) === "true";
}

export function markDisclosed(): void {
  localStorage.setItem(LS_TELEMETRY_DISCLOSED, "true");
}

// ---------------------------------------------------------------------------
// Event queue (pre-disclosure)
// ---------------------------------------------------------------------------

export function flushQueue(): void {
  if (!_initialized || !getTelemetryEnabled()) {
    _eventQueue = [];
    return;
  }
  for (const { event, properties } of _eventQueue) {
    _provider.track(event, properties);
  }
  _eventQueue = [];
}

export function dropQueue(): void {
  _eventQueue = [];
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

export function initTelemetry(): void {
  if (_initialized) return;

  const providerType = getProviderType();

  if (providerType === "noop") {
    _provider = new NoopProvider();
    _provider.init({ token: "" });
    _initialized = true;
  } else {
    const token = getToken(providerType);
    if (!token) {
      _provider = new NoopProvider();
      _initialized = false;
      return;
    }
    _provider = createProvider(providerType);
    _provider.init({ token, apiHost: getApiHost() });
    _initialized = true;
  }

  const deviceId = getDeviceId();

  // Register super properties (attached to every event).
  _provider.registerSuperProperties({
    app_version:
      typeof import.meta !== "undefined"
        ? ((import.meta.env?.VITE_APP_VERSION as string | undefined) ?? "")
        : "",
    platform: typeof navigator !== "undefined" ? navigator.platform : "unknown",
    session_id: getSessionId(),
    device_id: deviceId,
  });

  // If telemetry was explicitly disabled (user previously declined or toggled
  // off in settings), opt out. Don't opt out for fresh installs where the user
  // hasn't made a choice yet.
  const stored = localStorage.getItem(LS_TELEMETRY_ENABLED);
  if (stored === "false" || isEnvTelemetryDisabled()) {
    _provider.optOut();
  }
}

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------

/**
 * Track an event. If the disclosure hasn't been shown yet, events are
 * queued in memory and flushed (or dropped) once the user responds.
 *
 * With opt-in, telemetry is OFF by default. Pre-disclosure events are
 * still queued so they can be sent if the user opts in.
 */
export function track(
  event: string,
  properties?: Record<string, unknown>
): void {
  const props: Record<string, unknown> = {
    ...properties,
    timestamp: Date.now(),
  };

  // Queue events until the user has seen the disclosure
  if (!isDisclosed()) {
    if (_eventQueue.length < EVENT_QUEUE_CAP) {
      _eventQueue.push({ event, properties: props });
    }
    return;
  }

  // After disclosure, respect the user's choice
  if (!getTelemetryEnabled()) return;
  if (!_initialized) return;
  _provider.track(event, props);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/** Get app start time for calculating load_time_ms. */
export function getAppStartTime(): number {
  return _appStartTime;
}

/**
 * Create a debounced version of trackEvent for continuous inputs.
 * Fires at most once per `delayMs` for a given event+key combo.
 */
export function createDebouncedTracker(
  delayMs: number = 2000
): (event: string, properties?: Record<string, unknown>, key?: string) => void {
  const timers = new Map<string, ReturnType<typeof setTimeout>>();
  return (
    event: string,
    properties?: Record<string, unknown>,
    key?: string
  ) => {
    const timerKey = key ?? event;
    const existing = timers.get(timerKey);
    if (existing) clearTimeout(existing);
    timers.set(
      timerKey,
      setTimeout(() => {
        track(event, properties);
        timers.delete(timerKey);
      }, delayMs)
    );
  };
}
