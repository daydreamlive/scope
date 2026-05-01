/**
 * Survey eligibility logic and localStorage accessors.
 *
 * Two independent surveys:
 * - Sean Ellis (PMF snapshot): one-shot, gated on 45 days since first meaningful
 *   use AND active within the last 14 days.
 * - NPS (recurring pulse): shown no more than every 90 days, gated on 60 days
 *   since first meaningful use AND active within the last 14 days.
 *
 * "Meaningful use" is defined as a stream session that ran at least
 * MEANINGFUL_USE_MIN_SECONDS seconds.
 */

export const MEANINGFUL_USE_MIN_SECONDS = 30;

export const SEAN_ELLIS_MIN_AGE_DAYS = 45;
export const SEAN_ELLIS_ACTIVE_WINDOW_DAYS = 14;

export const NPS_MIN_AGE_DAYS = 60;
export const NPS_ACTIVE_WINDOW_DAYS = 14;
export const NPS_COOLDOWN_DAYS = 90;

export const POST_STREAM_DELAY_MS = 750;

const STORAGE_KEYS = {
  firstMeaningfulUseAt: "scope_first_meaningful_use_at",
  lastMeaningfulUseAt: "scope_last_meaningful_use_at",
  seanEllisShownAt: "scope_sean_ellis_shown_at",
  npsLastShownAt: "scope_nps_last_shown_at",
} as const;

// ---------------------------------------------------------------------------
// Safe localStorage helpers (Safari private mode / quota can throw)
// ---------------------------------------------------------------------------

function safeGetItem(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSetItem(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Ignore — quota / private mode. Survey just won't gate correctly, which
    // is acceptable (at worst the user sees it again).
  }
}

function readTimestamp(key: string): number | null {
  const raw = safeGetItem(key);
  if (!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

// ---------------------------------------------------------------------------
// Meaningful use tracking
// ---------------------------------------------------------------------------

export function recordMeaningfulUse(now: number = Date.now()): void {
  const nowStr = String(now);
  if (!safeGetItem(STORAGE_KEYS.firstMeaningfulUseAt)) {
    safeSetItem(STORAGE_KEYS.firstMeaningfulUseAt, nowStr);
  }
  safeSetItem(STORAGE_KEYS.lastMeaningfulUseAt, nowStr);
}

export function getFirstMeaningfulUseAt(): number | null {
  return readTimestamp(STORAGE_KEYS.firstMeaningfulUseAt);
}

export function getLastMeaningfulUseAt(): number | null {
  return readTimestamp(STORAGE_KEYS.lastMeaningfulUseAt);
}

// ---------------------------------------------------------------------------
// Shown-flag accessors
// ---------------------------------------------------------------------------

export function getSeanEllisShownAt(): number | null {
  return readTimestamp(STORAGE_KEYS.seanEllisShownAt);
}

export function markSeanEllisShown(now: number = Date.now()): void {
  safeSetItem(STORAGE_KEYS.seanEllisShownAt, String(now));
}

export function getNpsLastShownAt(): number | null {
  return readTimestamp(STORAGE_KEYS.npsLastShownAt);
}

export function markNpsShown(now: number = Date.now()): void {
  safeSetItem(STORAGE_KEYS.npsLastShownAt, String(now));
}

// ---------------------------------------------------------------------------
// Eligibility rules (pure)
// ---------------------------------------------------------------------------

const DAY_MS = 24 * 60 * 60 * 1000;

export interface EligibilityInputs {
  telemetryEnabled: boolean;
  firstMeaningfulUseAt: number | null;
  lastMeaningfulUseAt: number | null;
  now?: number;
}

export function isSeanEllisEligible(
  inputs: EligibilityInputs & { seanEllisShownAt: number | null }
): boolean {
  const now = inputs.now ?? Date.now();
  if (!inputs.telemetryEnabled) return false;
  if (inputs.seanEllisShownAt !== null) return false;
  if (inputs.firstMeaningfulUseAt === null) return false;
  if (inputs.lastMeaningfulUseAt === null) return false;
  if (now - inputs.firstMeaningfulUseAt < SEAN_ELLIS_MIN_AGE_DAYS * DAY_MS)
    return false;
  if (now - inputs.lastMeaningfulUseAt > SEAN_ELLIS_ACTIVE_WINDOW_DAYS * DAY_MS)
    return false;
  return true;
}

export function isNpsEligible(
  inputs: EligibilityInputs & { npsLastShownAt: number | null }
): boolean {
  const now = inputs.now ?? Date.now();
  if (!inputs.telemetryEnabled) return false;
  if (inputs.firstMeaningfulUseAt === null) return false;
  if (inputs.lastMeaningfulUseAt === null) return false;
  if (now - inputs.firstMeaningfulUseAt < NPS_MIN_AGE_DAYS * DAY_MS)
    return false;
  if (now - inputs.lastMeaningfulUseAt > NPS_ACTIVE_WINDOW_DAYS * DAY_MS)
    return false;
  if (
    inputs.npsLastShownAt !== null &&
    now - inputs.npsLastShownAt < NPS_COOLDOWN_DAYS * DAY_MS
  )
    return false;
  return true;
}
