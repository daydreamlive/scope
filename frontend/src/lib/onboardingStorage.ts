/**
 * Onboarding persistence helpers
 *
 * Primary storage: backend file at ~/.daydream-scope/onboarding.json via
 * GET/PUT /api/v1/onboarding/status. Survives reinstalls and cache clears.
 *
 * localStorage is used as a fast synchronous cache so the initial render
 * doesn't flash the onboarding overlay while waiting for the API response.
 * The async `fetchOnboardingStatus()` is the source of truth and should be
 * called once on app boot to reconcile.
 */

const CACHE_KEY = "scope_onboarding_completed";

// ---------------------------------------------------------------------------
// Synchronous cache (for initial render before API responds)
// ---------------------------------------------------------------------------

/** Synchronous check — reads the localStorage cache. */
export function isOnboardingCompletedSync(): boolean {
  try {
    return localStorage.getItem(CACHE_KEY) === "true";
  } catch {
    return false;
  }
}

function cacheCompleted(completed: boolean): void {
  try {
    if (completed) {
      localStorage.setItem(CACHE_KEY, "true");
    } else {
      localStorage.removeItem(CACHE_KEY);
    }
  } catch {
    // no-op
  }
}

// ---------------------------------------------------------------------------
// Async API helpers (source of truth)
// ---------------------------------------------------------------------------

interface OnboardingStatus {
  completed: boolean;
  inference_mode: string | null;
}

/** Fetch onboarding status from the backend. Updates the localStorage cache. */
export async function fetchOnboardingStatus(): Promise<OnboardingStatus> {
  try {
    const res = await fetch("/api/v1/onboarding/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: OnboardingStatus = await res.json();
    cacheCompleted(data.completed);
    return data;
  } catch {
    // If the API is unreachable (e.g. dev mode without backend), fall back to cache
    return { completed: isOnboardingCompletedSync(), inference_mode: null };
  }
}

/** Mark onboarding as completed on the backend + local cache. */
export async function markOnboardingCompleted(): Promise<void> {
  cacheCompleted(true);
  try {
    await fetch("/api/v1/onboarding/status", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ completed: true }),
    });
  } catch {
    // Cache is already set — worst case the backend catches up next call
  }
}

/** Persist the inference mode chosen during onboarding. */
export async function setInferenceMode(mode: "local" | "cloud"): Promise<void> {
  try {
    await fetch("/api/v1/onboarding/status", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        completed: isOnboardingCompletedSync(),
        inference_mode: mode,
      }),
    });
  } catch {
    // no-op
  }
}

/**
 * Reset onboarding so it shows on next launch.
 * Used by Settings → Advanced → "Show onboarding again".
 */
export async function resetOnboarding(): Promise<void> {
  cacheCompleted(false);
  try {
    await fetch("/api/v1/onboarding/status", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ completed: false }),
    });
  } catch {
    // no-op
  }
}
