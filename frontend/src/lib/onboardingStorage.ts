/**
 * Onboarding persistence helpers
 *
 * Storage: backend file at ~/.daydream-scope/onboarding.json via
 * GET/PUT /api/v1/onboarding/status. Survives reinstalls and cache clears.
 */

// ---------------------------------------------------------------------------
// Async API helpers (source of truth)
// ---------------------------------------------------------------------------

interface OnboardingStatus {
  completed: boolean;
  inference_mode: string | null;
  onboarding_style?: "teaching" | "simple" | null;
  referral_source?: string | null;
  use_case?: string | null;
}

/** Fetch onboarding status from the backend. */
export async function fetchOnboardingStatus(): Promise<OnboardingStatus> {
  try {
    const res = await fetch("/api/v1/onboarding/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    // If the API is unreachable (e.g. dev mode without backend), assume not completed
    return { completed: false, inference_mode: null };
  }
}

/** Mark onboarding as completed on the backend. */
export async function markOnboardingCompleted(): Promise<void> {
  try {
    await fetch("/api/v1/onboarding/status", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ completed: true }),
    });
  } catch {
    // no-op
  }
}

/** Persist the inference mode chosen during onboarding. */
export async function setInferenceMode(mode: "local" | "cloud"): Promise<void> {
  try {
    await fetch("/api/v1/onboarding/status", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inference_mode: mode }),
    });
  } catch {
    // no-op
  }
}

/** Persist survey answers collected during cloud connecting. */
export async function persistSurveyAnswers(answers: {
  referral_source: string | null;
  use_case: string | null;
}): Promise<void> {
  try {
    await fetch("/api/v1/onboarding/status", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(answers),
    });
  } catch {
    // no-op
  }
}

/** Persist the onboarding style chosen during the universal style step. */
export async function persistOnboardingStyle(
  style: "teaching" | "simple"
): Promise<void> {
  try {
    await fetch("/api/v1/onboarding/status", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ onboarding_style: style }),
    });
  } catch {
    // no-op
  }
}

/** Fire-and-forget: tell the backend to connect to the cloud relay. */
export async function activateCloudRelay(userId: string | null): Promise<void> {
  if (!userId) return;
  try {
    await fetch("/api/v1/cloud/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId }),
    });
  } catch (err) {
    console.error("[Onboarding] Failed to auto-connect to cloud:", err);
  }
}

/**
 * Reset onboarding so it shows on next launch.
 * Used by Settings → Advanced → "Show onboarding again".
 */
export async function resetOnboarding(): Promise<void> {
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
