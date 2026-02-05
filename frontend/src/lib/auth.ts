/**
 * Check if running in Electron desktop app
 */
function isElectron(): boolean {
  return typeof window !== "undefined" && "scope" in window;
}

/**
 * Get the redirect URL for OAuth callback
 * - In Electron: uses deep link protocol for callback
 * - In browser: uses current origin for HTTP redirect
 */
function getRedirectUrl(): string {
  if (isElectron()) {
    // Use deep link protocol for Electron callback
    return "daydream-scope://auth-callback";
  }
  if (typeof window !== "undefined") {
    return window.location.origin;
  }
  // Fallback for SSR or non-browser environments
  return "http://localhost:8000";
}

const DAYDREAM_AUTH_URL =
  (import.meta.env.VITE_DAYDREAM_AUTH_URL as string | undefined) ||
  `https://app.daydream.live/sign-in/local`;
const DAYDREAM_API_BASE =
  (import.meta.env.VITE_DAYDREAM_API_BASE as string | undefined) ||
  "https://api.daydream.live";
const API_KEY_STORAGE_KEY = "daydream_api_key";
const USER_ID_STORAGE_KEY = "daydream_user_id";
const USER_DISPLAY_NAME_STORAGE_KEY = "daydream_user_display_name";

/**
 * Get the stored Daydream API key from localStorage or environment variable
 */
export function getDaydreamAPIKey(): string | null {
  // First check localStorage for a user-authenticated key
  const storedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
  if (storedKey) {
    return storedKey;
  }

  // Fall back to environment variable if available
  const envKey = import.meta.env.VITE_DAYDREAM_API_KEY as string | undefined;
  return envKey || null;
}

/**
 * Get the stored Daydream user ID from localStorage
 */
export function getDaydreamUserId(): string | null {
  return localStorage.getItem(USER_ID_STORAGE_KEY);
}

/**
 * Get the stored Daydream user display name from localStorage
 */
export function getDaydreamUserDisplayName(): string | null {
  return localStorage.getItem(USER_DISPLAY_NAME_STORAGE_KEY);
}

/**
 * Save the Daydream auth credentials to localStorage
 */
export function saveDaydreamAuth(apiKey: string, userId: string | null): void {
  localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
  if (userId) {
    localStorage.setItem(USER_ID_STORAGE_KEY, userId);
  }
  // Dispatch custom event to notify components of auth state change
  window.dispatchEvent(new CustomEvent("daydream-auth-change"));
}

/**
 * Fetch user profile and store display name in localStorage
 */
export async function fetchAndStoreUserProfile(apiKey: string): Promise<void> {
  try {
    const response = await fetch(`${DAYDREAM_API_BASE}/users/profile`, {
      headers: { Authorization: `Bearer ${apiKey}` },
    });
    if (response.ok) {
      const profile = await response.json();
      const displayName = profile.email || profile.name || profile.username;
      if (displayName) {
        localStorage.setItem(USER_DISPLAY_NAME_STORAGE_KEY, displayName);
        window.dispatchEvent(new CustomEvent("daydream-auth-change"));
      }
    }
  } catch (e) {
    console.error("Failed to fetch user profile:", e);
  }
}

/**
 * Clear the stored Daydream auth credentials
 */
export function clearDaydreamAuth(): void {
  localStorage.removeItem(API_KEY_STORAGE_KEY);
  localStorage.removeItem(USER_ID_STORAGE_KEY);
  localStorage.removeItem(USER_DISPLAY_NAME_STORAGE_KEY);
  // Dispatch custom event to notify components of auth state change
  window.dispatchEvent(new CustomEvent("daydream-auth-change"));
}

/**
 * Check if user is authenticated (has an API key)
 */
export function isAuthenticated(): boolean {
  return getDaydreamAPIKey() !== null && getDaydreamUserId() !== null;
}

/**
 * Redirect to Daydream sign-in page
 * - In Electron: opens in system browser via IPC
 * - In browser: navigates directly
 */
export function redirectToSignIn(): void {
  const authUrl = `${DAYDREAM_AUTH_URL}?redirect_url=${encodeURIComponent(getRedirectUrl())}`;

  if (isElectron()) {
    // Open in system browser via Electron IPC
    (
      window as unknown as {
        scope: { openExternal: (url: string) => Promise<boolean> };
      }
    ).scope
      .openExternal(authUrl)
      .catch(err => {
        console.error("Failed to open auth URL in browser:", err);
      });
  } else {
    // Standard browser navigation
    window.location.href = authUrl;
  }
}

/**
 * Exchange a short-lived token for a long-lived API key
 */
export async function exchangeTokenForAPIKey(token: string): Promise<string> {
  // TODO if this fails, log out and redirect to sign in page
  // TODO check state queryparam
  const response = await fetch(`${DAYDREAM_API_BASE}/v1/api-key`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      name: "scope",
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Failed to exchange token for API key: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();

  // The API should return an object with an api_key field
  // Adjust this based on the actual API response structure
  if (!result.api_key && !result.apiKey && !result.key) {
    throw new Error("API response did not contain an API key");
  }

  return result.api_key || result.apiKey || result.key;
}

/**
 * Handle OAuth callback - extract token from URL and exchange it for API key
 * Returns true if callback was handled, false otherwise
 */
export async function handleOAuthCallback(): Promise<boolean> {
  const urlParams = new URLSearchParams(window.location.search);
  const token = urlParams.get("token");
  const userId = urlParams.get("userId");

  if (!token) {
    return false;
  }

  try {
    // Exchange the short-lived token for a long-lived API key
    const apiKey = await exchangeTokenForAPIKey(token);

    // Save the API key and userId to localStorage
    saveDaydreamAuth(apiKey, userId);

    // Fetch and store user profile
    await fetchAndStoreUserProfile(apiKey);

    // Clean up the URL by removing the token parameter
    const url = new URL(window.location.href);
    url.searchParams.delete("token");
    url.searchParams.delete("state");
    url.searchParams.delete("userId");
    window.history.replaceState({}, document.title, url.toString());

    return true;
  } catch (error) {
    console.error("Failed to exchange token for API key:", error);
    throw error;
  }
}

/**
 * Process auth callback data (used by Electron deep link handler)
 */
async function processAuthCallback(data: {
  token: string;
  userId: string | null;
}): Promise<void> {
  // Exchange the short-lived token for a long-lived API key
  const apiKey = await exchangeTokenForAPIKey(data.token);

  // Save the API key and userId to localStorage
  saveDaydreamAuth(apiKey, data.userId);

  // Fetch and store user profile
  await fetchAndStoreUserProfile(apiKey);
}

/**
 * Initialize Electron auth callback listener
 * Call this once when the app starts if running in Electron
 * Returns a cleanup function
 *
 * @param onSuccess - Optional callback when auth succeeds
 * @param onError - Optional callback when auth fails
 */
export function initElectronAuthListener(
  onSuccess?: () => void,
  onError?: (error: Error) => void
): (() => void) | null {
  if (!isElectron()) {
    return null;
  }

  const scopeApi = (
    window as unknown as {
      scope: {
        onAuthCallback: (
          callback: (data: { token: string; userId: string | null }) => void
        ) => () => void;
      };
    }
  ).scope;

  // Set up the auth callback listener
  const cleanup = scopeApi.onAuthCallback(data => {
    processAuthCallback(data)
      .then(() => {
        onSuccess?.();
      })
      .catch(err => {
        console.error("Failed to process Electron auth callback:", err);
        onError?.(err instanceof Error ? err : new Error(String(err)));
      });
  });

  return cleanup;
}
