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
const AUTH_STORAGE_KEY = "daydream_auth";
const AUTH_STATE_KEY = "daydream_auth_state";

/**
 * Generate a random state string for OAuth CSRF protection
 */
function generateAuthState(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return Array.from(array, byte => byte.toString(16).padStart(2, "0")).join("");
}

/**
 * Store the OAuth state in sessionStorage
 */
function storeAuthState(state: string): void {
  sessionStorage.setItem(AUTH_STATE_KEY, state);
}

/**
 * Verify and consume the OAuth state from sessionStorage
 * Returns true if state matches, false otherwise
 */
function verifyAuthState(state: string | null): boolean {
  const storedState = sessionStorage.getItem(AUTH_STATE_KEY);
  // Clear the stored state regardless of match (one-time use)
  sessionStorage.removeItem(AUTH_STATE_KEY);

  if (!state || !storedState) {
    return false;
  }
  return state === storedState;
}

interface DaydreamAuthData {
  apiKey: string;
  userId: string | null;
  displayName: string | null;
  cohortParticipant: boolean;
  isAdmin: boolean;
}

/**
 * Get the stored auth data from localStorage
 */
function getAuthData(): DaydreamAuthData | null {
  const data = localStorage.getItem(AUTH_STORAGE_KEY);
  if (!data) return null;
  try {
    return JSON.parse(data) as DaydreamAuthData;
  } catch {
    return null;
  }
}

/**
 * Save auth data to localStorage
 */
function setAuthData(data: DaydreamAuthData): void {
  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(data));
  window.dispatchEvent(new CustomEvent("daydream-auth-change"));
}

interface UserProfile {
  displayName: string | null;
  cohortParticipant: boolean;
  isAdmin: boolean;
}

/**
 * Fetch user profile from API
 */
async function fetchUserProfile(apiKey: string): Promise<UserProfile> {
  const response = await fetch(`${DAYDREAM_API_BASE}/users/profile`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch profile: ${response.status}`);
  }
  const profile = await response.json();
  return {
    displayName: profile.email || profile.name || profile.username || null,
    cohortParticipant: profile.cohortParticipant === true,
    isAdmin: profile.isAdmin === true,
  };
}

/**
 * Get the stored Daydream API key from localStorage or environment variable
 */
export function getDaydreamAPIKey(): string | null {
  // First check localStorage for a user-authenticated key
  const authData = getAuthData();
  if (authData?.apiKey) {
    return authData.apiKey;
  }

  // Fall back to environment variable if available
  const envKey = import.meta.env.VITE_DAYDREAM_API_KEY as string | undefined;
  return envKey || null;
}

/**
 * Get the stored Daydream user ID from localStorage
 */
export function getDaydreamUserId(): string | null {
  return getAuthData()?.userId ?? null;
}

/**
 * Get the stored Daydream user display name from localStorage
 */
export function getDaydreamUserDisplayName(): string | null {
  return getAuthData()?.displayName ?? null;
}

/**
 * Save the Daydream auth credentials and profile to localStorage
 */
export async function saveDaydreamAuth(
  apiKey: string,
  userId: string | null
): Promise<void> {
  try {
    const profile = await fetchUserProfile(apiKey);
    setAuthData({
      apiKey,
      userId,
      ...profile,
    });
  } catch (e) {
    // If profile fetch fails, save auth with defaults
    console.error("Failed to fetch user profile during auth:", e);
    setAuthData({
      apiKey,
      userId,
      displayName: null,
      cohortParticipant: false,
      isAdmin: false,
    });
  }
}

/**
 * Refresh user profile in localStorage (for existing auth)
 */
export async function refreshUserProfile(): Promise<void> {
  const authData = getAuthData();
  if (!authData) return;

  try {
    const profile = await fetchUserProfile(authData.apiKey);
    setAuthData({ ...authData, ...profile });
  } catch (e) {
    console.error("Failed to refresh user profile:", e);
  }
}

/**
 * Clear the stored Daydream auth credentials
 */
export function clearDaydreamAuth(): void {
  localStorage.removeItem(AUTH_STORAGE_KEY);
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
  // Generate and store state for CSRF protection
  const state = generateAuthState();
  storeAuthState(state);

  const authUrl = `${DAYDREAM_AUTH_URL}?redirect_url=${encodeURIComponent(getRedirectUrl())}&state=${encodeURIComponent(state)}&utm_source=scope`;

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
  // Error handling (toast + open account tab) is done in App.tsx via daydream-auth-error event
  const response = await fetch(`${DAYDREAM_API_BASE}/v1/api-key`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      name: "dd_scope_cloud",
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
  const state = urlParams.get("state");
  const userId = urlParams.get("userId");

  if (!token) {
    return false;
  }

  // Verify the state parameter to prevent CSRF attacks
  if (!verifyAuthState(state)) {
    // Clean up URL even on state mismatch
    const url = new URL(window.location.href);
    url.searchParams.delete("token");
    url.searchParams.delete("state");
    url.searchParams.delete("userId");
    window.history.replaceState({}, document.title, url.toString());

    throw new Error(
      "Invalid auth state. This may be a CSRF attack or the auth session expired. Please try signing in again."
    );
  }

  try {
    // Exchange the short-lived token for a long-lived API key
    const apiKey = await exchangeTokenForAPIKey(token);

    // Save auth credentials and fetch profile in one operation
    await saveDaydreamAuth(apiKey, userId);

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
  state: string | null;
}): Promise<void> {
  // Verify the state parameter to prevent CSRF attacks
  if (!verifyAuthState(data.state)) {
    throw new Error(
      "Invalid auth state. This may be a CSRF attack or the auth session expired. Please try signing in again."
    );
  }

  // Exchange the short-lived token for a long-lived API key
  const apiKey = await exchangeTokenForAPIKey(data.token);

  // Save auth credentials and fetch profile in one operation
  await saveDaydreamAuth(apiKey, data.userId);
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
          callback: (data: {
            token: string;
            userId: string | null;
            state: string | null;
          }) => void
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
