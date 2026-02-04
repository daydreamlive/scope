/**
 * Get the redirect URL for OAuth callback (the current origin)
 */
function getRedirectUrl(): string {
  if (typeof window !== "undefined") {
    return window.location.origin;
  }
  // Fallback for SSR or non-browser environments
  return "http://localhost:8000";
}

// const DAYDREAM_AUTH_URL = `https://app.livepeer.monster/sign-in/local?redirect_url=${encodeURIComponent(getRedirectUrl())}`;
const DAYDREAM_AUTH_URL = `https://streamdiffusion-git-mh-signin.preview.livepeer.monster/sign-in/local?redirect_url=${encodeURIComponent(getRedirectUrl())}`;
const DAYDREAM_API_BASE =
  (import.meta.env.VITE_DAYDREAM_API_BASE as string | undefined) ||
  "https://api.daydream.monster";
const API_KEY_STORAGE_KEY = "daydream_api_key";
const USER_ID_STORAGE_KEY = "daydream_user_id";

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
 * Save the Daydream auth credentials to localStorage
 */
export function saveDaydreamAuth(apiKey: string, userId: string | null): void {
  localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
  if (userId) {
    localStorage.setItem(USER_ID_STORAGE_KEY, userId);
  }
  console.log("Saved Daydream auth to localStorage");
  // Dispatch custom event to notify components of auth state change
  window.dispatchEvent(new CustomEvent("daydream-auth-change"));
}

/**
 * Clear the stored Daydream auth credentials
 */
export function clearDaydreamAuth(): void {
  localStorage.removeItem(API_KEY_STORAGE_KEY);
  localStorage.removeItem(USER_ID_STORAGE_KEY);
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
 */
export function redirectToSignIn(): void {
  window.location.href = DAYDREAM_AUTH_URL;
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
