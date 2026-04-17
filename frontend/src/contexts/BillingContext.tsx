import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
  type ReactNode,
} from "react";
import {
  fetchCreditsBalance,
  setOverageEnabled,
  DASHBOARD_USAGE_URL,
} from "../lib/billing";
import {
  getDaydreamAPIKey,
  isAuthenticated,
  redirectToSignIn,
} from "../lib/auth";
import { getDeviceId } from "../lib/deviceId";
import { openExternalUrl } from "../lib/openExternal";
import { useCloudStatus } from "../hooks/useCloudStatus";
import { toast } from "sonner";

// Default GPU type used for rate lookups when the cloud backend doesn't expose
// one. Scope cloud streams currently default to h100, the highest tier; using
// it keeps the displayed cost a safe upper bound.
const DEFAULT_GPU_TYPE = "h100";

// sessionStorage key for a URL to open after the user completes OAuth. Used to
// route unauthenticated "Subscribe" / "Upgrade" clicks through sign-in and
// then land them on the intended billing page.
const POST_AUTH_URL_KEY = "scope_post_auth_url";

export interface BillingState {
  tier: "free" | "pro" | "max";
  credits: { balance: number; periodCredits: number } | null;
  subscription: {
    status: string;
    currentPeriodEnd: string;
    cancelAtPeriodEnd: boolean;
    overageEnabled: boolean;
  } | null;
  creditsPerMin: number;
  isLoading: boolean;
  billingError: boolean;
}

interface BillingContextValue extends BillingState {
  refresh: () => Promise<void>;
  openCheckout: () => Promise<void>;
  toggleOverage: (enabled: boolean) => Promise<void>;
  showPaywall: boolean;
  setShowPaywall: (show: boolean) => void;
  paywallReason: "credits_exhausted" | "subscribe" | null;
  setPaywallReason: (reason: "credits_exhausted" | "subscribe" | null) => void;
}

const defaultState: BillingContextValue = {
  tier: "free",
  credits: null,
  subscription: null,
  creditsPerMin: 7.5,
  isLoading: true,
  billingError: false,
  refresh: async () => {},
  openCheckout: async () => {},
  toggleOverage: async () => {},
  showPaywall: false,
  setShowPaywall: () => {},
  paywallReason: null,
  setPaywallReason: () => {},
};

const BillingContext = createContext<BillingContextValue>(defaultState);

export function useBilling() {
  return useContext(BillingContext);
}

export function BillingProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<BillingState>({
    tier: "free",
    credits: null,
    subscription: null,
    creditsPerMin: 7.5,
    isLoading: true,
    billingError: false,
  });
  const [showPaywall, setShowPaywall] = useState(false);
  const [paywallReason, setPaywallReason] = useState<
    "credits_exhausted" | "subscribe" | null
  >(null);

  const { isConnected } = useCloudStatus();
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const bgPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Previous warning level — toasts fire only on transitions that worsen
  // severity, so re-entering a threshold after a top-up re-arms the warning.
  const prevWarningLevelRef = useRef<"ok" | "low" | "critical" | "grace">("ok");
  // Same transition model for the pro-tier upsell toast.
  const prevUpsellLevelRef = useRef<"ok" | "low">("ok");

  const refresh = useCallback(async () => {
    try {
      const apiKey = getDaydreamAPIKey();
      if (!apiKey) {
        // Signed out (or never signed in): reset to defaults so a previous
        // user's tier/credits/subscription don't leak into the UI.
        setState({
          tier: "free",
          credits: null,
          subscription: null,
          creditsPerMin: 7.5,
          isLoading: false,
          billingError: false,
        });
        return;
      }

      const deviceId = getDeviceId();
      const data = await fetchCreditsBalance(apiKey, deviceId);
      // creditsPerMin can be a number (old API) or Record<string, number> (new API)
      const rawRate = data.creditsPerMin;
      const rateMap =
        typeof rawRate === "object" && rawRate !== null
          ? (rawRate as Record<string, number>)
          : null;
      const scopeRate = rateMap
        ? (rateMap[DEFAULT_GPU_TYPE] ?? rateMap.h100 ?? 7.5)
        : ((rawRate as number) ?? 7.5);

      setState({
        tier: data.tier,
        credits: data.credits,
        subscription: data.subscription,
        creditsPerMin: scopeRate,
        isLoading: false,
        billingError: false,
      });
    } catch (err) {
      console.error("[Billing] Failed to refresh:", err);
      setState(prev => ({ ...prev, isLoading: false, billingError: true }));
    }
  }, []);

  // Initial load + react to auth changes (sign-in / sign-out / token refresh).
  // Independent of cloud status: a signed-in user should see their plan and
  // credit balance even when they aren't actively streaming.
  useEffect(() => {
    refresh();
    const handler = () => {
      refresh();
    };
    window.addEventListener("daydream-auth-change", handler);
    window.addEventListener("daydream-auth-success", handler);
    window.addEventListener("daydream-auth-error", handler);
    return () => {
      window.removeEventListener("daydream-auth-change", handler);
      window.removeEventListener("daydream-auth-success", handler);
      window.removeEventListener("daydream-auth-error", handler);
    };
  }, [refresh]);

  // Poll balance every 15s while cloud-connected (live credit drain updates).
  useEffect(() => {
    if (isConnected) {
      refresh();
      pollRef.current = setInterval(refresh, 15_000);
    } else {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [isConnected, refresh]);

  // Background poll every 30s when authenticated but not streaming, so top-ups,
  // subscription changes, and credit deductions are reflected in the UI.
  // Pauses when the window is hidden to save resources.
  useEffect(() => {
    if (isConnected) {
      // Fast poll above handles this case — skip background poll.
      if (bgPollRef.current) clearInterval(bgPollRef.current);
      bgPollRef.current = null;
      return;
    }

    // No early apiKey check here — refresh() already no-ops when no key is
    // present, and checking at effect setup time would miss sign-ins that
    // happen after the effect runs.

    const startBgPoll = () => {
      if (bgPollRef.current) clearInterval(bgPollRef.current);
      bgPollRef.current = setInterval(refresh, 30_000);
    };

    const stopBgPoll = () => {
      if (bgPollRef.current) clearInterval(bgPollRef.current);
      bgPollRef.current = null;
    };

    const onVisibility = () => {
      if (document.hidden) {
        stopBgPoll();
      } else {
        refresh(); // Immediately refresh when tab becomes visible
        startBgPoll();
      }
    };

    if (!document.hidden) startBgPoll();
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      stopBgPoll();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [isConnected, refresh]);

  // Low credit warnings — fire only on transitions that worsen the severity
  // level. A top-up that moves the balance back above a threshold resets the
  // level, so the next drop re-arms the toast.
  useEffect(() => {
    if (!isConnected || !state.credits || state.tier === "free") {
      prevWarningLevelRef.current = "ok";
      prevUpsellLevelRef.current = "ok";
      return;
    }
    const { balance, periodCredits } = state.credits;
    const pct = periodCredits > 0 ? balance / periodCredits : 1;
    const minutesLeft =
      state.creditsPerMin > 0 ? Math.round(balance / state.creditsPerMin) : 0;

    const severity = { ok: 0, low: 1, critical: 2, grace: 3 } as const;
    let level: "ok" | "low" | "critical" | "grace" = "ok";
    if (balance > 0 && minutesLeft <= 1) level = "grace";
    else if (pct <= 0.05) level = "critical";
    else if (pct <= 0.15) level = "low";

    const prev = prevWarningLevelRef.current;
    if (severity[level] > severity[prev]) {
      if (level === "grace") {
        toast.warning(
          "Your stream will end in about 1 minute. Add credits to keep going.",
          {
            duration: 60000,
            action: {
              label: "Add Credits",
              onClick: () => {
                setPaywallReason("credits_exhausted");
                setShowPaywall(true);
              },
            },
          }
        );
      } else if (level === "critical") {
        toast.warning(
          `Credits critically low — ${Math.round(balance)} credits remaining (~${minutesLeft} min)`,
          { duration: 10000 }
        );
      } else if (level === "low") {
        toast.warning(
          `Credits running low — ${Math.round(balance)} credits remaining (~${minutesLeft} min)`
        );
      }
    }
    prevWarningLevelRef.current = level;

    // Proactive upsell at 80% usage for Pro tier — fires each time the user
    // re-enters the low bucket from ok (but not repeatedly while staying in
    // it, and not while they're already in critical/grace).
    const upsellLevel: "ok" | "low" =
      state.tier === "pro" && pct <= 0.2 && pct > 0.05 ? "low" : "ok";
    if (upsellLevel === "low" && prevUpsellLevelRef.current === "ok") {
      toast.info(
        "Running low on credits? Upgrade to Max for more credits per month.",
        { duration: 8000 }
      );
    }
    prevUpsellLevelRef.current = upsellLevel;
  }, [isConnected, state.credits, state.tier, state.creditsPerMin]);

  // Listen for credits-exhausted events from API error handling
  useEffect(() => {
    const handler = () => {
      setPaywallReason("credits_exhausted");
      setShowPaywall(true);
    };
    window.addEventListener("billing:credits-exhausted", handler);
    return () =>
      window.removeEventListener("billing:credits-exhausted", handler);
  }, []);

  const openCheckout = useCallback(async () => {
    // If the user is signed in, open the billing page directly. Otherwise,
    // stash the destination in sessionStorage and route through the OAuth
    // sign-in flow — the effect below will open the URL when auth succeeds.
    if (isAuthenticated()) {
      openExternalUrl(DASHBOARD_USAGE_URL);
      return;
    }
    try {
      sessionStorage.setItem(POST_AUTH_URL_KEY, DASHBOARD_USAGE_URL);
    } catch {
      // sessionStorage unavailable — redirect anyway so the user can sign in
    }
    redirectToSignIn();
  }, []);

  // After a successful sign-in (browser OAuth callback or Electron IPC),
  // resume any pending "open billing after login" intent exactly once.
  useEffect(() => {
    const resumePendingUrl = () => {
      try {
        const pending = sessionStorage.getItem(POST_AUTH_URL_KEY);
        if (!pending) return;
        sessionStorage.removeItem(POST_AUTH_URL_KEY);
        openExternalUrl(pending);
      } catch {
        // sessionStorage unavailable — nothing to resume
      }
    };
    window.addEventListener("daydream-auth-success", resumePendingUrl);
    // Clear any stale pending URL on sign-in errors so it doesn't fire later
    const clearPendingUrl = () => {
      try {
        sessionStorage.removeItem(POST_AUTH_URL_KEY);
      } catch {
        // ignore
      }
    };
    window.addEventListener("daydream-auth-error", clearPendingUrl);
    return () => {
      window.removeEventListener("daydream-auth-success", resumePendingUrl);
      window.removeEventListener("daydream-auth-error", clearPendingUrl);
    };
  }, []);

  const toggleOverage = useCallback(
    async (enabled: boolean) => {
      try {
        const apiKey = getDaydreamAPIKey();
        if (!apiKey) return;
        await setOverageEnabled(apiKey, enabled);
        toast.success(
          enabled ? "Overage billing enabled" : "Overage billing disabled"
        );
        await refresh();
      } catch (err) {
        console.error("[Billing] Overage toggle failed:", err);
        const msg = err instanceof Error ? err.message : "Unknown error";
        toast.error(`Failed to update overage setting: ${msg}`, {
          description: "If this persists, contact support@daydream.live",
        });
      }
    },
    [refresh]
  );

  const value = useMemo<BillingContextValue>(
    () => ({
      ...state,
      refresh,
      openCheckout,
      toggleOverage,
      showPaywall,
      setShowPaywall,
      paywallReason,
      setPaywallReason,
    }),
    [state, refresh, openCheckout, toggleOverage, showPaywall, paywallReason]
  );

  return (
    <BillingContext.Provider value={value}>{children}</BillingContext.Provider>
  );
}
