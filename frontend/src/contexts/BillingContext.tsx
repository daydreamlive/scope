import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
} from "react";
import {
  fetchCreditsBalance,
  sendTrialHeartbeat,
  createPortalSession,
  setOverageEnabled,
  requestInferenceToken,
} from "../lib/billing";

const SUBSCRIBE_URL = "https://app.daydream.live/dashboard/usage";
import { getDaydreamAPIKey } from "../lib/auth";
import { getDeviceId } from "../lib/deviceId";
import { openExternalUrl } from "../lib/openExternal";
import { useCloudStatus } from "../hooks/useCloudStatus";
import { toast } from "sonner";

// Default GPU type used for rate lookups when the cloud backend doesn't expose
// one. Scope cloud streams currently default to h100, the highest tier; using
// it keeps the displayed cost a safe upper bound.
const DEFAULT_GPU_TYPE = "h100";

export interface BillingState {
  tier: "free" | "pro" | "max";
  credits: { balance: number; periodCredits: number } | null;
  subscription: {
    status: string;
    currentPeriodEnd: string;
    cancelAtPeriodEnd: boolean;
    overageEnabled: boolean;
  } | null;
  trial: {
    secondsUsed: number;
    secondsLimit: number;
    exhausted: boolean;
  } | null;
  creditsPerMin: number;
  allRates: Record<string, number> | null;
  isLoading: boolean;
  billingError: boolean;
}

interface BillingContextValue extends BillingState {
  refresh: () => Promise<void>;
  openCheckout: (tier: "pro" | "max") => Promise<void>;
  openPortal: () => Promise<void>;
  toggleOverage: (enabled: boolean) => Promise<void>;
  showPaywall: boolean;
  setShowPaywall: (show: boolean) => void;
  paywallReason: "trial_exhausted" | "credits_exhausted" | "subscribe" | null;
  setPaywallReason: (
    reason: "trial_exhausted" | "credits_exhausted" | "subscribe" | null
  ) => void;
  /** Get a valid inference token (requests new one if expired) */
  getInferenceToken: () => Promise<string | null>;
}

const defaultState: BillingContextValue = {
  tier: "free",
  credits: null,
  subscription: null,
  trial: null,
  creditsPerMin: 7.5,
  allRates: null,
  isLoading: true,
  billingError: false,
  refresh: async () => {},
  openCheckout: async () => {},
  openPortal: async () => {},
  toggleOverage: async () => {},
  showPaywall: false,
  setShowPaywall: () => {},
  paywallReason: null,
  setPaywallReason: () => {},
  getInferenceToken: async () => null,
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
    trial: null,
    creditsPerMin: 7.5,
    allRates: null,
    isLoading: true,
    billingError: false,
  });
  const [showPaywall, setShowPaywall] = useState(false);
  const [paywallReason, setPaywallReason] = useState<
    "trial_exhausted" | "credits_exhausted" | "subscribe" | null
  >(null);

  const { isConnected } = useCloudStatus();
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const heartbeatRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const prevTrialExhausted = useRef(false);

  // Inference token cache — refresh before 5-min expiry
  const inferenceTokenRef = useRef<{
    token: string;
    expiresAt: number;
  } | null>(null);

  // Warning thresholds (tracked so we only toast once per threshold)
  const creditWarningShown = useRef<"none" | "low" | "critical" | "grace">(
    "none"
  );
  const trialWarningShown = useRef<"none" | "2min" | "30sec">("none");
  const upsellShown = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const apiKey = getDaydreamAPIKey();
      if (!apiKey) {
        // Not authenticated — try trial status
        const deviceId = getDeviceId();
        const { secondsUsed, secondsLimit, exhausted } =
          await sendTrialHeartbeat(null, deviceId);
        setState(prev => ({
          ...prev,
          tier: "free",
          credits: null,
          subscription: null,
          trial: { secondsUsed, secondsLimit, exhausted },
          isLoading: false,
          billingError: false,
        }));
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
        : (rawRate as number);

      setState({
        tier: data.tier,
        credits: data.credits,
        subscription: data.subscription,
        trial: data.trial,
        creditsPerMin: scopeRate,
        allRates: rateMap,
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

  // Trial heartbeat every 60s when streaming on free trial
  useEffect(() => {
    if (
      isConnected &&
      state.tier === "free" &&
      state.trial &&
      !state.trial.exhausted
    ) {
      heartbeatRef.current = setInterval(async () => {
        try {
          const apiKey = getDaydreamAPIKey();
          const deviceId = getDeviceId();
          const result = await sendTrialHeartbeat(apiKey, deviceId);
          setState(prev => ({
            ...prev,
            trial: {
              secondsUsed: result.secondsUsed,
              secondsLimit: result.secondsLimit,
              exhausted: result.exhausted,
            },
          }));
        } catch (err) {
          console.error("[Billing] Trial heartbeat failed:", err);
        }
      }, 60_000);
    } else {
      if (heartbeatRef.current) clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
    return () => {
      if (heartbeatRef.current) clearInterval(heartbeatRef.current);
    };
  }, [isConnected, state.tier, state.trial?.exhausted]);

  // Show paywall when trial exhausts
  useEffect(() => {
    if (state.trial?.exhausted && !prevTrialExhausted.current) {
      setPaywallReason("trial_exhausted");
      setShowPaywall(true);
    }
    prevTrialExhausted.current = state.trial?.exhausted ?? false;
  }, [state.trial?.exhausted]);

  // Low credit warnings — toast once per threshold, with grace period warning
  useEffect(() => {
    if (!isConnected || !state.credits || state.tier === "free") {
      creditWarningShown.current = "none";
      upsellShown.current = false;
      return;
    }
    const { balance, periodCredits } = state.credits;
    const pct = periodCredits > 0 ? balance / periodCredits : 1;
    const minutesLeft =
      state.creditsPerMin > 0 ? Math.round(balance / state.creditsPerMin) : 0;

    // Grace period: ~1 min of credits left — warn that stream will end soon
    if (
      minutesLeft <= 1 &&
      balance > 0 &&
      creditWarningShown.current !== "grace"
    ) {
      creditWarningShown.current = "grace";
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
    } else if (
      pct <= 0.05 &&
      creditWarningShown.current !== "critical" &&
      creditWarningShown.current !== "grace"
    ) {
      creditWarningShown.current = "critical";
      toast.warning(
        `Credits critically low — ${Math.round(balance)} credits remaining (~${minutesLeft} min)`,
        { duration: 10000 }
      );
    } else if (
      pct <= 0.15 &&
      pct > 0.05 &&
      creditWarningShown.current === "none"
    ) {
      creditWarningShown.current = "low";
      toast.warning(
        `Credits running low — ${Math.round(balance)} credits remaining (~${minutesLeft} min)`
      );
    }

    // Proactive upsell at 80% usage for Pro tier
    if (
      state.tier === "pro" &&
      pct <= 0.2 &&
      pct > 0.05 &&
      !upsellShown.current
    ) {
      upsellShown.current = true;
      toast.info(
        "Running low on credits? Upgrade to Max for more credits per month.",
        { duration: 8000 }
      );
    }
  }, [isConnected, state.credits, state.tier, state.creditsPerMin]);

  // Trial time warnings — toast at 2 min and 30 sec remaining
  useEffect(() => {
    if (!isConnected || !state.trial || state.trial.exhausted) {
      trialWarningShown.current = "none";
      return;
    }
    const remaining = state.trial.secondsLimit - state.trial.secondsUsed;

    if (remaining <= 30 && trialWarningShown.current !== "30sec") {
      trialWarningShown.current = "30sec";
      toast.warning("Free trial ending in 30 seconds. Subscribe to continue.", {
        duration: 30000,
      });
    } else if (
      remaining <= 120 &&
      remaining > 30 &&
      trialWarningShown.current === "none"
    ) {
      trialWarningShown.current = "2min";
      toast.warning(
        "Free trial ending in 2 minutes. Subscribe or switch to local inference."
      );
    }
  }, [isConnected, state.trial]);

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

  const getInferenceToken = useCallback(async (): Promise<string | null> => {
    // Return cached token if still valid (with 60s buffer)
    const cached = inferenceTokenRef.current;
    if (cached && cached.expiresAt > Date.now() + 60_000) {
      return cached.token;
    }

    try {
      const apiKey = getDaydreamAPIKey();
      if (!apiKey) return null;
      const deviceId = getDeviceId();
      const result = await requestInferenceToken(apiKey, deviceId);

      if (!result.authorized || !result.token) {
        inferenceTokenRef.current = null;
        return null;
      }

      inferenceTokenRef.current = {
        token: result.token,
        expiresAt: new Date(result.expiresAt!).getTime(),
      };
      return result.token;
    } catch (err) {
      console.error("[Billing] Failed to get inference token:", err);
      return null;
    }
  }, []);

  const openPortal = useCallback(async () => {
    try {
      const apiKey = getDaydreamAPIKey();
      if (!apiKey) {
        toast.error("Please sign in first");
        return;
      }
      const { portalUrl } = await createPortalSession(apiKey);
      openExternalUrl(portalUrl);
      toast.info("Opening subscription management in your browser...");
    } catch (err) {
      console.error("[Billing] Portal failed:", err);
      const msg = err instanceof Error ? err.message : "Unknown error";
      toast.error(`Failed to open subscription management: ${msg}`, {
        description: "If this persists, contact support@daydream.live",
      });
    }
  }, []);

  const openCheckout = useCallback(async (_tier: "pro" | "max") => {
    openExternalUrl(SUBSCRIBE_URL);
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

  return (
    <BillingContext.Provider
      value={{
        ...state,
        refresh,
        openCheckout,
        openPortal,
        toggleOverage,
        showPaywall,
        setShowPaywall,
        paywallReason,
        setPaywallReason,
        getInferenceToken,
      }}
    >
      {children}
    </BillingContext.Provider>
  );
}
