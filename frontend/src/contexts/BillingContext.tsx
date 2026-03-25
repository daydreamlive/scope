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
  createCheckoutSession,
  createPortalSession,
  setOverageEnabled,
  requestInferenceToken,
} from "../lib/billing";
import { getDaydreamAPIKey } from "../lib/auth";
import { getDeviceId } from "../lib/deviceId";
import { useCloudStatus } from "../hooks/useCloudStatus";
import { toast } from "sonner";

export interface BillingState {
  tier: "free" | "basic" | "pro";
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
  isLoading: boolean;
}

interface BillingContextValue extends BillingState {
  refresh: () => Promise<void>;
  openCheckout: (tier: "basic" | "pro") => Promise<void>;
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
  isLoading: true,
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

function openExternalUrl(url: string) {
  if (typeof window !== "undefined" && "scope" in window) {
    (
      window as unknown as { scope: { openExternal: (url: string) => void } }
    ).scope.openExternal(url);
  } else {
    window.open(url, "_blank");
  }
}

export function BillingProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<BillingState>({
    tier: "free",
    credits: null,
    subscription: null,
    trial: null,
    creditsPerMin: 7.5,
    isLoading: true,
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
  const creditWarningShown = useRef<"none" | "low" | "critical">("none");
  const trialWarningShown = useRef<"none" | "2min" | "30sec">("none");

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
        }));
        return;
      }

      const deviceId = getDeviceId();
      const data = await fetchCreditsBalance(apiKey, deviceId);
      setState({
        tier: data.tier as "free" | "basic" | "pro",
        credits: data.credits,
        subscription: data.subscription,
        trial: data.trial,
        creditsPerMin: data.creditsPerMin,
        isLoading: false,
      });
    } catch (err) {
      console.error("[Billing] Failed to refresh:", err);
      setState(prev => ({ ...prev, isLoading: false }));
    }
  }, []);

  // Poll balance every 15s when cloud-connected
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

  // Low credit warnings — toast once per threshold
  useEffect(() => {
    if (!isConnected || !state.credits || state.tier === "free") {
      creditWarningShown.current = "none";
      return;
    }
    const { balance, periodCredits } = state.credits;
    const pct = periodCredits > 0 ? balance / periodCredits : 1;

    if (pct <= 0.05 && creditWarningShown.current !== "critical") {
      creditWarningShown.current = "critical";
      toast.warning(
        `Credits critically low — ${Math.round(balance)} credits remaining (${state.creditsPerMin} credits/min)`,
        { duration: 10000 }
      );
    } else if (
      pct <= 0.15 &&
      pct > 0.05 &&
      creditWarningShown.current === "none"
    ) {
      creditWarningShown.current = "low";
      toast.warning(
        `Credits running low — ${Math.round(balance)} credits remaining`
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

  const openCheckout = useCallback(async (tier: "basic" | "pro") => {
    try {
      const apiKey = getDaydreamAPIKey();
      if (!apiKey) {
        toast.error("Please sign in first");
        return;
      }
      const { checkoutUrl } = await createCheckoutSession(apiKey, tier);
      openExternalUrl(checkoutUrl);
      toast.info("Opening Stripe Checkout in your browser...");
    } catch (err) {
      console.error("[Billing] Checkout failed:", err);
      toast.error("Failed to open checkout. Please try again.");
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
      toast.error("Failed to open subscription management.");
    }
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
        toast.error("Failed to update overage setting.");
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
