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
  setOverageEnabled,
  DASHBOARD_USAGE_URL,
} from "../lib/billing";
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
  creditsPerMin: number;
  isLoading: boolean;
  billingError: boolean;
}

interface BillingContextValue extends BillingState {
  refresh: () => Promise<void>;
  openCheckout: (tier: "pro" | "max") => Promise<void>;
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

  // Warning thresholds (tracked so we only toast once per threshold)
  const creditWarningShown = useRef<"none" | "low" | "critical" | "grace">(
    "none"
  );
  const upsellShown = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const apiKey = getDaydreamAPIKey();
      if (!apiKey) {
        setState(prev => ({ ...prev, isLoading: false }));
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

  // tier param accepted for future use (tier-specific checkout pages)
  const openCheckout = useCallback(async (_tier: "pro" | "max") => {
    openExternalUrl(DASHBOARD_USAGE_URL);
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
        toggleOverage,
        showPaywall,
        setShowPaywall,
        paywallReason,
        setPaywallReason,
      }}
    >
      {children}
    </BillingContext.Provider>
  );
}
