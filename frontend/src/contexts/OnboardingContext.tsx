import {
  createContext,
  useContext,
  useReducer,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import {
  fetchOnboardingStatus,
  markOnboardingCompleted,
  setInferenceMode as persistInferenceMode,
} from "../lib/onboardingStorage";
import { trackEvent } from "../lib/analytics";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type OnboardingPhase =
  | "loading" // waiting for backend status check
  | "idle" // returning user or post-completion
  | "inference" // step 1: local vs cloud
  | "cloud_auth" // step 2a: sign in (only if cloud chosen)
  | "cloud_connecting" // step 2b: waiting for cloud relay connection
  | "workflow"; // step 3: starter workflow picker

export interface OnboardingState {
  phase: OnboardingPhase;
  inferenceMode: "local" | "cloud" | null;
  onboardingStyle: "teaching" | "simple" | null;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

type OnboardingAction =
  | { type: "SELECT_INFERENCE_MODE"; mode: "local" | "cloud" }
  | { type: "COMPLETE_AUTH" }
  | { type: "CLOUD_CONNECTED" }
  | { type: "SET_ONBOARDING_STYLE"; style: "teaching" | "simple" }
  | { type: "WORKFLOW_READY" }
  | { type: "START_FROM_SCRATCH" }
  | { type: "IMPORT_WORKFLOW_READY" }
  | { type: "GO_BACK" }
  | {
      type: "LOADED";
      completed: boolean;
      onboardingStyle?: "teaching" | "simple" | null;
    };

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function reducer(
  state: OnboardingState,
  action: OnboardingAction
): OnboardingState {
  switch (action.type) {
    case "SELECT_INFERENCE_MODE":
      persistInferenceMode(action.mode);
      trackEvent("onboarding_inference_selected", { mode: action.mode });
      // Set a sessionStorage flag so we can resume after a full-page auth
      // redirect. This is consumed exactly once in the LOADED handler.
      try {
        sessionStorage.setItem("scope_onboarding_resume", action.mode);
      } catch {
        // no-op
      }
      return {
        ...state,
        inferenceMode: action.mode,
        phase: action.mode === "cloud" ? "cloud_auth" : "workflow",
      };

    case "COMPLETE_AUTH":
      trackEvent("onboarding_auth_completed");
      return { ...state, phase: "cloud_connecting" };

    case "CLOUD_CONNECTED":
      trackEvent("onboarding_cloud_connected");
      return { ...state, phase: "workflow" };

    case "SET_ONBOARDING_STYLE":
      return { ...state, onboardingStyle: action.style };

    case "WORKFLOW_READY":
      trackEvent("onboarding_workflow_downloaded");
      trackEvent("onboarding_completed");
      markOnboardingCompleted();
      return { ...state, phase: "idle" };

    case "START_FROM_SCRATCH":
      trackEvent("onboarding_started_from_scratch");
      trackEvent("onboarding_completed");
      markOnboardingCompleted();
      return { ...state, phase: "idle" };

    case "IMPORT_WORKFLOW_READY":
      trackEvent("onboarding_imported_workflow");
      trackEvent("onboarding_completed");
      markOnboardingCompleted();
      return { ...state, phase: "idle" };

    case "GO_BACK": {
      // Navigate backwards through the onboarding flow
      switch (state.phase) {
        case "cloud_auth":
          return { ...state, phase: "inference", inferenceMode: null };
        case "cloud_connecting":
          return { ...state, phase: "cloud_auth" };
        case "workflow":
          if (state.inferenceMode === "cloud")
            return { ...state, phase: "cloud_connecting" };
          return { ...state, phase: "inference", inferenceMode: null };
        default:
          return state;
      }
    }

    case "LOADED": {
      if (action.completed)
        return {
          ...state,
          phase: "idle",
          onboardingStyle: action.onboardingStyle ?? null,
        };
      // Check if we're resuming after an auth redirect (sessionStorage flag
      // is set right before the redirect and consumed here exactly once)
      const resumeMode = sessionStorage.getItem("scope_onboarding_resume");
      if (resumeMode) {
        sessionStorage.removeItem("scope_onboarding_resume");
        if (resumeMode === "cloud") {
          // Land on cloud_auth so the green-check success state shows
          // briefly before CloudAuthStep auto-advances to cloud_connecting
          return { ...state, phase: "cloud_auth", inferenceMode: "cloud" };
        }
        if (resumeMode === "local") {
          return { ...state, phase: "workflow", inferenceMode: "local" };
        }
      }
      return { ...state, phase: "inference" };
    }

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

interface OnboardingContextValue {
  state: OnboardingState;
  /** True while any onboarding UI should render */
  isOnboarding: boolean;
  /** True only during the full-screen overlay phases */
  isOverlayVisible: boolean;
  selectInferenceMode: (mode: "local" | "cloud") => void;
  completeAuth: () => void;
  cloudConnected: () => void;
  setOnboardingStyle: (style: "teaching" | "simple") => void;
  workflowReady: () => void;
  startFromScratch: () => void;
  importWorkflowReady: () => void;
  goBack: () => void;
}

const OnboardingContext = createContext<OnboardingContextValue | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

const initialState: OnboardingState = {
  phase: "loading",
  inferenceMode: null,
  onboardingStyle: null,
};

export function OnboardingProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Fetch onboarding status from the backend on mount. This is the sole
  // source of truth — no localStorage cache.
  useEffect(() => {
    fetchOnboardingStatus().then(status => {
      dispatch({
        type: "LOADED",
        completed: status.completed,
        onboardingStyle: status.onboarding_style ?? null,
      });
      if (!status.completed) {
        trackEvent("onboarding_started", { is_first_launch: true });
      }
    });
  }, []);

  const selectInferenceMode = useCallback(
    (mode: "local" | "cloud") =>
      dispatch({ type: "SELECT_INFERENCE_MODE", mode }),
    []
  );
  const completeAuth = useCallback(
    () => dispatch({ type: "COMPLETE_AUTH" }),
    []
  );
  const cloudConnected = useCallback(
    () => dispatch({ type: "CLOUD_CONNECTED" }),
    []
  );
  const setOnboardingStyle = useCallback(
    (style: "teaching" | "simple") =>
      dispatch({ type: "SET_ONBOARDING_STYLE", style }),
    []
  );
  const workflowReady = useCallback(
    () => dispatch({ type: "WORKFLOW_READY" }),
    []
  );
  const startFromScratch = useCallback(
    () => dispatch({ type: "START_FROM_SCRATCH" }),
    []
  );
  const importWorkflowReady = useCallback(
    () => dispatch({ type: "IMPORT_WORKFLOW_READY" }),
    []
  );
  const goBack = useCallback(() => dispatch({ type: "GO_BACK" }), []);

  const isOnboarding = state.phase !== "idle";
  const isOverlayVisible = [
    "loading",
    "inference",
    "cloud_auth",
    "cloud_connecting",
    "workflow",
  ].includes(state.phase);

  return (
    <OnboardingContext.Provider
      value={{
        state,
        isOnboarding,
        isOverlayVisible,
        selectInferenceMode,
        completeAuth,
        cloudConnected,
        setOnboardingStyle,
        workflowReady,
        startFromScratch,
        importWorkflowReady,
        goBack,
      }}
    >
      {children}
    </OnboardingContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useOnboarding(): OnboardingContextValue {
  const ctx = useContext(OnboardingContext);
  if (!ctx) {
    throw new Error("useOnboarding must be used inside <OnboardingProvider>");
  }
  return ctx;
}
