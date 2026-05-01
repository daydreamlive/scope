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
import { isDisclosed as checkTelemetryDisclosed } from "../lib/telemetry";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type OnboardingPhase =
  | "loading" // waiting for backend status check
  | "idle" // returning user or post-completion
  | "inference" // step 1: local vs cloud
  | "cloud_auth" // step 2a: sign in (only if cloud chosen)
  | "cloud_connecting" // step 2b: waiting for cloud relay connection
  | "telemetry_disclosure" // telemetry opt-in disclosure (local mode only)
  | "workflow" // step 3: starter workflow picker
  | "downloading" // step 3b: workflow downloading
  | "completed"; // persist and transition to idle

export interface OnboardingState {
  phase: OnboardingPhase;
  inferenceMode: "local" | "cloud" | null;
  onboardingStyle: "teaching" | "simple" | null;
  selectedWorkflowId: string | null;
  downloadFailures: number;
  referralSource: string | null;
  useCase: string | null;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

type OnboardingAction =
  | { type: "SELECT_INFERENCE_MODE"; mode: "local" | "cloud" }
  | { type: "COMPLETE_AUTH" }
  | { type: "CLOUD_CONNECTED" }
  | { type: "SET_ONBOARDING_STYLE"; style: "teaching" | "simple" }
  | { type: "SELECT_WORKFLOW"; workflowId: string }
  | { type: "START_DOWNLOADING" }
  | { type: "DOWNLOAD_FAILED" }
  | { type: "WORKFLOW_READY" }
  | { type: "START_FROM_SCRATCH" }
  | { type: "IMPORT_WORKFLOW_READY" }
  | { type: "GO_BACK" }
  | { type: "COMPLETE" }
  | { type: "TELEMETRY_DISCLOSED" }
  | {
      type: "SET_SURVEY_ANSWERS";
      referralSource: string | null;
      useCase: string | null;
    }
  | {
      type: "LOADED";
      completed: boolean;
      onboardingStyle?: "teaching" | "simple" | null;
      inferenceMode?: "local" | "cloud" | null;
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
        phase:
          action.mode === "cloud"
            ? "cloud_auth"
            : checkTelemetryDisclosed()
              ? "workflow"
              : "telemetry_disclosure",
      };

    case "TELEMETRY_DISCLOSED":
      return { ...state, phase: "workflow" };

    case "COMPLETE_AUTH":
      return { ...state, phase: "cloud_connecting" };

    case "CLOUD_CONNECTED":
      return { ...state, phase: "workflow" };

    case "SET_ONBOARDING_STYLE":
      return { ...state, onboardingStyle: action.style };

    case "SELECT_WORKFLOW":
      return { ...state, selectedWorkflowId: action.workflowId };

    case "START_DOWNLOADING":
      return { ...state, phase: "downloading" };

    case "DOWNLOAD_FAILED":
      return {
        ...state,
        phase: "workflow",
        downloadFailures: state.downloadFailures + 1,
      };

    case "WORKFLOW_READY":
      trackEvent("onboarding_completed", {
        inference_mode: state.inferenceMode,
        onboarding_style: state.onboardingStyle,
        selected_workflow: state.selectedWorkflowId,
        referral_source: state.referralSource,
        use_case: state.useCase,
      });
      markOnboardingCompleted();
      return { ...state, phase: "idle" };

    case "START_FROM_SCRATCH":
      trackEvent("onboarding_completed", {
        inference_mode: state.inferenceMode,
        onboarding_style: state.onboardingStyle,
        selected_workflow: null,
        referral_source: state.referralSource,
        use_case: state.useCase,
      });
      markOnboardingCompleted();
      return { ...state, phase: "idle", selectedWorkflowId: null };

    case "IMPORT_WORKFLOW_READY":
      trackEvent("onboarding_completed", {
        inference_mode: state.inferenceMode,
        onboarding_style: state.onboardingStyle,
        selected_workflow: "imported",
        referral_source: state.referralSource,
        use_case: state.useCase,
      });
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

    case "SET_SURVEY_ANSWERS":
      return {
        ...state,
        referralSource: action.referralSource,
        useCase: action.useCase,
      };

    case "LOADED": {
      if (action.completed)
        return {
          ...state,
          phase: "idle",
          onboardingStyle: action.onboardingStyle ?? null,
          inferenceMode: action.inferenceMode ?? null,
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
          return {
            ...state,
            phase: checkTelemetryDisclosed()
              ? "workflow"
              : "telemetry_disclosure",
            inferenceMode: "local",
          };
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
  selectWorkflow: (workflowId: string) => void;
  startDownloading: () => void;
  downloadFailed: () => void;
  workflowReady: () => void;
  startFromScratch: () => void;
  importWorkflowReady: () => void;
  goBack: () => void;
  telemetryDisclosed: () => void;
  setSurveyAnswers: (
    referralSource: string | null,
    useCase: string | null
  ) => void;
}

const OnboardingContext = createContext<OnboardingContextValue | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

const initialState: OnboardingState = {
  phase: "loading",
  inferenceMode: null,
  onboardingStyle: null,
  selectedWorkflowId: null,
  downloadFailures: 0,
  referralSource: null,
  useCase: null,
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
        inferenceMode:
          (status.inference_mode as "local" | "cloud" | null) ?? null,
      });
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
  const selectWorkflow = useCallback(
    (workflowId: string) => dispatch({ type: "SELECT_WORKFLOW", workflowId }),
    []
  );
  const startDownloading = useCallback(
    () => dispatch({ type: "START_DOWNLOADING" }),
    []
  );
  const downloadFailed = useCallback(
    () => dispatch({ type: "DOWNLOAD_FAILED" }),
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
  const telemetryDisclosed = useCallback(
    () => dispatch({ type: "TELEMETRY_DISCLOSED" }),
    []
  );
  const setSurveyAnswers = useCallback(
    (referralSource: string | null, useCase: string | null) =>
      dispatch({ type: "SET_SURVEY_ANSWERS", referralSource, useCase }),
    []
  );

  const isOnboarding = state.phase !== "idle";
  const isOverlayVisible = [
    "loading",
    "inference",
    "cloud_auth",
    "cloud_connecting",
    "telemetry_disclosure",
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
        selectWorkflow,
        startDownloading,
        downloadFailed,
        workflowReady,
        startFromScratch,
        importWorkflowReady,
        goBack,
        telemetryDisclosed,
        setSurveyAnswers,
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
