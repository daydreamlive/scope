import {
  createContext,
  useContext,
  useReducer,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import {
  isOnboardingCompletedSync,
  fetchOnboardingStatus,
  markOnboardingCompleted,
  setInferenceMode as persistInferenceMode,
} from "../lib/onboardingStorage";
import { trackEvent } from "../lib/analytics";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type OnboardingPhase =
  | "idle" // returning user or post-completion
  | "inference" // step 1: local vs cloud
  | "cloud_auth" // step 2: sign in (only if cloud chosen)
  | "workflow" // step 3: starter workflow picker
  | "downloading" // step 3b: workflow downloading
  | "tour" // step 4: popover walkthrough
  | "completed"; // persist and transition to idle

export type TourStep = 0 | 1 | 2 | 3 | 4 | 5;

const TOTAL_TOUR_STEPS = 6;

export interface OnboardingState {
  phase: OnboardingPhase;
  tourStep: TourStep;
  inferenceMode: "local" | "cloud" | null;
  selectedWorkflowId: string | null;
  downloadFailures: number;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

type OnboardingAction =
  | { type: "SELECT_INFERENCE_MODE"; mode: "local" | "cloud" }
  | { type: "COMPLETE_AUTH" }
  | { type: "SKIP_AUTH" }
  | { type: "SELECT_WORKFLOW"; workflowId: string }
  | { type: "START_DOWNLOADING" }
  | { type: "DOWNLOAD_FAILED" }
  | { type: "WORKFLOW_READY" }
  | { type: "START_FROM_SCRATCH" }
  | { type: "IMPORT_WORKFLOW_READY" }
  | { type: "ADVANCE_TOUR" }
  | { type: "SKIP_TOUR" }
  | { type: "COMPLETE" }
  | { type: "BACKEND_SAYS_COMPLETED" };

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
      return {
        ...state,
        inferenceMode: action.mode,
        phase: action.mode === "cloud" ? "cloud_auth" : "workflow",
      };

    case "COMPLETE_AUTH":
      trackEvent("onboarding_auth_completed");
      return { ...state, phase: "workflow" };

    case "SKIP_AUTH":
      trackEvent("onboarding_auth_skipped");
      return { ...state, inferenceMode: "local", phase: "workflow" };

    case "SELECT_WORKFLOW":
      return { ...state, selectedWorkflowId: action.workflowId };

    case "START_DOWNLOADING":
      trackEvent("onboarding_workflow_selected", {
        workflowId: state.selectedWorkflowId,
      });
      return { ...state, phase: "downloading" };

    case "DOWNLOAD_FAILED":
      return {
        ...state,
        phase: "workflow",
        downloadFailures: state.downloadFailures + 1,
      };

    case "WORKFLOW_READY":
      trackEvent("onboarding_workflow_downloaded", {
        workflowId: state.selectedWorkflowId,
      });
      markOnboardingCompleted();
      return { ...state, phase: "tour", tourStep: 0 };

    case "START_FROM_SCRATCH":
      trackEvent("onboarding_started_from_scratch");
      markOnboardingCompleted();
      return {
        ...state,
        phase: "tour",
        tourStep: 0,
        selectedWorkflowId: null,
      };

    case "IMPORT_WORKFLOW_READY":
      trackEvent("onboarding_imported_workflow");
      markOnboardingCompleted();
      return { ...state, phase: "tour", tourStep: 0 };

    case "ADVANCE_TOUR": {
      const nextStep = (state.tourStep + 1) as TourStep;
      if (nextStep >= TOTAL_TOUR_STEPS) {
        trackEvent("onboarding_tour_completed");
        trackEvent("onboarding_completed");
        return { ...state, phase: "idle", tourStep: 0 };
      }
      trackEvent("onboarding_tour_step_viewed", { step: nextStep });
      return { ...state, tourStep: nextStep };
    }

    case "SKIP_TOUR":
      trackEvent("onboarding_tour_dismissed", { at_step: state.tourStep });
      trackEvent("onboarding_completed");
      return { ...state, phase: "idle", tourStep: 0 };

    case "COMPLETE":
      trackEvent("onboarding_completed");
      return { ...state, phase: "idle" };

    case "BACKEND_SAYS_COMPLETED":
      // Backend file says onboarding was already completed — skip to idle
      return { ...state, phase: "idle" };

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
  /** True only during the popover tour */
  isTourActive: boolean;

  selectInferenceMode: (mode: "local" | "cloud") => void;
  completeAuth: () => void;
  skipAuth: () => void;
  selectWorkflow: (workflowId: string) => void;
  startDownloading: () => void;
  downloadFailed: () => void;
  workflowReady: () => void;
  startFromScratch: () => void;
  importWorkflowReady: () => void;
  advanceTour: () => void;
  skipTour: () => void;
}

const OnboardingContext = createContext<OnboardingContextValue | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

const initialState: OnboardingState = {
  phase: isOnboardingCompletedSync() ? "idle" : "inference",
  tourStep: 0,
  inferenceMode: null,
  selectedWorkflowId: null,
  downloadFailures: 0,
};

export function OnboardingProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Reconcile with backend file on mount. The localStorage cache gives us a
  // fast synchronous initial value, but the backend file is the source of
  // truth. If the backend says completed (e.g. localStorage was cleared but
  // the file persists), skip straight to idle.
  useEffect(() => {
    fetchOnboardingStatus().then(status => {
      if (status.completed) {
        dispatch({ type: "BACKEND_SAYS_COMPLETED" });
      }
    });
  }, []);

  // Fire onboarding_started exactly once if we're entering onboarding
  useEffect(() => {
    if (initialState.phase !== "idle") {
      trackEvent("onboarding_started", {
        is_first_launch: true,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fire tour step 0 viewed when entering tour
  useEffect(() => {
    if (state.phase === "tour" && state.tourStep === 0) {
      trackEvent("onboarding_tour_step_viewed", { step: 0 });
    }
  }, [state.phase, state.tourStep]);

  const selectInferenceMode = useCallback(
    (mode: "local" | "cloud") =>
      dispatch({ type: "SELECT_INFERENCE_MODE", mode }),
    []
  );
  const completeAuth = useCallback(
    () => dispatch({ type: "COMPLETE_AUTH" }),
    []
  );
  const skipAuth = useCallback(() => dispatch({ type: "SKIP_AUTH" }), []);
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
  const advanceTour = useCallback(() => dispatch({ type: "ADVANCE_TOUR" }), []);
  const skipTour = useCallback(() => dispatch({ type: "SKIP_TOUR" }), []);

  const isOnboarding = state.phase !== "idle";
  const isOverlayVisible = [
    "inference",
    "cloud_auth",
    "workflow",
    "downloading",
  ].includes(state.phase);
  const isTourActive = state.phase === "tour";

  return (
    <OnboardingContext.Provider
      value={{
        state,
        isOnboarding,
        isOverlayVisible,
        isTourActive,
        selectInferenceMode,
        completeAuth,
        skipAuth,
        selectWorkflow,
        startDownloading,
        downloadFailed,
        workflowReady,
        startFromScratch,
        importWorkflowReady,
        advanceTour,
        skipTour,
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
