import { useEffect, useState, useRef, useCallback } from "react";
import { Loader2, CheckCircle2 } from "lucide-react";
import { useCloudStatus } from "../../hooks/useCloudStatus";
import { getDaydreamUserId } from "../../lib/auth";
import { persistSurveyAnswers } from "../../lib/onboardingStorage";
import { useOnboarding } from "../../contexts/OnboardingContext";
import { useTelemetry } from "../../contexts/TelemetryContext";
import { CloudSurveyScreens, type SurveyAnswers } from "./CloudSurveyScreens";
import { TelemetryDisclosure } from "./TelemetryDisclosure";

/** Fire-and-forget: tell the backend to connect to the cloud relay. */
async function activateCloudRelay() {
  const userId = getDaydreamUserId();
  if (!userId) return;
  try {
    await fetch("/api/v1/cloud/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId }),
    });
  } catch (err) {
    console.error("[Onboarding] Failed to auto-connect to cloud:", err);
  }
}

interface CloudConnectingStepProps {
  onConnected: () => void;
  onBack?: () => void;
}

export function CloudConnectingStep({
  onConnected,
  onBack,
}: CloudConnectingStepProps) {
  const { isConnected, isConnecting, connectStage, refresh } = useCloudStatus();
  const { setOnboardingStyle, setSurveyAnswers } = useOnboarding();
  const {
    isDisclosed: telemetryDisclosed,
    markDisclosed,
    setEnabled: setTelemetryEnabled,
    flushQueue,
    dropQueue,
  } = useTelemetry();
  const didConnect = useRef(false);

  const [surveyDone, setSurveyDone] = useState(false);
  const [localTelemetryDisclosed, setLocalTelemetryDisclosed] =
    useState(telemetryDisclosed);
  const [telemetrySkippedSurvey, setTelemetrySkippedSurvey] = useState(false);
  const [surveyAnswers, setSurveyAnswersLocal] = useState<SurveyAnswers | null>(
    null
  );

  // Ensure cloud relay is connecting on mount
  useEffect(() => {
    if (didConnect.current) return;
    didConnect.current = true;
    activateCloudRelay().then(() => refresh());
  }, [refresh]);

  // Keep polling while this step is visible
  useEffect(() => {
    if (isConnected) return;
    const timer = setInterval(refresh, 1_500);
    return () => clearInterval(timer);
  }, [isConnected, refresh]);

  // Advance when survey (or skip), connection, and telemetry disclosure are done
  useEffect(() => {
    if (!isConnected || !localTelemetryDisclosed) return;

    // Path A: survey completed normally
    if (surveyDone && surveyAnswers) {
      persistSurveyAnswers({
        onboarding_style: surveyAnswers.onboardingStyle,
        referral_source: surveyAnswers.referralSource,
        use_case: surveyAnswers.useCase,
      });
      setOnboardingStyle(surveyAnswers.onboardingStyle);
      setSurveyAnswers(surveyAnswers.referralSource, surveyAnswers.useCase);
      const timer = setTimeout(onConnected, 500);
      return () => clearTimeout(timer);
    }

    // Path B: telemetry declined → survey skipped, onboarding_style selected via inline picker
    if (telemetrySkippedSurvey && surveyAnswers) {
      persistSurveyAnswers({
        onboarding_style: surveyAnswers.onboardingStyle,
        referral_source: null,
        use_case: null,
      });
      setOnboardingStyle(surveyAnswers.onboardingStyle);
      setSurveyAnswers(null, null);
      const timer = setTimeout(onConnected, 500);
      return () => clearTimeout(timer);
    }
  }, [
    isConnected,
    surveyDone,
    surveyAnswers,
    localTelemetryDisclosed,
    telemetrySkippedSurvey,
    onConnected,
    setOnboardingStyle,
    setSurveyAnswers,
  ]);

  const handleSurveyComplete = useCallback((answers: SurveyAnswers) => {
    setSurveyAnswersLocal(answers);
    setSurveyDone(true);
  }, []);

  const handleTelemetryAccept = useCallback(() => {
    markDisclosed();
    setTelemetryEnabled(true);
    flushQueue();
    setLocalTelemetryDisclosed(true);
  }, [markDisclosed, setTelemetryEnabled, flushQueue]);

  const handleTelemetryDecline = useCallback(() => {
    markDisclosed();
    dropQueue();
    setLocalTelemetryDisclosed(true);
    // "No thanks" = skip the survey, go straight to onboarding style picker
    setTelemetrySkippedSurvey(true);
  }, [markDisclosed, dropQueue]);

  // --- Step 1: Telemetry disclosure (shown FIRST, replaces survey intro) ---
  if (!localTelemetryDisclosed) {
    return (
      <div className="flex flex-col items-center gap-6 w-full max-w-md mx-auto">
        <TelemetryDisclosure
          onAccept={handleTelemetryAccept}
          onDecline={handleTelemetryDecline}
        />
        {/* Small connection status at bottom */}
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {isConnected ? (
            <>
              <CheckCircle2 className="h-3 w-3 text-green-500" />
              <span>Connected</span>
            </>
          ) : (
            <>
              <Loader2 className="h-3 w-3 animate-spin" />
              <span>
                {isConnecting && connectStage ? connectStage : "Connecting..."}
              </span>
            </>
          )}
        </div>
      </div>
    );
  }

  // --- Step 2a: Full survey (if user accepted telemetry) ---
  if (!surveyDone && !telemetrySkippedSurvey) {
    return (
      <div className="flex flex-col items-center gap-6 w-full max-w-md mx-auto">
        <CloudSurveyScreens
          onComplete={handleSurveyComplete}
          onBack={onBack}
          initialScreen="referral"
        />
        {/* Small connection status at bottom */}
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {isConnected ? (
            <>
              <CheckCircle2 className="h-3 w-3 text-green-500" />
              <span>Connected</span>
            </>
          ) : (
            <>
              <Loader2 className="h-3 w-3 animate-spin" />
              <span>
                {isConnecting && connectStage ? connectStage : "Connecting..."}
              </span>
            </>
          )}
        </div>
      </div>
    );
  }

  // --- Step 2b: Telemetry declined → skip survey, show only onboarding style picker ---
  if (telemetrySkippedSurvey && !surveyAnswers) {
    return (
      <div className="flex flex-col items-center gap-6 w-full max-w-md mx-auto">
        <CloudSurveyScreens
          onComplete={handleSurveyComplete}
          initialScreen="onboarding_style"
        />
        {/* Small connection status at bottom */}
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {isConnected ? (
            <>
              <CheckCircle2 className="h-3 w-3 text-green-500" />
              <span>Connected</span>
            </>
          ) : (
            <>
              <Loader2 className="h-3 w-3 animate-spin" />
              <span>
                {isConnecting && connectStage ? connectStage : "Connecting..."}
              </span>
            </>
          )}
        </div>
      </div>
    );
  }

  // --- Survey done, waiting for cloud ---
  if (!isConnected) {
    return (
      <div className="flex flex-col items-center gap-6 w-full max-w-md mx-auto text-center">
        <h2 className="text-2xl font-semibold text-foreground">
          Almost there...
        </h2>
        <Loader2 className="h-8 w-8 text-muted-foreground animate-spin" />
        <p className="text-sm text-muted-foreground">
          Finishing connection to Daydream Cloud
        </p>
      </div>
    );
  }

  // --- Both done, brief green check before auto-advance ---
  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-md mx-auto text-center">
      <CheckCircle2 className="h-8 w-8 text-green-500" />
      <p className="text-sm text-foreground">Connected</p>
    </div>
  );
}
