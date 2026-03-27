import { useEffect, useState, useRef, useCallback } from "react";
import { Loader2, CheckCircle2 } from "lucide-react";
import { useCloudStatus } from "../../hooks/useCloudStatus";
import { getDaydreamUserId } from "../../lib/auth";
import {
  persistSurveyAnswers,
  activateCloudRelay,
} from "../../lib/onboardingStorage";
import { useOnboarding } from "../../contexts/OnboardingContext";
import { CloudSurveyScreens, type SurveyAnswers } from "./CloudSurveyScreens";

interface CloudConnectingStepProps {
  onConnected: () => void;
  onBack?: () => void;
}

export function CloudConnectingStep({
  onConnected,
  onBack,
}: CloudConnectingStepProps) {
  const { isConnected, isConnecting, connectStage, refresh } = useCloudStatus();
  const { setOnboardingStyle } = useOnboarding();
  const didConnect = useRef(false);

  const [surveyDone, setSurveyDone] = useState(false);
  const [surveyAnswers, setSurveyAnswers] = useState<SurveyAnswers | null>(
    null
  );

  // Ensure cloud relay is connecting on mount
  useEffect(() => {
    if (didConnect.current) return;
    didConnect.current = true;
    activateCloudRelay(getDaydreamUserId()).then(() => refresh());
  }, [refresh]);

  // Keep polling while this step is visible
  useEffect(() => {
    if (isConnected) return;
    const timer = setInterval(refresh, 1_500);
    return () => clearInterval(timer);
  }, [isConnected, refresh]);

  // Advance when both survey and connection are done
  useEffect(() => {
    if (isConnected && surveyDone && surveyAnswers) {
      // Persist survey answers to backend
      persistSurveyAnswers({
        onboarding_style: surveyAnswers.onboardingStyle,
        referral_source: surveyAnswers.referralSource,
        use_case: surveyAnswers.useCase,
      });
      setOnboardingStyle(surveyAnswers.onboardingStyle);
      const timer = setTimeout(onConnected, 500);
      return () => clearTimeout(timer);
    }
  }, [isConnected, surveyDone, surveyAnswers, onConnected, setOnboardingStyle]);

  const handleSurveyComplete = useCallback((answers: SurveyAnswers) => {
    setSurveyAnswers(answers);
    setSurveyDone(true);
  }, []);

  // --- Survey in progress ---
  if (!surveyDone) {
    return (
      <div className="flex flex-col items-center gap-6 w-full max-w-md mx-auto">
        <CloudSurveyScreens onComplete={handleSurveyComplete} onBack={onBack} />
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
