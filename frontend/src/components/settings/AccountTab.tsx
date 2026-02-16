import { DaydreamAccountSection } from "./DaydreamAccountSection";

interface AccountTabProps {
  /** Callback to refresh pipeline list after cloud mode toggle */
  onPipelinesRefresh?: () => Promise<unknown>;
  /** Disable the toggle (e.g., when streaming) */
  cloudDisabled?: boolean;
}

export function AccountTab({
  onPipelinesRefresh,
  cloudDisabled,
}: AccountTabProps) {
  return (
    <div className="space-y-4">
      <DaydreamAccountSection
        onPipelinesRefresh={onPipelinesRefresh}
        disabled={cloudDisabled}
      />
    </div>
  );
}
