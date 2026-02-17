import { DaydreamAccountSection } from "./DaydreamAccountSection";

interface AccountTabProps {
  /** Callback to refresh pipeline list after cloud mode toggle */
  onPipelinesRefresh?: () => Promise<unknown>;
  /** Disable the toggle (e.g., when streaming) */
  cloudDisabled?: boolean;
  /** Whether the cloud is currently connecting */
  isConnecting?: boolean;
  /** Whether the dialog cannot be closed (auth gating mode) */
  preventClose?: boolean;
}

export function AccountTab({
  onPipelinesRefresh,
  cloudDisabled,
  isConnecting,
  preventClose,
}: AccountTabProps) {
  return (
    <div className="space-y-4">
      <DaydreamAccountSection
        onPipelinesRefresh={onPipelinesRefresh}
        disabled={cloudDisabled}
        isConnecting={isConnecting}
        preventClose={preventClose}
      />
    </div>
  );
}
