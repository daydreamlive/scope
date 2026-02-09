import { DaydreamAccountSection } from "./DaydreamAccountSection";

interface AccountTabProps {
  /** Callback when cloud mode status changes */
  onCloudStatusChange?: (connected: boolean) => void;
  /** Callback when connecting state changes */
  onCloudConnectingChange?: (connecting: boolean) => void;
  /** Callback to refresh pipeline list after cloud mode toggle */
  onPipelinesRefresh?: () => Promise<unknown>;
  /** Disable the toggle (e.g., when streaming) */
  cloudDisabled?: boolean;
}

export function AccountTab({
  onCloudStatusChange,
  onCloudConnectingChange,
  onPipelinesRefresh,
  cloudDisabled,
}: AccountTabProps) {
  return (
    <div className="space-y-4">
      <DaydreamAccountSection
        onStatusChange={onCloudStatusChange}
        onConnectingChange={onCloudConnectingChange}
        onPipelinesRefresh={onPipelinesRefresh}
        disabled={cloudDisabled}
      />
    </div>
  );
}
