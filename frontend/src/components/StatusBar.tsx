interface StatusBarProps {
  className?: string;
  fps?: number;
  bitrate?: number;
  originalFPS?: number | null;
  interpolatedFPS?: number | null;
}

export function StatusBar({
  className = "",
  fps,
  bitrate,
  originalFPS,
  interpolatedFPS,
}: StatusBarProps) {
  const MetricItem = ({
    label,
    value,
    unit = "",
  }: {
    label: string;
    value: number | string;
    unit?: string;
  }) => (
    <div className="flex items-center gap-1 text-xs">
      <span className="font-medium">{label}:</span>
      <span className="font-mono">
        {value}
        {unit}
      </span>
    </div>
  );

  const formatBitrate = (bps?: number): string => {
    if (bps === undefined || bps === 0) return "N/A";

    if (bps >= 1000000) {
      return `${(bps / 1000000).toFixed(1)} Mbps`;
    } else {
      return `${Math.round(bps / 1000)} kbps`;
    }
  };

  const fpsValue = fps !== undefined && fps > 0 ? fps.toFixed(1) : "N/A";
  const bitrateValue = formatBitrate(bitrate);
  const originalFPSValue =
    originalFPS !== null && originalFPS !== undefined && originalFPS > 0
      ? originalFPS.toFixed(1)
      : null;
  const interpolatedFPSValue =
    interpolatedFPS !== null &&
    interpolatedFPS !== undefined &&
    interpolatedFPS > 0
      ? interpolatedFPS.toFixed(1)
      : null;

  // Show detailed FPS (Original + Interpolated) only when RIFE is enabled (interpolatedFPS exists)
  // Otherwise show regular FPS
  const showDetailedFPS = interpolatedFPSValue !== null;

  return (
    <div
      className={`border-t bg-muted/30 px-6 py-2 flex items-center justify-end flex-shrink-0 ${className}`}
    >
      <div className="flex items-center gap-6">
        {showDetailedFPS ? (
          <>
            {originalFPSValue !== null && (
              <MetricItem label="Original FPS" value={originalFPSValue} unit=" fps" />
            )}
            {interpolatedFPSValue !== null && (
              <MetricItem
                label="Interpolated FPS"
                value={interpolatedFPSValue}
                unit=" fps"
              />
            )}
          </>
        ) : (
        <MetricItem label="FPS" value={fpsValue} />
        )}
        <MetricItem label="Bitrate" value={bitrateValue} />
      </div>
    </div>
  );
}
