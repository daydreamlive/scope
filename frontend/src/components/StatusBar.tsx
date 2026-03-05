interface StatusBarProps {
  className?: string;
  fps?: number;
  bitrate?: number;
}

export function StatusBar({ className = "", fps, bitrate }: StatusBarProps) {
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

  return (
    <div
      className={`border-t bg-muted/30 px-6 py-2 flex items-center justify-end flex-shrink-0 ${className}`}
    >
      <div className="flex items-center gap-6">
        <MetricItem label="FPS" value={fpsValue} />
        <MetricItem label="Bitrate" value={bitrateValue} />
      </div>
    </div>
  );
}
