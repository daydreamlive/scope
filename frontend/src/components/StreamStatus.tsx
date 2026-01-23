import { useEffect, useState } from "react";
import { getStreamStatus, type StreamStatus as StreamStatusType } from "../lib/daydream";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface StreamStatusProps {
  streamId: string | null;
  isActive: boolean;
}

export function StreamStatus({ streamId, isActive }: StreamStatusProps) {
  const [status, setStatus] = useState<StreamStatusType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!streamId || !isActive) {
      setStatus(null);
      setError(null);
      return;
    }

    let intervalId: number;

    const fetchStatus = async () => {
      try {
        setIsLoading(true);
        const result = await getStreamStatus(streamId);
        setStatus(result);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch stream status:", err);
        setError(err instanceof Error ? err.message : "Failed to fetch status");
      } finally {
        setIsLoading(false);
      }
    };

    // Fetch immediately
    fetchStatus();

    // Then poll every 5 seconds
    intervalId = setInterval(fetchStatus, 5000);

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [streamId, isActive]);

  // Don't render if no stream or not active
  if (!streamId || !isActive) {
    return null;
  }

  return (
    <div className="fixed bottom-4 left-4 z-50 w-80">
      <Card className="bg-background/95 backdrop-blur shadow-lg">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center justify-between">
            <span>Stream Status</span>
            {isLoading && (
              <span className="text-xs text-muted-foreground">Updating...</span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {error ? (
            <div className="text-xs text-red-500">{error}</div>
          ) : status ? (
            <>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Inference Status:</span>
                  <span className="font-mono font-medium">{(status as any)?.data?.state ?? ""}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">WHIP Connection:</span>
                  <span className="font-mono font-medium">{(status as any)?.data?.gateway_status?.ingest_metrics?.stats?.conn_quality ?? ""}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Stream ID:</span>
                  <span className="font-mono text-[10px] truncate max-w-[180px]">
                    {streamId}
                  </span>
                </div>
                {status.created_at && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Created:</span>
                    <span className="font-mono">
                      {new Date(status.created_at).toLocaleTimeString()}
                    </span>
                  </div>
                )}
                {status.updated_at && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Updated:</span>
                    <span className="font-mono">
                      {new Date(status.updated_at).toLocaleTimeString()}
                    </span>
                  </div>
                )}
                {/* Display other simple fields */}
                {Object.entries(status)
                  .filter(
                    ([key]) =>
                      !["id", "status", "created_at", "updated_at", "data"].includes(key)
                  )
                  .map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-muted-foreground capitalize">
                        {key.replace(/_/g, " ")}:
                      </span>
                      <span className="font-mono truncate max-w-[180px]">
                        {typeof value === "object"
                          ? JSON.stringify(value)
                          : String(value)}
                      </span>
                    </div>
                  ))}
              </div>
              {/* Display data field in a large text area if it exists */}
              {status.data && (
                <div className="space-y-1">
                  <span className="text-xs text-muted-foreground">Data:</span>
                  <textarea
                    readOnly
                    value={JSON.stringify(status.data, null, 2)}
                    className="w-full h-48 p-2 text-[10px] font-mono bg-muted/50 rounded border border-border resize-y"
                  />
                </div>
              )}
            </>
          ) : (
            <div className="text-xs text-muted-foreground">
              Loading status...
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
