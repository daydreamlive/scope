import { useState, useEffect, useCallback } from "react";
import { BookOpenText, Loader2 } from "lucide-react";
import { Button } from "../ui/button";
import { openExternalUrl } from "@/lib/openExternal";

interface OscStatus {
  enabled: boolean;
  listening: boolean;
  port: number | null;
  host: string | null;
}

interface OscTabProps {
  isActive: boolean;
}

export function OscTab({ isActive }: OscTabProps) {
  const [status, setStatus] = useState<OscStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch("/api/v1/osc/status");
      if (res.ok) {
        setStatus(await res.json());
      }
    } catch (err) {
      console.error("Failed to fetch OSC status:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isActive) {
      fetchStatus();
    }
  }, [isActive, fetchStatus]);

  const handleOpenDocs = () => {
    openExternalUrl(`${window.location.origin}/api/v1/osc/docs`);
  };

  return (
    <div className="space-y-4">
      <div className="rounded-lg bg-muted/50 p-4 space-y-4">
        {/* Status */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Status
          </span>
          <div className="flex-1 flex items-center justify-end">
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            ) : status?.listening ? (
              <span className="text-sm text-green-500">
                Listening on UDP port {status.port}
              </span>
            ) : (
              <span className="text-sm text-muted-foreground">
                Not listening
              </span>
            )}
          </div>
        </div>

        {/* Port */}
        {status?.port && (
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-foreground w-32">
              UDP Port
            </span>
            <div className="flex-1 flex items-center justify-end">
              <span className="text-sm text-muted-foreground font-mono">
                {status.port}
              </span>
            </div>
          </div>
        )}

        {/* Docs */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Reference
          </span>
          <div className="flex-1 flex items-center justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={handleOpenDocs}
              className="gap-1.5"
            >
              <BookOpenText className="h-4 w-4" />
              Open OSC Docs
            </Button>
          </div>
        </div>

        {/* Quick-start hint */}
        <div className="text-xs text-muted-foreground leading-relaxed">
          Send OSC messages to{" "}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            /scope/&lt;param&gt;
          </code>{" "}
          on UDP port{" "}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            {status?.port ?? "..."}
          </code>{" "}
          to control pipeline parameters in real time. Click{" "}
          <strong>Open OSC Docs</strong> for the full path reference.
        </div>
      </div>
    </div>
  );
}
