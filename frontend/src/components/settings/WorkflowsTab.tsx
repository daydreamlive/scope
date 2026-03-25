import { useState, useEffect, useCallback } from "react";
import { ExternalLink, Search, Loader2, Play, Download } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";

const DAYDREAM_API_BASE =
  (import.meta.env.VITE_DAYDREAM_API_BASE as string | undefined) ||
  "https://api.daydream.live";
const DAYDREAM_APP_BASE =
  (import.meta.env.VITE_DAYDREAM_APP_BASE as string | undefined) ||
  "https://app.daydream.live";

interface DaydreamWorkflow {
  id: string;
  creatorUsername: string;
  name: string;
  slug: string;
  description: string | null;
  thumbnailUrl: string | null;
  videoUrl: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  workflowData: Record<string, any> | null;
  downloadCount: number;
  version: string | null;
  featured: boolean;
}

interface WorkflowsResponse {
  workflows: DaydreamWorkflow[];
  totalCount: number;
  hasMore: boolean;
}

interface WorkflowsTabProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onLoad: (workflowData: Record<string, any>) => void;
}

export function WorkflowsTab({ onLoad }: WorkflowsTabProps) {
  const [workflows, setWorkflows] = useState<DaydreamWorkflow[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 300);
    return () => clearTimeout(timer);
  }, [search]);

  const fetchWorkflows = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        limit: "50",
        sortBy: "popularity",
      });
      if (debouncedSearch) {
        params.set("search", debouncedSearch);
      }
      const response = await fetch(
        `${DAYDREAM_API_BASE}/v1/workflows?${params.toString()}`
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch workflows (${response.status})`);
      }
      const data: WorkflowsResponse = await response.json();
      setWorkflows(data.workflows);
    } catch (err) {
      console.error("Failed to fetch workflows:", err);
      setError(err instanceof Error ? err.message : "Failed to load workflows");
    } finally {
      setIsLoading(false);
    }
  }, [debouncedSearch]);

  useEffect(() => {
    fetchWorkflows();
  }, [fetchWorkflows]);

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search workflows..."
          className="pl-9"
        />
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      ) : error ? (
        <div className="rounded-lg bg-muted/50 p-4 text-center">
          <p className="text-sm text-muted-foreground">{error}</p>
          <Button
            variant="outline"
            size="sm"
            className="mt-2"
            onClick={fetchWorkflows}
          >
            Retry
          </Button>
        </div>
      ) : workflows.length === 0 ? (
        <p className="text-sm text-muted-foreground text-center py-8">
          {debouncedSearch
            ? "No workflows found matching your search."
            : "No workflows available."}
        </p>
      ) : (
        <div className="space-y-3">
          {workflows.map(wf => {
            const daydreamUrl =
              wf.creatorUsername && wf.slug
                ? `${DAYDREAM_APP_BASE}/workflows/${wf.creatorUsername}/${wf.slug}`
                : `${DAYDREAM_APP_BASE}/workflows`;
            return (
              <div
                key={wf.id}
                className="flex items-start gap-3 p-3 rounded-md border border-border bg-card"
              >
                {wf.thumbnailUrl ? (
                  <img
                    src={wf.thumbnailUrl}
                    alt=""
                    className="h-16 w-28 rounded-md object-cover shrink-0"
                  />
                ) : (
                  <div className="h-16 w-28 rounded-md bg-muted shrink-0 flex items-center justify-center">
                    <Play className="h-5 w-5 text-muted-foreground" />
                  </div>
                )}
                <div className="space-y-1 min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm font-medium text-foreground">
                      {wf.name}
                    </span>
                    {wf.version && (
                      <span className="text-xs text-muted-foreground">
                        v{wf.version}
                      </span>
                    )}
                  </div>
                  {wf.creatorUsername && (
                    <p className="text-xs text-muted-foreground">
                      by {wf.creatorUsername}
                    </p>
                  )}
                  {wf.description && (
                    <p className="text-xs text-muted-foreground line-clamp-2">
                      {wf.description}
                    </p>
                  )}
                  {wf.downloadCount > 0 && (
                    <div className="flex items-center gap-1 text-xs text-muted-foreground/70">
                      <Download className="h-3 w-3" />
                      <span>{wf.downloadCount}</span>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    title="View on Daydream"
                    asChild
                  >
                    <a
                      href={daydreamUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <ExternalLink className="h-4 w-4" />
                    </a>
                  </Button>
                  {wf.workflowData && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs"
                      onClick={() => onLoad(wf.workflowData!)}
                    >
                      Load
                    </Button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
