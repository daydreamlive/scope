import { useState, useEffect, useCallback } from "react";
import { ExternalLink, Search, Loader2, Play, Download, Camera } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { getWorkflowsForStyle } from "../onboarding/starterWorkflows";
import { getHardwareInfo } from "../../lib/api";

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
  const [hasGpu, setHasGpu] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHardwareInfo()
      .then(info => {
        if (!cancelled) setHasGpu(info.vram_gb != null && info.vram_gb > 0);
      })
      .catch(() => {
        if (!cancelled) setHasGpu(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // On non-GPU devices, show only CPU-compatible workflows (Camera Preview);
  // otherwise show the simple-mode starter workflows (teaching workflows have
  // notes and missing source nodes that only make sense during onboarding).
  const starterWorkflows =
    hasGpu === false
      ? getWorkflowsForStyle("local")
      : getWorkflowsForStyle("simple");

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
    <div className="space-y-6">
      {/* Getting Started — starter workflows */}
      {starterWorkflows.length > 0 && (
        <div className="space-y-3">
          <div>
            <h3 className="text-sm font-medium text-foreground">
              Getting Started
            </h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Try these starter workflows to get familiar with Scope.
            </p>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {starterWorkflows.map(sw => (
              <button
                key={sw.id}
                onClick={() => onLoad(sw.workflow)}
                className="group relative rounded-lg border border-border bg-card overflow-hidden text-left hover:border-foreground/20 transition-colors"
              >
                <div className="aspect-video w-full overflow-hidden bg-muted">
                  {sw.thumbnail ? (
                    <img
                      src={sw.thumbnail}
                      alt={sw.title}
                      className="h-full w-full object-cover group-hover:scale-105 transition-transform duration-200"
                    />
                  ) : (
                    <div className="h-full w-full flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
                      <Camera className="h-8 w-8 text-slate-500" />
                    </div>
                  )}
                </div>
                <div className="p-2.5">
                  <p className="text-xs font-medium text-foreground truncate">
                    {sw.title}
                  </p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">
                    {sw.category}
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Divider */}
      {starterWorkflows.length > 0 && (
        <div className="border-t border-border" />
      )}

      {/* Community workflows */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-foreground">
          Community Workflows
        </h3>
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
    </div>
  );
}
