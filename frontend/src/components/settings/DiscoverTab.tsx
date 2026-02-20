import { useState, useEffect, useCallback } from "react";
import { Download, ExternalLink, Search, Loader2 } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Badge } from "../ui/badge";

const DAYDREAM_API_BASE =
  (import.meta.env.VITE_DAYDREAM_API_BASE as string | undefined) ||
  "https://api.daydream.live";
const DAYDREAM_APP_BASE =
  (import.meta.env.VITE_DAYDREAM_APP_BASE as string | undefined) ||
  "https://app.daydream.live";

function GitHubIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      aria-hidden="true"
    >
      <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2Z" />
    </svg>
  );
}

interface DaydreamPlugin {
  id: string;
  creatorId: string;
  creatorUsername: string;
  name: string;
  slug: string;
  description: string | null;
  iconUrl: string | null;
  pluginType: string;
  category: string | null;
  tags: string[];
  learnMoreUrl: string | null;
  repositoryUrl: string | null;
  downloadCount: number;
  version: string | null;
}

interface DiscoverResponse {
  plugins: DaydreamPlugin[];
  totalCount: number;
  hasMore: boolean;
}

function normalizeRepoUrl(url: string): string {
  return url
    .replace(/^git\+/, "")
    .replace(/\.git$/, "")
    .replace(/\/$/, "")
    .toLowerCase();
}

interface DiscoverTabProps {
  onInstall: (packageSpec: string) => void;
  installedRepoUrls: string[];
  isInstalling?: boolean;
}

export function DiscoverTab({
  onInstall,
  installedRepoUrls,
  isInstalling = false,
}: DiscoverTabProps) {
  const [plugins, setPlugins] = useState<DaydreamPlugin[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 300);
    return () => clearTimeout(timer);
  }, [search]);

  const fetchDiscoverPlugins = useCallback(async () => {
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
        `${DAYDREAM_API_BASE}/v1/plugins?${params.toString()}`
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch plugins (${response.status})`);
      }
      const data: DiscoverResponse = await response.json();
      setPlugins(data.plugins);
    } catch (err) {
      console.error("Failed to fetch discover plugins:", err);
      setError(err instanceof Error ? err.message : "Failed to load plugins");
    } finally {
      setIsLoading(false);
    }
  }, [debouncedSearch]);

  useEffect(() => {
    fetchDiscoverPlugins();
  }, [fetchDiscoverPlugins]);

  const installedSet = new Set(installedRepoUrls.map(normalizeRepoUrl));

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search plugins..."
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
            onClick={fetchDiscoverPlugins}
          >
            Retry
          </Button>
        </div>
      ) : plugins.length === 0 ? (
        <p className="text-sm text-muted-foreground text-center py-8">
          {debouncedSearch
            ? "No plugins found matching your search."
            : "No plugins available."}
        </p>
      ) : (
        <div className="space-y-3">
          {plugins.map(plugin => {
            const isInstalled = plugin.repositoryUrl
              ? installedSet.has(normalizeRepoUrl(plugin.repositoryUrl))
              : false;
            const daydreamUrl =
              plugin.creatorUsername && plugin.slug
                ? `${DAYDREAM_APP_BASE}/plugins/${plugin.creatorUsername}/${plugin.slug}`
                : plugin.learnMoreUrl ||
                  `${DAYDREAM_APP_BASE}/plugins?search=${encodeURIComponent(plugin.name)}`;
            return (
              <div
                key={plugin.id}
                className="flex items-start gap-3 p-3 rounded-md border border-border bg-card"
              >
                {plugin.iconUrl ? (
                  <img
                    src={plugin.iconUrl}
                    alt=""
                    className="h-9 w-9 rounded-md object-cover shrink-0 mt-0.5"
                  />
                ) : (
                  <div className="h-9 w-9 rounded-md bg-muted shrink-0 mt-0.5 flex items-center justify-center">
                    <span className="text-xs text-muted-foreground font-medium">
                      {plugin.name.charAt(0).toUpperCase()}
                    </span>
                  </div>
                )}
                <div className="space-y-1.5 min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm font-medium text-foreground">
                      {plugin.name}
                    </span>
                    {plugin.version && (
                      <span className="text-xs text-muted-foreground">
                        v{plugin.version}
                      </span>
                    )}
                  </div>
                  {plugin.description && (
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {plugin.description}
                    </p>
                  )}
                  {plugin.tags.length > 0 && (
                    <div className="flex gap-1 flex-wrap">
                      {plugin.tags.slice(0, 4).map(tag => (
                        <span
                          key={tag}
                          className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded"
                        >
                          {tag}
                        </span>
                      ))}
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
                  {plugin.repositoryUrl && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      title="View on GitHub"
                      asChild
                    >
                      <a
                        href={plugin.repositoryUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <GitHubIcon className="h-4 w-4" />
                      </a>
                    </Button>
                  )}
                  {isInstalled ? (
                    <Badge variant="outline" className="text-xs">
                      Installed
                    </Badge>
                  ) : plugin.repositoryUrl ? (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      title={`Install ${plugin.name}`}
                      disabled={isInstalling}
                      onClick={() => onInstall(`git+${plugin.repositoryUrl}`)}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
