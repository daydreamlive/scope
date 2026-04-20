import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  getAgentConfig,
  testAgentConnection,
  updateAgentConfig,
  type AgentConfigResponse,
  type AgentProvider,
} from "@/lib/api";
import { Button } from "../ui/button";
import { Input } from "../ui/input";

interface AgentProviderTabProps {
  isActive: boolean;
}

const PROVIDER_LABELS: Record<AgentProvider, string> = {
  anthropic: "Anthropic (Claude)",
  openai_compatible: "OpenAI-compatible",
  self_hosted: "Self-hosted (Ollama / vLLM / LM Studio)",
};

const PROVIDER_DEFAULT_MODEL: Record<AgentProvider, string> = {
  anthropic: "claude-sonnet-4-6",
  openai_compatible: "gpt-4o",
  self_hosted: "llama3.1",
};

const PROVIDER_DEFAULT_BASE_URL: Record<AgentProvider, string> = {
  anthropic: "",
  openai_compatible: "https://api.openai.com/v1",
  self_hosted: "http://localhost:11434/v1",
};

const ANTHROPIC_MODEL_OPTIONS = [
  "claude-sonnet-4-6",
  "claude-opus-4-7",
  "claude-haiku-4-5-20251001",
];

export function AgentProviderTab({ isActive }: AgentProviderTabProps) {
  const [config, setConfig] = useState<AgentConfigResponse | null>(null);
  const [provider, setProvider] = useState<AgentProvider>("anthropic");
  const [model, setModel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);

  const fetchConfig = useCallback(async () => {
    try {
      const cfg = await getAgentConfig();
      setConfig(cfg);
      setProvider(cfg.provider);
      setModel(cfg.model);
      setBaseUrl(cfg.base_url ?? "");
    } catch (e) {
      toast.error(
        e instanceof Error ? e.message : "Failed to load agent config"
      );
    }
  }, []);

  useEffect(() => {
    if (isActive) void fetchConfig();
  }, [isActive, fetchConfig]);

  const handleProviderChange = (next: AgentProvider) => {
    setProvider(next);
    setModel(PROVIDER_DEFAULT_MODEL[next]);
    setBaseUrl(PROVIDER_DEFAULT_BASE_URL[next]);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await updateAgentConfig({
        provider,
        model: model.trim() || PROVIDER_DEFAULT_MODEL[provider],
        base_url: baseUrl.trim() || null,
      });
      toast.success("Agent config saved");
      await fetchConfig();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to save config");
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    try {
      const result = await testAgentConnection();
      if (result.ok) {
        toast.success("Connection OK");
      } else {
        toast.error(`Connection failed: ${result.error ?? "unknown"}`);
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Test failed");
    } finally {
      setTesting(false);
    }
  };

  const keySourceLabel = (p: AgentProvider): string => {
    const source = config?.key_sources[p];
    if (source === "env_var") return "Set via environment variable";
    if (source === "stored") return "Stored in ~/.daydream-scope/";
    if (p === "self_hosted") return "Usually no key required";
    return "Not configured — add one in API Keys tab";
  };

  return (
    <div className="space-y-6 max-w-xl">
      <div>
        <h3 className="text-lg font-semibold mb-1">Agent Provider</h3>
        <p className="text-sm text-muted-foreground">
          Choose which model powers the in-app agent. API keys live in the API
          Keys tab.
        </p>
      </div>

      <div className="space-y-3">
        <label className="text-sm font-medium">Provider</label>
        <div className="space-y-2">
          {(Object.keys(PROVIDER_LABELS) as AgentProvider[]).map(p => (
            <label
              key={p}
              className="flex items-start gap-3 p-3 border border-border rounded-md cursor-pointer hover:bg-muted/30"
            >
              <input
                type="radio"
                name="agent-provider"
                value={p}
                checked={provider === p}
                onChange={() => handleProviderChange(p)}
                className="mt-0.5"
              />
              <div className="flex-1">
                <div className="text-sm font-medium">{PROVIDER_LABELS[p]}</div>
                <div className="text-xs text-muted-foreground mt-0.5">
                  {keySourceLabel(p)}
                </div>
              </div>
            </label>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Model</label>
        {provider === "anthropic" ? (
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="w-full bg-background border border-border rounded-md px-3 py-2 text-sm"
          >
            {ANTHROPIC_MODEL_OPTIONS.map(m => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
            {!ANTHROPIC_MODEL_OPTIONS.includes(model) && model && (
              <option value={model}>{model}</option>
            )}
          </select>
        ) : (
          <Input
            value={model}
            onChange={e => setModel(e.target.value)}
            placeholder={PROVIDER_DEFAULT_MODEL[provider]}
            className="text-sm"
          />
        )}
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">
          Base URL{" "}
          <span className="text-xs text-muted-foreground font-normal">
            (optional override)
          </span>
        </label>
        <Input
          value={baseUrl}
          onChange={e => setBaseUrl(e.target.value)}
          placeholder={PROVIDER_DEFAULT_BASE_URL[provider] || "Default"}
          className="text-sm"
        />
      </div>

      <div className="flex gap-2 pt-2">
        <Button onClick={handleSave} disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </Button>
        <Button variant="outline" onClick={handleTest} disabled={testing}>
          {testing ? "Testing…" : "Test connection"}
        </Button>
      </div>
    </div>
  );
}
