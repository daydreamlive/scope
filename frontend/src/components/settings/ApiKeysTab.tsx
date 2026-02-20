import { useState, useEffect, useCallback } from "react";
import { ExternalLink, Info, Save, Trash2 } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { toast } from "sonner";
import { getApiKeys, setApiKey, deleteApiKey } from "@/lib/api";
import type { ApiKeyInfo } from "@/lib/api";

interface ApiKeysTabProps {
  isActive: boolean;
}

export function ApiKeysTab({ isActive }: ApiKeysTabProps) {
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  // Track which keys the user is actively editing, and their input values
  const [editingValues, setEditingValues] = useState<Record<string, string>>(
    {}
  );
  const [savingKeys, setSavingKeys] = useState<Set<string>>(new Set());

  const fetchKeys = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await getApiKeys();
      setKeys(response.keys);
    } catch (error) {
      console.error("Failed to fetch API keys:", error);
      toast.error("Failed to load API keys");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isActive) {
      fetchKeys();
      // Reset editing state when tab activates
      setEditingValues({});
    }
  }, [isActive, fetchKeys]);

  const handleSave = async (keyInfo: ApiKeyInfo) => {
    const value = editingValues[keyInfo.id];
    if (!value?.trim()) return;

    setSavingKeys(prev => new Set(prev).add(keyInfo.id));
    try {
      const response = await setApiKey(keyInfo.id, value.trim());
      if (response.success) {
        toast.success(response.message);
        // Clear editing state and refresh
        setEditingValues(prev => {
          const next = { ...prev };
          delete next[keyInfo.id];
          return next;
        });
        await fetchKeys();
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to save API key"
      );
    } finally {
      setSavingKeys(prev => {
        const next = new Set(prev);
        next.delete(keyInfo.id);
        return next;
      });
    }
  };

  const handleDelete = async (keyInfo: ApiKeyInfo) => {
    setSavingKeys(prev => new Set(prev).add(keyInfo.id));
    try {
      const response = await deleteApiKey(keyInfo.id);
      if (response.success) {
        toast.success(response.message);
        setEditingValues(prev => {
          const next = { ...prev };
          delete next[keyInfo.id];
          return next;
        });
        await fetchKeys();
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to remove API key"
      );
    } finally {
      setSavingKeys(prev => {
        const next = new Set(prev);
        next.delete(keyInfo.id);
        return next;
      });
    }
  };

  const isEditing = (id: string) => id in editingValues;

  if (isLoading && keys.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">Loading API keys...</div>
    );
  }

  return (
    <div className="space-y-4">
      {keys.map(keyInfo => {
        const isEnvVar = keyInfo.source === "env_var";
        const isSaving = savingKeys.has(keyInfo.id);
        const currentlyEditing = isEditing(keyInfo.id);
        const inputValue = currentlyEditing ? editingValues[keyInfo.id] : "";

        return (
          <div key={keyInfo.id}>
            <div className="flex items-center gap-2">
              <label
                className="text-sm font-medium shrink-0"
                title={keyInfo.description}
              >
                {keyInfo.name}
              </label>
              {keyInfo.key_url && (
                <a
                  href={keyInfo.key_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground transition-colors shrink-0"
                  title="Get a token"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              )}
              {keyInfo.id === "civitai" && (
                <span
                  className="text-muted-foreground shrink-0"
                  title="Saved to disk to persist across restarts"
                >
                  <Info className="h-3.5 w-3.5" />
                </span>
              )}
              {isEnvVar ? (
                <Input
                  type="text"
                  disabled
                  placeholder={`Set via ${keyInfo.env_var} environment variable`}
                  className="flex-1 text-sm"
                />
              ) : (
                <>
                  <Input
                    type={currentlyEditing ? "password" : "text"}
                    placeholder="Enter token..."
                    value={
                      keyInfo.is_set && !currentlyEditing
                        ? "••••••••••••••••"
                        : inputValue
                    }
                    disabled={keyInfo.is_set && !currentlyEditing}
                    onChange={e =>
                      setEditingValues(prev => ({
                        ...prev,
                        [keyInfo.id]: e.target.value,
                      }))
                    }
                    onFocus={() => {
                      if (keyInfo.is_set && !currentlyEditing) {
                        // Start editing with empty value
                        setEditingValues(prev => ({
                          ...prev,
                          [keyInfo.id]: "",
                        }));
                      }
                    }}
                    className="flex-1 text-sm"
                  />
                  {currentlyEditing ? (
                    <Button
                      size="sm"
                      onClick={() => handleSave(keyInfo)}
                      disabled={!inputValue.trim() || isSaving}
                      className="gap-1.5"
                    >
                      <Save className="h-3.5 w-3.5" />
                      Save
                    </Button>
                  ) : keyInfo.is_set ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDelete(keyInfo)}
                      disabled={isSaving}
                      className="gap-1.5 text-muted-foreground hover:text-destructive"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  ) : null}
                </>
              )}
            </div>
          </div>
        );
      })}
      {keys.length === 0 && !isLoading && (
        <div className="text-sm text-muted-foreground">
          No API keys configured.
        </div>
      )}
    </div>
  );
}
