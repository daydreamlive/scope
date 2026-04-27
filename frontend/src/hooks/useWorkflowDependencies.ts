/**
 * Hooks for managing LoRA downloads and plugin installs during workflow import.
 */

import { useCallback } from "react";
import { toast } from "sonner";
import type {
  ScopeWorkflow,
  WorkflowLoRA,
  LoRADownloadRequest,
} from "../lib/workflowApi";
import {
  downloadLoRA,
  installPlugin,
  restartServer,
  waitForServer,
} from "../lib/api";
import { useDependencyTracker } from "./useDependencyTracker";

// ---------------------------------------------------------------------------
// LoRA downloads
// ---------------------------------------------------------------------------

export type LoRADownloadStatus = "idle" | "downloading" | "done" | "error";

function buildLoRADownloadRequest(
  lora: WorkflowLoRA
): LoRADownloadRequest | null {
  const prov = lora.provenance;
  if (!prov || prov.source === "local") return null;

  if (prov.source === "huggingface") {
    return {
      source: "huggingface",
      repo_id: prov.repo_id,
      hf_filename: prov.hf_filename,
      expected_sha256: lora.sha256,
    };
  }

  if (prov.source === "civitai") {
    // Try to use explicit version_id; if missing, extract from CivitAI URL.
    const versionId =
      prov.version_id ??
      prov.url?.match(/civitai\.com\/api\/download\/models\/(\d+)/)?.[1];

    if (versionId) {
      return {
        source: "civitai",
        model_id: prov.model_id,
        version_id: versionId,
        url: prov.url,
        expected_sha256: lora.sha256,
      };
    }

    // Fall back to direct URL download if we can't resolve a version_id.
    if (prov.url) {
      return {
        source: "url",
        url: prov.url,
        expected_sha256: lora.sha256,
      };
    }
  }

  if (prov.source === "url" && prov.url) {
    return {
      source: "url",
      url: prov.url,
      expected_sha256: lora.sha256,
    };
  }

  return null;
}

export function useLoRADownloads(
  workflow: ScopeWorkflow | null,
  onDownloadComplete?: () => void | Promise<void>
) {
  const { statuses, initialize, setStatus, getPending, reset, someActive } =
    useDependencyTracker("downloading");

  const downloadOne = useCallback(
    async (filename: string) => {
      if (!workflow) return;

      const lora = workflow.pipelines
        .flatMap(p => p.loras)
        .find(l => l.filename === filename);
      if (!lora) return;

      const req = buildLoRADownloadRequest(lora);
      if (!req) {
        toast.error("No download source available for this LoRA");
        return;
      }

      setStatus(filename, "downloading");
      try {
        await downloadLoRA(req);
        setStatus(filename, "done");
        toast.success(`Downloaded ${filename}`);
        if (onDownloadComplete) await onDownloadComplete();
      } catch (err) {
        setStatus(filename, "error");
        toast.error(`Failed to download ${filename}`, {
          description: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [workflow, onDownloadComplete, setStatus]
  );

  const downloadAll = useCallback(async () => {
    const pending = getPending();
    await Promise.allSettled(pending.map(f => downloadOne(f)));
  }, [getPending, downloadOne]);

  return {
    downloads: statuses as Record<string, LoRADownloadStatus>,
    initialize,
    downloadOne,
    downloadAll,
    reset,
    someDownloading: someActive,
  };
}

// ---------------------------------------------------------------------------
// Plugin installs
// ---------------------------------------------------------------------------

export type PluginInstallStatus = "idle" | "installing" | "done" | "error";

function findPluginInstallSpec(
  workflow: ScopeWorkflow,
  pluginName: string
): string | null {
  const sources = [
    ...workflow.pipelines.map(p => p.source),
    ...(workflow.nodes ?? []).map(n => n.source),
  ];
  for (const source of sources) {
    if (source.plugin_name === pluginName) {
      return source.package_spec ?? source.plugin_name ?? null;
    }
  }
  return null;
}

export function usePluginInstalls(
  workflow: ScopeWorkflow | null,
  onRestartComplete?: () => void | Promise<void>,
  confirmInstall?: (installSpec: string) => Promise<boolean>
) {
  const { statuses, initialize, setStatus, getPending, reset, someActive } =
    useDependencyTracker("installing");

  const doRestartServer = useCallback(async () => {
    toast.info("Restarting server to load new plugins...", {
      id: "plugin-server-restart",
    });
    try {
      const oldStartTime = await restartServer();
      await waitForServer(oldStartTime);
      toast.success("Server restarted successfully");
      if (onRestartComplete) await onRestartComplete();
    } catch {
      toast.error(
        "Server did not restart in time. You may need to restart manually."
      );
    }
  }, [onRestartComplete]);

  const installOne = useCallback(
    async (pluginName: string, { skipRestart = false } = {}) => {
      if (!workflow) return;

      const installSpec = findPluginInstallSpec(workflow, pluginName);
      if (!installSpec) {
        toast.error("No install source available for this plugin");
        return;
      }

      if (confirmInstall) {
        const confirmed = await confirmInstall(installSpec);
        if (!confirmed) return;
      }

      setStatus(pluginName, "installing");
      try {
        const result = await installPlugin({ package: installSpec });
        if (!result.success) {
          throw new Error(result.message || "Installation failed");
        }
        setStatus(pluginName, "done");
        toast.success(`Installed ${pluginName}`);
        if (!skipRestart) {
          await doRestartServer();
        }
      } catch (err) {
        setStatus(pluginName, "error");
        toast.error(`Failed to install ${pluginName}`, {
          description: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [workflow, doRestartServer, confirmInstall, setStatus]
  );

  const installAll = useCallback(async () => {
    const pending = getPending();
    await Promise.allSettled(
      pending.map(name => installOne(name, { skipRestart: true }))
    );
    await doRestartServer();
  }, [getPending, installOne, doRestartServer]);

  return {
    installs: statuses as Record<string, PluginInstallStatus>,
    initialize,
    installOne,
    installAll,
    reset,
    someInstalling: someActive,
  };
}
