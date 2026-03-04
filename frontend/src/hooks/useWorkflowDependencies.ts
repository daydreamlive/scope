/**
 * Hooks for managing LoRA downloads and plugin installs during workflow import.
 */

import { useState, useCallback } from "react";
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
    return {
      source: "civitai",
      model_id: prov.model_id,
      version_id: prov.version_id,
      expected_sha256: lora.sha256,
    };
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

export function useLoRADownloads(workflow: ScopeWorkflow | null) {
  const [downloads, setDownloads] = useState<
    Record<string, LoRADownloadStatus>
  >({});

  const initialize = useCallback(
    (missingFilenames: string[]) => {
      const initial: Record<string, LoRADownloadStatus> = {};
      for (const name of missingFilenames) {
        initial[name] = "idle";
      }
      setDownloads(initial);
    },
    [setDownloads]
  );

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

      setDownloads(prev => ({ ...prev, [filename]: "downloading" }));
      try {
        await downloadLoRA(req);
        setDownloads(prev => ({ ...prev, [filename]: "done" }));
        toast.success(`Downloaded ${filename}`);
      } catch (err) {
        setDownloads(prev => ({ ...prev, [filename]: "error" }));
        toast.error(`Failed to download ${filename}`, {
          description: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [workflow]
  );

  const downloadAll = useCallback(async () => {
    const pending = Object.entries(downloads)
      .filter(([, s]) => s === "idle" || s === "error")
      .map(([name]) => name);
    await Promise.allSettled(pending.map(f => downloadOne(f)));
  }, [downloads, downloadOne]);

  const reset = useCallback(() => setDownloads({}), []);

  const someDownloading = Object.values(downloads).some(
    s => s === "downloading"
  );

  return {
    downloads,
    initialize,
    downloadOne,
    downloadAll,
    reset,
    someDownloading,
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
  for (const p of workflow.pipelines) {
    if (p.source.plugin_name === pluginName) {
      return p.source.package_spec ?? p.source.plugin_name ?? null;
    }
  }
  return null;
}

export function usePluginInstalls(workflow: ScopeWorkflow | null) {
  const [installs, setInstalls] = useState<Record<string, PluginInstallStatus>>(
    {}
  );

  const initialize = useCallback(
    (pluginNames: string[]) => {
      const initial: Record<string, PluginInstallStatus> = {};
      for (const name of pluginNames) {
        initial[name] = "idle";
      }
      setInstalls(initial);
    },
    [setInstalls]
  );

  const doRestartServer = useCallback(async () => {
    toast.info("Restarting server to load new plugins...");
    try {
      const oldStartTime = await restartServer();
      await waitForServer(oldStartTime);
      toast.success("Server restarted successfully");
    } catch {
      toast.error(
        "Server did not restart in time. You may need to restart manually."
      );
    }
  }, []);

  const installOne = useCallback(
    async (pluginName: string, { skipRestart = false } = {}) => {
      if (!workflow) return;

      const installSpec = findPluginInstallSpec(workflow, pluginName);
      if (!installSpec) {
        toast.error("No install source available for this plugin");
        return;
      }

      if (
        !window.confirm(
          `This will install the package "${installSpec}" via pip. Only proceed if you trust the workflow source.\n\nContinue?`
        )
      ) {
        return;
      }

      setInstalls(prev => ({ ...prev, [pluginName]: "installing" }));
      try {
        const result = await installPlugin({ package: installSpec });
        if (!result.success) {
          throw new Error(result.message || "Installation failed");
        }
        setInstalls(prev => ({ ...prev, [pluginName]: "done" }));
        toast.success(`Installed ${pluginName}`);
        if (!skipRestart) {
          await doRestartServer();
        }
      } catch (err) {
        setInstalls(prev => ({ ...prev, [pluginName]: "error" }));
        toast.error(`Failed to install ${pluginName}`, {
          description: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [workflow, doRestartServer]
  );

  const installAll = useCallback(async () => {
    const pending = Object.entries(installs)
      .filter(([, s]) => s === "idle" || s === "error")
      .map(([name]) => name);
    await Promise.allSettled(
      pending.map(name => installOne(name, { skipRestart: true }))
    );
    await doRestartServer();
  }, [installs, installOne, doRestartServer]);

  const reset = useCallback(() => setInstalls({}), []);

  const someInstalling = Object.values(installs).some(s => s === "installing");

  return {
    installs,
    initialize,
    installOne,
    installAll,
    reset,
    someInstalling,
  };
}
