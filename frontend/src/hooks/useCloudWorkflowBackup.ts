/**
 * Auto-saves the current workflow to localStorage while cloud mode is connected.
 *
 * On cloud disconnect (manual or unexpected), marks the backup as needing
 * restore. When cloud reconnects, offers the user a toast to restore the
 * previous session state via the same workflow import path.
 */

import { useEffect, useRef } from "react";
import { toast } from "sonner";
import {
  buildScopeWorkflow,
  workflowToSettings,
  workflowTimelineToPrompts,
  workflowToPromptState,
  type WorkflowPromptState,
} from "../lib/workflowSettings";
import type { ScopeWorkflow } from "../lib/workflowApi";
import type { SettingsState, PipelineInfo } from "../types";
import type { TimelinePrompt } from "../components/PromptTimeline";
import type { LoRAFileInfo, PluginInfo } from "../lib/api";

const BACKUP_KEY = "scope-cloud-workflow-backup";
const DISCONNECT_FLAG_KEY = "scope-cloud-workflow-disconnected";
const DEBOUNCE_MS = 2000;
const RESTORE_TOAST_DURATION_MS = 30_000;

interface BackupData {
  workflow: ScopeWorkflow;
  savedAt: string;
  pipelineId: string;
}

function readBackup(): BackupData | null {
  try {
    const raw = localStorage.getItem(BACKUP_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function writeBackup(data: BackupData): void {
  try {
    localStorage.setItem(BACKUP_KEY, JSON.stringify(data));
  } catch (err) {
    console.error("[CloudWorkflowBackup] Failed to save:", err);
  }
}

function clearBackup(): void {
  localStorage.removeItem(BACKUP_KEY);
  localStorage.removeItem(DISCONNECT_FLAG_KEY);
}

function markDisconnected(): void {
  localStorage.setItem(DISCONNECT_FLAG_KEY, "true");
}

function isMarkedDisconnected(): boolean {
  return localStorage.getItem(DISCONNECT_FLAG_KEY) === "true";
}

interface UseCloudWorkflowBackupOptions {
  settings: SettingsState;
  timelinePrompts: TimelinePrompt[];
  promptState: WorkflowPromptState;
  pipelineInfoMap: Record<string, PipelineInfo> | null;
  loraFiles: LoRAFileInfo[];
  plugins: PluginInfo[];
  scopeVersion: string;
  isCloudConnected: boolean;
  onRestore: (
    settings: Partial<SettingsState>,
    timelinePrompts: TimelinePrompt[],
    promptState: WorkflowPromptState | null
  ) => void;
}

export function useCloudWorkflowBackup({
  settings,
  timelinePrompts,
  promptState,
  pipelineInfoMap,
  loraFiles,
  plugins,
  scopeVersion,
  isCloudConnected,
  onRestore,
}: UseCloudWorkflowBackupOptions) {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevConnectedRef = useRef(false);
  const restoreOfferPendingRef = useRef(false);

  // Auto-save workflow to localStorage while cloud is connected.
  // Debounced to avoid writing on every keystroke / slider drag.
  useEffect(() => {
    if (!isCloudConnected || !pipelineInfoMap) return;

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      if (restoreOfferPendingRef.current) return;

      try {
        const pluginInfoMap = new Map<string, PluginInfo>(
          plugins.map(p => [p.name, p])
        );

        const workflow = buildScopeWorkflow({
          name: "Cloud Session Backup",
          settings,
          timelinePrompts,
          promptState,
          pipelineInfoMap,
          loraFiles,
          pluginInfoMap,
          scopeVersion: scopeVersion ?? "unknown",
        });

        writeBackup({
          workflow,
          savedAt: new Date().toISOString(),
          pipelineId: settings.pipelineId,
        });
      } catch (err) {
        console.error("[CloudWorkflowBackup] Failed to build workflow:", err);
      }
    }, DEBOUNCE_MS);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [
    isCloudConnected,
    settings,
    timelinePrompts,
    promptState,
    pipelineInfoMap,
    loraFiles,
    plugins,
    scopeVersion,
  ]);

  // Detect disconnect → reconnect transitions and offer restore.
  useEffect(() => {
    const wasConnected = prevConnectedRef.current;
    prevConnectedRef.current = isCloudConnected;

    if (wasConnected && !isCloudConnected) {
      markDisconnected();
      return;
    }

    if (!wasConnected && isCloudConnected && isMarkedDisconnected()) {
      const backup = readBackup();
      if (!backup) {
        localStorage.removeItem(DISCONNECT_FLAG_KEY);
        return;
      }

      restoreOfferPendingRef.current = true;

      const finishOffer = () => {
        restoreOfferPendingRef.current = false;
        localStorage.removeItem(DISCONNECT_FLAG_KEY);
      };

      toast("Restore previous cloud session?", {
        description: `Pipeline: ${backup.pipelineId}`,
        action: {
          label: "Restore",
          onClick: () => {
            const importedSettings = workflowToSettings(
              backup.workflow,
              loraFiles
            );
            const importedTimeline = workflowTimelineToPrompts(
              backup.workflow.timeline
            );
            const importedPromptState = workflowToPromptState(backup.workflow);
            onRestore(importedSettings, importedTimeline, importedPromptState);
            clearBackup();
            finishOffer();
            toast.success("Cloud session restored");
          },
        },
        duration: RESTORE_TOAST_DURATION_MS,
        onDismiss: () => {
          clearBackup();
          finishOffer();
        },
        onAutoClose: () => {
          clearBackup();
          finishOffer();
        },
      });
    }
  }, [isCloudConnected, loraFiles, onRestore]);
}
