import { useState, useEffect, useCallback, useRef } from "react";
import type { ShortcutDefinition } from "../../lib/shortcuts";
import { SHORTCUTS, getShortcutsByCategory } from "../../lib/shortcuts";
import {
  getEffectiveShortcuts,
  saveOverride,
  resetOverride,
  resetAllOverrides,
  findConflict,
  buildKeysString,
  type ShortcutOverride,
} from "../../lib/shortcutOverrides";

interface RecordingState {
  shortcutId: string;
  /** Captured override so far (null = waiting for keypress) */
  captured: ShortcutOverride | null;
  conflict: ShortcutDefinition | null;
}

function formatKeyEvent(e: KeyboardEvent): ShortcutOverride | null {
  // Ignore bare modifier keypresses
  if (["Meta", "Control", "Shift", "Alt"].includes(e.key)) return null;

  return {
    key: e.key.length === 1 ? e.key.toLowerCase() : e.key,
    metaOrCtrl: e.metaKey || e.ctrlKey || undefined,
    shift: e.shiftKey || undefined,
    alt: e.altKey || undefined,
  };
}

export function ShortcutsTab() {
  const [shortcuts, setShortcuts] = useState(getEffectiveShortcuts);
  const [recording, setRecording] = useState<RecordingState | null>(null);
  const recordingRef = useRef(recording);
  recordingRef.current = recording;

  const refresh = useCallback(() => {
    setShortcuts(getEffectiveShortcuts());
  }, []);

  // Global keydown listener for recording mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const rec = recordingRef.current;
      if (!rec) return;

      e.preventDefault();
      e.stopPropagation();

      // Escape cancels recording
      if (e.key === "Escape") {
        setRecording(null);
        return;
      }

      const override = formatKeyEvent(e);
      if (!override) return;

      const conflict = findConflict(rec.shortcutId, override);
      setRecording({
        ...rec,
        captured: override,
        conflict: conflict ?? null,
      });
    };

    if (recording) {
      window.addEventListener("keydown", handleKeyDown, true);
      return () => window.removeEventListener("keydown", handleKeyDown, true);
    }
  }, [recording]);

  const handleStartRecording = (id: string) => {
    setRecording({ shortcutId: id, captured: null, conflict: null });
  };

  const handleConfirm = () => {
    if (!recording?.captured) return;
    saveOverride(recording.shortcutId, recording.captured);
    setRecording(null);
    refresh();
  };

  const handleCancel = () => {
    setRecording(null);
  };

  const handleResetOne = (id: string) => {
    resetOverride(id);
    refresh();
  };

  const handleResetAll = () => {
    resetAllOverrides();
    refresh();
  };

  const categories = getShortcutsByCategory(shortcuts);

  // Check if a shortcut differs from its default
  const isOverridden = (id: string): boolean => {
    const effective = shortcuts.find(s => s.id === id);
    const original = SHORTCUTS.find(s => s.id === id);
    if (!effective || !original) return false;
    return effective.keys !== original.keys;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Keyboard Shortcuts</h2>
          <p className="text-sm text-muted-foreground">
            Customize keyboard shortcuts. Click &quot;Record&quot; to set a new
            key combination.
          </p>
        </div>
        <button
          onClick={handleResetAll}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded border border-border hover:border-foreground/20"
        >
          Reset All
        </button>
      </div>

      {categories.map(({ category, label, items }) => (
        <div key={category}>
          <h3 className="text-[11px] font-semibold tracking-widest text-muted-foreground uppercase mb-3">
            {label}
          </h3>
          <div className="space-y-1">
            {items.map(shortcut => {
              const isRecording = recording?.shortcutId === shortcut.id;
              const overridden = isOverridden(shortcut.id);

              return (
                <div
                  key={shortcut.id}
                  className={`flex items-center gap-3 py-2 px-3 rounded-md ${
                    isRecording
                      ? "bg-accent/50 ring-1 ring-accent"
                      : "hover:bg-muted/50"
                  }`}
                >
                  {/* Label & description */}
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium">
                      {shortcut.label}
                      {shortcut.builtIn && (
                        <span className="ml-1.5 text-[10px] text-muted-foreground">
                          (built-in)
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground truncate">
                      {shortcut.description}
                    </div>
                  </div>

                  {/* Current binding */}
                  <div className="flex items-center gap-2">
                    {isRecording && recording.captured ? (
                      <div className="flex items-center gap-2">
                        <kbd className="inline-flex items-center gap-1 rounded border border-accent bg-accent/20 px-2 py-0.5 font-mono text-[11px] text-foreground">
                          {buildKeysString(recording.captured)}
                        </kbd>
                        {recording.conflict && (
                          <span className="text-[11px] text-destructive">
                            Conflicts with &quot;{recording.conflict.label}
                            &quot;
                          </span>
                        )}
                        <button
                          onClick={handleConfirm}
                          disabled={!!recording.conflict}
                          className="text-xs px-2 py-0.5 rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          Save
                        </button>
                        <button
                          onClick={handleCancel}
                          className="text-xs px-2 py-0.5 rounded border border-border hover:bg-muted"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : isRecording ? (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground animate-pulse">
                          Press a key combo...
                        </span>
                        <button
                          onClick={handleCancel}
                          className="text-xs px-2 py-0.5 rounded border border-border hover:bg-muted"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <>
                        <kbd
                          className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 font-mono text-[11px] ${
                            overridden
                              ? "border-accent bg-accent/10 text-foreground"
                              : "border-border bg-muted text-muted-foreground"
                          }`}
                        >
                          {shortcut.keys}
                        </kbd>
                        {!shortcut.builtIn && (
                          <button
                            onClick={() => handleStartRecording(shortcut.id)}
                            className="text-xs px-2 py-0.5 rounded border border-border text-muted-foreground hover:text-foreground hover:border-foreground/20 transition-colors"
                          >
                            Record
                          </button>
                        )}
                        {overridden && (
                          <button
                            onClick={() => handleResetOne(shortcut.id)}
                            className="text-[10px] px-1.5 py-0.5 rounded text-muted-foreground hover:text-foreground transition-colors"
                            title="Reset to default"
                          >
                            Reset
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
