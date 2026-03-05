/**
 * Generic state management for dependency resolution (LoRA downloads, plugin installs).
 *
 * Tracks a set of named items through idle -> active -> done/error transitions.
 */

import { useState, useCallback } from "react";

export function useDependencyTracker<
  TStatus extends string = "idle" | "done" | "error",
>(activeStatus: TStatus) {
  const [statuses, setStatuses] = useState<Record<string, string>>({});

  const initialize = useCallback(
    (names: string[]) => {
      const initial: Record<string, string> = {};
      for (const name of names) {
        initial[name] = "idle";
      }
      setStatuses(initial);
    },
    [setStatuses]
  );

  const setStatus = useCallback(
    (name: string, status: string) => {
      setStatuses(prev => ({ ...prev, [name]: status }));
    },
    [setStatuses]
  );

  const getPending = useCallback(
    () =>
      Object.entries(statuses)
        .filter(([, s]) => s === "idle" || s === "error")
        .map(([name]) => name),
    [statuses]
  );

  const reset = useCallback(() => setStatuses({}), []);

  const someActive = Object.values(statuses).some(s => s === activeStatus);

  return { statuses, initialize, setStatus, getPending, reset, someActive };
}
