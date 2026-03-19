/**
 * Reads `node_values` messages from the WebRTC data channel and provides
 * the latest computed output values for each node to the graph editor.
 *
 * This hook replaces frontend-side evaluation (useSubgraphEval, useValueForwarding,
 * useParentValueBridge, computePatternValue) by consuming values computed by
 * the backend GraphEngine.
 */

import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";

/** Snapshot of all node output values keyed by node ID then port name. */
export type NodeValuesSnapshot = Record<string, Record<string, unknown>>;

interface UseBackendNodeValuesOptions {
  /** Ref to the mutable node values object from useWebRTC */
  nodeValuesRef: React.RefObject<NodeValuesSnapshot>;
  /** Subscribe to value change notifications */
  subscribeNodeValues: (listener: () => void) => () => void;
  /** Whether graph-mode backend evaluation is active */
  enabled: boolean;
}

/**
 * Provides a reactive snapshot of backend-computed node output values.
 *
 * Components call `getNodeValue(nodeId, portName)` to read the latest value
 * for a specific output port. The snapshot updates whenever the backend sends
 * a `node_values` message over the data channel.
 */
export function useBackendNodeValues({
  nodeValuesRef,
  subscribeNodeValues,
  enabled,
}: UseBackendNodeValuesOptions) {
  // Track a version number that increments on every update to trigger re-renders
  const versionRef = useRef(0);
  const listenerCallbacksRef = useRef(new Set<() => void>());

  useEffect(() => {
    if (!enabled) return;

    const unsubscribe = subscribeNodeValues(() => {
      versionRef.current += 1;
      for (const cb of listenerCallbacksRef.current) {
        cb();
      }
    });
    return unsubscribe;
  }, [enabled, subscribeNodeValues]);

  // useSyncExternalStore for a reactive snapshot
  const subscribe = useCallback((onStoreChange: () => void) => {
    listenerCallbacksRef.current.add(onStoreChange);
    return () => {
      listenerCallbacksRef.current.delete(onStoreChange);
    };
  }, []);

  const getSnapshot = useCallback((): NodeValuesSnapshot => {
    if (!enabled) return {};
    return nodeValuesRef.current ?? {};
  }, [enabled, nodeValuesRef]);

  const values = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  /** Get a specific node's output value for a given port. */
  const getNodeValue = useCallback(
    (nodeId: string, portName: string): unknown => {
      return values[nodeId]?.[portName] ?? null;
    },
    [values]
  );

  /** Get all output values for a specific node. */
  const getNodeOutputs = useCallback(
    (nodeId: string): Record<string, unknown> | undefined => {
      return values[nodeId];
    },
    [values]
  );

  return {
    /** Full snapshot of all node values */
    values,
    /** Get a single port value */
    getNodeValue,
    /** Get all outputs for a node */
    getNodeOutputs,
  };
}
