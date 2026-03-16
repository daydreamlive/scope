/**
 * useParentValueBridge
 *
 * When the user is inside a subgraph, parent-level producer nodes (MIDI,
 * animated controls, sliders, etc.) are "frozen" in the navigation stack and
 * their React components are unmounted.  This hook re-establishes live
 * connections to those producers so that values flow into the boundary input
 * node's `portValues` on the current canvas.
 */

import { useEffect, useRef, useCallback } from "react";
import type { Node } from "@xyflow/react";
import type {
  FlowNodeData,
  SubgraphPort,
  SerializedSubgraphNode,
  SerializedSubgraphEdge,
} from "../../../../lib/graphUtils";
import { parseHandleId, buildHandleId } from "../../../../lib/graphUtils";
import { getAnyValueFromNode } from "../../utils/getValueFromNode";
import type { GraphLevel } from "../subgraph/useGraphNavigation";
import { BOUNDARY_INPUT_ID } from "../subgraph/useGraphNavigation";
import { computePatternValue } from "../../utils/computePatternValue";
import { evaluateInnerGraph } from "../subgraph/useSubgraphEval";
import type { MidiChannelDef } from "../../nodes/MidiNode";

/* ── Types ────────────────────────────────────────────────────────────────── */

interface SourceMapping {
  portName: string;
  sourceNode: Node<FlowNodeData>;
  sourceHandle: string | null | undefined;
}

type SetNodes = (
  updater: (nds: Node<FlowNodeData>[]) => Node<FlowNodeData>[]
) => void;

/* ── Hook ─────────────────────────────────────────────────────────────────── */

export function useParentValueBridge(
  stackRef: { readonly current: GraphLevel[] },
  depth: number,
  setNodes: SetNodes
) {
  const setNodesRef = useRef(setNodes);
  setNodesRef.current = setNodes;

  const lastControlValues = useRef<Record<string, number>>({});

  // Accumulator for batched writes
  const pendingValues = useRef<Record<string, unknown>>({});
  const rafHandle = useRef<number | null>(null);

  // Live MIDI values keyed by "nodeId:handleName" – updated by MIDI listeners
  const liveMidiValues = useRef<Record<string, number>>({});

  const buildMappings = useCallback((): SourceMapping[] => {
    const stack = stackRef.current;
    if (stack.length === 0) return [];

    const top = stack[stack.length - 1];
    const sgNodeId = top.subgraphNodeId;
    const parentEdges = top.edges;
    const parentNodes = top.nodes;

    const sgNode = parentNodes.find(n => n.id === sgNodeId);
    if (!sgNode) return [];
    const inputPorts: SubgraphPort[] = sgNode.data.subgraphInputs ?? [];

    const mappings: SourceMapping[] = [];
    for (const port of inputPorts) {
      if (port.portType !== "param") continue;
      const handleId = buildHandleId("param", port.name);
      const edge = parentEdges.find(
        e => e.target === sgNodeId && e.targetHandle === handleId
      );
      if (!edge) continue;
      const srcNode = parentNodes.find(n => n.id === edge.source);
      if (!srcNode) continue;
      mappings.push({
        portName: port.name,
        sourceNode: srcNode,
        sourceHandle: edge.sourceHandle,
      });
    }
    return mappings;
  }, [stackRef]);

  // Flush pending values into the boundary input node
  const flush = useCallback(() => {
    const vals = { ...pendingValues.current };
    if (Object.keys(vals).length === 0) {
      rafHandle.current = null;
      return;
    }
    pendingValues.current = {};
    rafHandle.current = null;

    setNodesRef.current(nds =>
      nds.map(n => {
        if (n.id !== BOUNDARY_INPUT_ID) return n;
        const prev = (n.data.portValues ?? {}) as Record<string, unknown>;
        let changed = false;
        for (const [k, v] of Object.entries(vals)) {
          if (prev[k] !== v) {
            changed = true;
            break;
          }
        }
        if (!changed) return n;
        return {
          ...n,
          data: { ...n.data, portValues: { ...prev, ...vals } },
        };
      })
    );
  }, []);

  const scheduleFlush = useCallback(() => {
    if (rafHandle.current !== null) return;
    rafHandle.current = requestAnimationFrame(flush);
  }, [flush]);

  const writeValue = useCallback(
    (portName: string, value: unknown) => {
      pendingValues.current[portName] = value;
      scheduleFlush();
    },
    [scheduleFlush]
  );

  // ── MIDI bridge ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (depth === 0) return;
    const mappings = buildMappings();

    // Collect ALL MIDI nodes referenced in the parent stack (direct or
    // upstream of a subgraph source) so we can set up listeners for them.
    const stack = stackRef.current;
    const top = stack[stack.length - 1];
    const parentNodes = top?.nodes ?? [];
    const parentEdges = top?.edges ?? [];

    // Direct MIDI → boundary mappings
    const directMidiMappings = mappings.filter(
      m => m.sourceNode.data.nodeType === "midi"
    );

    // Also find all MIDI nodes that feed into subgraph source nodes
    const sgMappings = mappings.filter(
      m => m.sourceNode.data.nodeType === "subgraph"
    );
    const indirectMidiNodeIds = new Set<string>();
    for (const sgMapping of sgMappings) {
      for (const edge of parentEdges) {
        if (edge.target !== sgMapping.sourceNode.id) continue;
        const srcNode = parentNodes.find(n => n.id === edge.source);
        if (srcNode?.data.nodeType === "midi") {
          indirectMidiNodeIds.add(srcNode.id);
        }
      }
    }

    // Merge: all unique MIDI nodes that need listeners
    const allMidiNodeIds = new Set<string>();
    for (const m of directMidiMappings) allMidiNodeIds.add(m.sourceNode.id);
    for (const id of indirectMidiNodeIds) allMidiNodeIds.add(id);

    if (allMidiNodeIds.size === 0) return;

    let midiAccess: MIDIAccess | null = null;
    let cleanups: (() => void)[] = [];

    const setup = async () => {
      if (!navigator.requestMIDIAccess) return;
      try {
        midiAccess = await navigator.requestMIDIAccess({ sysex: false });
      } catch {
        return;
      }

      for (const midiNodeId of allMidiNodeIds) {
        const midiNode = parentNodes.find(n => n.id === midiNodeId);
        if (!midiNode) continue;
        const deviceId = midiNode.data.midiDeviceId as string | undefined;
        if (!deviceId) continue;
        const channels = (midiNode.data.midiChannels ?? []) as MidiChannelDef[];
        const input = midiAccess.inputs.get(deviceId);
        if (!input) continue;

        const handler = (event: MIDIMessageEvent) => {
          const d = event.data;
          if (!d || d.length < 2) return;
          const status = d[0];
          const command = status & 0xf0;
          const midiChannel = status & 0x0f;
          const noteOrCC = d[1];
          const value = d.length > 2 ? d[2] : 0;
          const normalized = value / 127;

          for (let chIdx = 0; chIdx < channels.length; chIdx++) {
            const chDef = channels[chIdx];
            if (midiChannel !== chDef.channel) continue;
            let matched = false;
            let val = 0;
            if (
              chDef.type === "cc" &&
              command === 0xb0 &&
              chDef.cc === noteOrCC
            ) {
              matched = true;
              val = normalized;
            }
            if (chDef.type === "note" && chDef.cc === noteOrCC) {
              if (command === 0x90 && value > 0) {
                matched = true;
                val = normalized;
              } else if (
                command === 0x80 ||
                (command === 0x90 && value === 0)
              ) {
                matched = true;
                val = 0;
              }
            }
            if (matched) {
              // Store in shared live store
              liveMidiValues.current[`${midiNodeId}:midi_${chIdx}`] = val;
              // Also directly write for direct MIDI → boundary mappings
              for (const dm of directMidiMappings) {
                if (dm.sourceNode.id !== midiNodeId) continue;
                const parsed = parseHandleId(dm.sourceHandle);
                if (parsed && parsed.name === `midi_${chIdx}`) {
                  writeValue(dm.portName, val);
                }
              }
            }
          }
        };

        input.addEventListener("midimessage", handler as EventListener);
        cleanups.push(() =>
          input.removeEventListener("midimessage", handler as EventListener)
        );
      }
    };

    setup();
    return () => {
      cleanups.forEach(fn => fn());
      cleanups = [];
    };
  }, [depth, buildMappings, writeValue, stackRef]);

  // ── Control animation bridge ─────────────────────────────────────────────
  useEffect(() => {
    if (depth === 0) return;
    const mappings = buildMappings();
    const controlMappings = mappings.filter(
      m =>
        m.sourceNode.data.nodeType === "control" &&
        m.sourceNode.data.isPlaying === true
    );
    if (controlMappings.length === 0) return;

    let running = true;
    let handle: number;

    const animate = () => {
      if (!running) return;
      const now = Date.now();

      for (const mapping of controlMappings) {
        const sn = mapping.sourceNode.data;
        const pattern = (sn.controlPattern ?? "sine") as
          | "sine"
          | "bounce"
          | "random_walk"
          | "linear"
          | "step";
        const speed = (sn.controlSpeed as number) ?? 1.0;
        const min = (sn.controlMin as number) ?? 0;
        const max = (sn.controlMax as number) ?? 1.0;
        const controlType = (sn.controlType as string) ?? "float";
        const t = now / 1000;
        const last = lastControlValues.current[mapping.portName] ?? min;

        const raw = computePatternValue(pattern, t, speed, min, max, last);
        lastControlValues.current[mapping.portName] = raw;
        const final = controlType === "int" ? Math.round(raw) : raw;
        writeValue(mapping.portName, final);
      }

      handle = requestAnimationFrame(animate);
    };

    handle = requestAnimationFrame(animate);
    return () => {
      running = false;
      cancelAnimationFrame(handle);
    };
  }, [depth, buildMappings, writeValue]);

  // ── Timeline animation bridge ────────────────────────────────────────────
  // When the parent has a playing timeline that feeds into the current
  // subgraph, we need to keep its playback alive and fire triggers.
  useEffect(() => {
    if (depth === 0) return;
    const mappings = buildMappings();
    const timelineMappings = mappings.filter(
      m =>
        m.sourceNode.data.nodeType === "timeline" &&
        m.sourceNode.data.isPlaying === true
    );
    if (timelineMappings.length === 0) return;

    // Group by source timeline node
    const byTimeline = new Map<string, SourceMapping[]>();
    for (const m of timelineMappings) {
      const arr = byTimeline.get(m.sourceNode.id) ?? [];
      arr.push(m);
      byTimeline.set(m.sourceNode.id, arr);
    }

    // Per-timeline state
    const timelineState = new Map<
      string,
      {
        startTime: number;
        offset: number;
        duration: number;
        loop: boolean;
        triggers: Array<{ id: string; time: number }>;
        firedSet: Set<string>;
      }
    >();

    for (const [nodeId, _portMappings] of byTimeline) {
      const srcNode = mappings.find(m => m.sourceNode.id === nodeId)
        ?.sourceNode;
      if (!srcNode) continue;
      const d = srcNode.data;
      const duration = (d.timelineDuration as number) ?? 10;
      const loop = (d.timelineLoop as boolean) ?? false;
      const triggers = (d.timelineTriggers ?? []) as Array<{
        id: string;
        time: number;
      }>;

      // Reconstruct position from wall-clock timestamps if available
      const wallStart = d._timelineWallStart as number | undefined;
      const wallOffset = (d._timelineWallOffset as number) ?? 0;
      let reconstructedOffset: number;

      if (wallStart) {
        const wallElapsed = (Date.now() - wallStart) / 1000;
        reconstructedOffset = wallOffset + wallElapsed;
        if (loop && duration > 0) {
          reconstructedOffset = reconstructedOffset % duration;
        } else if (reconstructedOffset > duration) {
          reconstructedOffset = duration;
        }
      } else {
        reconstructedOffset = (d.timelineCurrentTime as number) ?? 0;
      }

      const firedSet = new Set<string>();
      // Mark triggers already past the current position
      for (const t of triggers) {
        if (t.time <= reconstructedOffset) firedSet.add(t.id);
      }
      timelineState.set(nodeId, {
        startTime: performance.now(),
        offset: reconstructedOffset,
        duration,
        loop,
        triggers,
        firedSet,
      });
    }

    let running = true;
    let handle: number;

    const animate = () => {
      if (!running) return;

      for (const [nodeId, state] of timelineState) {
        const elapsed =
          (performance.now() - state.startTime) / 1000;
        let newTime = state.offset + elapsed;

        if (newTime >= state.duration) {
          if (state.loop) {
            newTime = newTime % state.duration;
            state.startTime = performance.now();
            state.offset = 0;
            state.firedSet.clear();
          } else {
            newTime = state.duration;
            // Timeline finished — write 0s for all triggers
            const portMaps = byTimeline.get(nodeId) ?? [];
            for (const pm of portMaps) {
              writeValue(pm.portName, 0);
            }
            continue;
          }
        }

        // Check triggers
        for (const trigger of state.triggers) {
          if (
            trigger.time <= newTime &&
            !state.firedSet.has(trigger.id)
          ) {
            state.firedSet.add(trigger.id);

            // Find port mapping for this trigger
            const portMaps = byTimeline.get(nodeId) ?? [];
            for (const pm of portMaps) {
              const parsed = parseHandleId(pm.sourceHandle);
              if (parsed && parsed.name === `trigger_${trigger.id}`) {
                writeValue(pm.portName, 1);
                // Reset after brief pulse
                setTimeout(() => {
                  writeValue(pm.portName, 0);
                }, 50);
              }
            }
          }
        }
      }

      handle = requestAnimationFrame(animate);
    };

    handle = requestAnimationFrame(animate);
    return () => {
      running = false;
      cancelAnimationFrame(handle);
    };
  }, [depth, buildMappings, writeValue]);

  // ── Subgraph source evaluation (rAF loop) ──────────────────────────────
  useEffect(() => {
    if (depth === 0) return;
    const mappings = buildMappings();
    const sgMappings = mappings.filter(
      m => m.sourceNode.data.nodeType === "subgraph"
    );
    if (sgMappings.length === 0) return;

    const stack = stackRef.current;
    const top = stack[stack.length - 1];
    const parentNodes = top?.nodes ?? [];
    const parentEdges = top?.edges ?? [];

    // Group mappings by source subgraph node id
    const bySource = new Map<string, SourceMapping[]>();
    for (const m of sgMappings) {
      const arr = bySource.get(m.sourceNode.id) ?? [];
      arr.push(m);
      bySource.set(m.sourceNode.id, arr);
    }

    let running = true;
    let handle: number;

    const evaluate = () => {
      if (!running) return;

      for (const [sgId, portMappings] of bySource) {
        const sgNode = parentNodes.find(n => n.id === sgId);
        if (!sgNode) continue;

        const sgInputs: SubgraphPort[] = sgNode.data.subgraphInputs ?? [];
        const sgOutputs: SubgraphPort[] = sgNode.data.subgraphOutputs ?? [];
        const innerNodes = (sgNode.data.subgraphNodes ??
          []) as SerializedSubgraphNode[];
        const innerEdges = (sgNode.data.subgraphEdges ??
          []) as SerializedSubgraphEdge[];
        if (innerNodes.length === 0) continue;

        // Build input values for the source subgraph by reading from its
        // upstream parent nodes, using live MIDI values where available.
        const inputPortValues: Record<string, unknown> = {};
        for (const port of sgInputs) {
          if (port.portType !== "param") continue;
          const handleId = buildHandleId("param", port.name);
          const edge = parentEdges.find(
            e => e.target === sgId && e.targetHandle === handleId
          );
          if (!edge) continue;
          const srcNode = parentNodes.find(n => n.id === edge.source);
          if (!srcNode) continue;

          const srcParsed = parseHandleId(edge.sourceHandle);
          if (!srcParsed) continue;

          // Check live MIDI store first
          const midiKey = `${srcNode.id}:${srcParsed.name}`;
          if (
            srcNode.data.nodeType === "midi" &&
            liveMidiValues.current[midiKey] !== undefined
          ) {
            inputPortValues[port.name] = liveMidiValues.current[midiKey];
          } else {
            // Fallback to frozen static value
            inputPortValues[port.name] = getAnyValueFromNode(
              srcNode,
              edge.sourceHandle
            );
          }
        }

        // Evaluate the inner graph
        const outputValues = evaluateInnerGraph(
          innerNodes,
          innerEdges,
          sgInputs,
          sgOutputs,
          inputPortValues
        );

        // Write the relevant output port values to the boundary
        for (const pm of portMappings) {
          const parsed = parseHandleId(pm.sourceHandle);
          if (!parsed) continue;
          const val = outputValues[parsed.name];
          if (val !== undefined && val !== null) {
            writeValue(pm.portName, val);
          }
        }
      }

      handle = requestAnimationFrame(evaluate);
    };

    handle = requestAnimationFrame(evaluate);
    return () => {
      running = false;
      cancelAnimationFrame(handle);
    };
  }, [depth, buildMappings, writeValue, stackRef]);

  // ── Static snapshot for all other producer types ─────────────────────────
  useEffect(() => {
    if (depth === 0) return;
    const mappings = buildMappings();

    for (const mapping of mappings) {
      const t = mapping.sourceNode.data.nodeType;
      // Skip types handled by live bridges above
      if (t === "midi") continue;
      if (t === "control" && mapping.sourceNode.data.isPlaying) continue;
      if (t === "timeline" && mapping.sourceNode.data.isPlaying) continue;
      if (t === "subgraph") continue; // handled by subgraph eval rAF

      const val = getAnyValueFromNode(mapping.sourceNode, mapping.sourceHandle);
      if (val !== null && val !== undefined) {
        writeValue(mapping.portName, val);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depth]);

  // Cleanup rAF on unmount
  useEffect(() => {
    return () => {
      if (rafHandle.current !== null) cancelAnimationFrame(rafHandle.current);
    };
  }, []);
}
