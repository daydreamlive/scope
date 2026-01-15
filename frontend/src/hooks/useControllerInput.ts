import { useEffect, useRef, useCallback, useState } from "react";

/**
 * Controller input state matching the backend CtrlInput format.
 * Uses W3C event.code strings for key identification.
 */
export interface ControllerInputState {
  /** Set of currently pressed keys (W3C event.code values) */
  button: string[];
  /** Mouse velocity/delta as [dx, dy] tuple */
  mouse: [number, number];
}

/**
 * Configuration for the controller input hook.
 */
export interface ControllerInputConfig {
  /** Target send rate in Hz (default: 60) */
  sendRateHz?: number;
  /** Mouse sensitivity multiplier (default: 0.002) */
  mouseSensitivity?: number;
  /** Keys to capture (default: WASD, arrows, space, shift) */
  capturedKeys?: Set<string>;
}

/** Default keys to capture */
const DEFAULT_CAPTURED_KEYS = new Set([
  "KeyW",
  "KeyA",
  "KeyS",
  "KeyD",
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "Space",
  "ShiftLeft",
  "ShiftRight",
  "KeyQ",
  "KeyE",
  "KeyR",
  "KeyF",
  "KeyC",
  "KeyX",
  "KeyZ",
]);

/** Map browser MouseEvent.button values to descriptive names */
const MOUSE_BUTTON_NAMES: Record<number, string> = {
  0: "MouseLeft",
  1: "MouseMiddle",
  2: "MouseRight",
  3: "MouseBack",
  4: "MouseForward",
};

/**
 * Hook for capturing WASD keyboard and mouse input for streaming to backend.
 *
 * Uses a pygame-inspired state dictionary pattern:
 * - Tracks which keys are currently held down (not just press events)
 * - Accumulates mouse deltas between send intervals
 * - Sends state snapshots at a fixed rate (default 60Hz)
 *
 * @param sendFn Function to send controller input to backend
 * @param enabled Whether controller input capture is enabled
 * @param targetRef Ref to the element that should capture input (for pointer lock)
 * @param config Optional configuration
 */
export function useControllerInput(
  sendFn: (params: { ctrl_input: ControllerInputState }) => void,
  enabled: boolean,
  targetRef: React.RefObject<HTMLElement | null>,
  config?: ControllerInputConfig
) {
  const {
    sendRateHz = 60,
    mouseSensitivity = 1.5,
    capturedKeys = DEFAULT_CAPTURED_KEYS,
  } = config || {};

  // State for UI feedback
  const [isPointerLocked, setIsPointerLocked] = useState(false);
  const [pressedKeys, setPressedKeys] = useState<Set<string>>(new Set());

  // Refs for tracking input state (mutable for performance)
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const pressedMouseButtonsRef = useRef<Set<string>>(new Set());
  const mouseDeltaRef = useRef<[number, number]>([0, 0]);
  const lastSentStateRef = useRef<string>("");
  const sendIntervalRef = useRef<number | null>(null);

  // Handle keyboard events
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!enabled || !isPointerLocked) return;

      // Ignore if typing in an input field
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable
      ) {
        return;
      }

      if (capturedKeys.has(e.code)) {
        e.preventDefault();
        pressedKeysRef.current.add(e.code);
        setPressedKeys(new Set(pressedKeysRef.current));
      }
    },
    [enabled, isPointerLocked, capturedKeys]
  );

  const handleKeyUp = useCallback(
    (e: KeyboardEvent) => {
      if (!enabled) return;

      if (capturedKeys.has(e.code)) {
        e.preventDefault();
        pressedKeysRef.current.delete(e.code);
        setPressedKeys(new Set(pressedKeysRef.current));
      }
    },
    [enabled, capturedKeys]
  );

  // Handle mouse movement (only when pointer is locked)
  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!enabled || !isPointerLocked) return;

      // Accumulate mouse deltas
      mouseDeltaRef.current[0] += e.movementX * mouseSensitivity;
      mouseDeltaRef.current[1] += e.movementY * mouseSensitivity;
    },
    [enabled, isPointerLocked, mouseSensitivity]
  );

  // Handle mouse button down (only when pointer is locked)
  const handleMouseDown = useCallback(
    (e: MouseEvent) => {
      if (!enabled || !isPointerLocked) return;

      const buttonName = MOUSE_BUTTON_NAMES[e.button];
      if (buttonName) {
        e.preventDefault();
        pressedMouseButtonsRef.current.add(buttonName);
        // Combine mouse buttons with keyboard keys for UI state
        setPressedKeys(
          new Set([
            ...pressedKeysRef.current,
            ...pressedMouseButtonsRef.current,
          ])
        );
      }
    },
    [enabled, isPointerLocked]
  );

  // Handle mouse button up
  const handleMouseUp = useCallback(
    (e: MouseEvent) => {
      if (!enabled) return;

      const buttonName = MOUSE_BUTTON_NAMES[e.button];
      if (buttonName) {
        e.preventDefault();
        pressedMouseButtonsRef.current.delete(buttonName);
        setPressedKeys(
          new Set([
            ...pressedKeysRef.current,
            ...pressedMouseButtonsRef.current,
          ])
        );
      }
    },
    [enabled]
  );

  // Prevent context menu when pointer is locked
  const handleContextMenu = useCallback(
    (e: MouseEvent) => {
      if (isPointerLocked) {
        e.preventDefault();
      }
    },
    [isPointerLocked]
  );

  // Handle pointer lock changes
  const handlePointerLockChange = useCallback(() => {
    const isLocked = document.pointerLockElement === targetRef.current;
    setIsPointerLocked(isLocked);

    if (!isLocked) {
      // Clear pressed keys and mouse buttons when pointer lock is released
      pressedKeysRef.current.clear();
      pressedMouseButtonsRef.current.clear();
      setPressedKeys(new Set());
      mouseDeltaRef.current = [0, 0];
    }
  }, [targetRef]);

  // Request pointer lock
  const requestPointerLock = useCallback(() => {
    if (targetRef.current && enabled) {
      targetRef.current.requestPointerLock();
    }
  }, [targetRef, enabled]);

  // Release pointer lock
  const releasePointerLock = useCallback(() => {
    if (document.pointerLockElement) {
      document.exitPointerLock();
    }
  }, []);

  // Send controller input at fixed interval
  const sendControllerInput = useCallback(() => {
    if (!enabled || !isPointerLocked) return;

    const state: ControllerInputState = {
      // Combine keyboard keys and mouse buttons into a single array
      button: [
        ...Array.from(pressedKeysRef.current),
        ...Array.from(pressedMouseButtonsRef.current),
      ],
      mouse: [...mouseDeltaRef.current] as [number, number],
    };

    // Only send if state has changed (optimization)
    const stateStr = JSON.stringify(state);
    if (stateStr !== lastSentStateRef.current) {
      sendFn({ ctrl_input: state });
      lastSentStateRef.current = stateStr;
    }

    // Reset mouse delta after sending (it's accumulated between sends)
    mouseDeltaRef.current = [0, 0];
  }, [enabled, isPointerLocked, sendFn]);

  // Set up event listeners
  useEffect(() => {
    if (!enabled) return;

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("contextmenu", handleContextMenu);
    document.addEventListener("pointerlockchange", handlePointerLockChange);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("contextmenu", handleContextMenu);
      document.removeEventListener(
        "pointerlockchange",
        handlePointerLockChange
      );
    };
  }, [
    enabled,
    handleKeyDown,
    handleKeyUp,
    handleMouseMove,
    handleMouseDown,
    handleMouseUp,
    handleContextMenu,
    handlePointerLockChange,
  ]);

  // Set up send interval
  useEffect(() => {
    if (!enabled || !isPointerLocked) {
      if (sendIntervalRef.current) {
        clearInterval(sendIntervalRef.current);
        sendIntervalRef.current = null;
      }
      return;
    }

    const intervalMs = 1000 / sendRateHz;
    sendIntervalRef.current = window.setInterval(
      sendControllerInput,
      intervalMs
    );

    return () => {
      if (sendIntervalRef.current) {
        clearInterval(sendIntervalRef.current);
        sendIntervalRef.current = null;
      }
    };
  }, [enabled, isPointerLocked, sendRateHz, sendControllerInput]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (sendIntervalRef.current) {
        clearInterval(sendIntervalRef.current);
      }
      if (document.pointerLockElement) {
        document.exitPointerLock();
      }
    };
  }, []);

  return {
    /** Whether pointer lock is currently active */
    isPointerLocked,
    /** Set of currently pressed keys (for UI display) */
    pressedKeys,
    /** Request pointer lock on the target element */
    requestPointerLock,
    /** Release pointer lock */
    releasePointerLock,
  };
}
