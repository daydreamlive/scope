/**
 * Controller Transport Abstraction
 *
 * This module provides an abstraction layer for sending controller input
 * to the backend. Currently uses WebRTC data channel, but can be swapped
 * for GPU frame sharing or other transport mechanisms in the future.
 */

import type { ControllerInputState } from "../hooks/useControllerInput";

/**
 * Interface for controller input transport.
 * Implementations can use different transport mechanisms.
 */
export interface ControllerTransport {
  /** Send controller input state to backend */
  send(input: ControllerInputState): void;
  /** Check if transport is ready to send */
  isReady(): boolean;
}

/**
 * WebRTC Data Channel transport implementation.
 * Uses the existing WebRTC data channel for sending controller input.
 */
export class DataChannelTransport implements ControllerTransport {
  sendParameterUpdate: (params: { ctrl_input: ControllerInputState }) => void;

  constructor(
    sendParameterUpdate: (params: { ctrl_input: ControllerInputState }) => void
  ) {
    this.sendParameterUpdate = sendParameterUpdate;
  }

  send(input: ControllerInputState): void {
    this.sendParameterUpdate({ ctrl_input: input });
  }

  isReady(): boolean {
    // The sendParameterUpdate function handles checking if the channel is open
    return true;
  }
}

/**
 * Create a controller transport using WebRTC data channel.
 *
 * @param sendParameterUpdate Function from useWebRTC hook
 * @returns ControllerTransport instance
 */
export function createDataChannelTransport(
  sendParameterUpdate: (params: { ctrl_input: ControllerInputState }) => void
): ControllerTransport {
  return new DataChannelTransport(sendParameterUpdate);
}

/**
 * Future: GPU Shared Memory transport
 *
 * This would be used when running locally with the Electron app
 * to bypass WebRTC and share frames directly on the GPU.
 *
 * Example implementation:
 *
 * export class SharedMemoryTransport implements ControllerTransport {
 *   constructor(private sharedBuffer: SharedArrayBuffer) {}
 *
 *   send(input: ControllerInputState): void {
 *     // Write directly to shared memory
 *   }
 *
 *   isReady(): boolean {
 *     return this.sharedBuffer !== null;
 *   }
 * }
 */
