/**
 * Type definitions for Web MIDI API
 * These types are based on the Web MIDI API specification
 * https://www.w3.org/TR/webmidi/
 */

interface MIDIAccess extends EventTarget {
  inputs: MIDIInputMap;
  outputs: MIDIOutputMap;
  sysexEnabled: boolean;
  onstatechange: ((this: MIDIAccess, ev: MIDIConnectionEvent) => any) | null;
}

interface MIDIInputMap extends Map<string, MIDIInput> {}

interface MIDIOutputMap extends Map<string, MIDIOutput> {}

interface MIDIInput extends MIDIPort {
  onmidimessage: ((this: MIDIInput, ev: MIDIMessageEvent) => any) | null;
}

interface MIDIOutput extends MIDIPort {
  send(data: number[] | Uint8Array, timestamp?: number): void;
  clear(): void;
}

interface MIDIPort extends EventTarget {
  id: string;
  manufacturer: string | null;
  name: string | null;
  type: MIDIPortType;
  version: string | null;
  state: MIDIPortDeviceState;
  connection: MIDIPortConnectionState;
  onstatechange: ((this: MIDIPort, ev: MIDIConnectionEvent) => any) | null;
  open(): Promise<MIDIPort>;
  close(): Promise<MIDIPort>;
}

type MIDIPortType = "input" | "output";
type MIDIPortDeviceState = "disconnected" | "connected";
type MIDIPortConnectionState = "open" | "closed" | "pending";

interface MIDIMessageEvent extends Event {
  data: Uint8Array;
  receivedTime: number;
}

interface MIDIConnectionEvent extends Event {
  port: MIDIPort;
}

interface Navigator {
  requestMIDIAccess?(options?: { sysex: boolean }): Promise<MIDIAccess>;
}
