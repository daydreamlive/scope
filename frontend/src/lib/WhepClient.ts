// from https://github.com/livepeer/pipelines/blob/main/apps/streamdiffusion/src/lib/WhepClient.ts
export type GetVideoElement = () => HTMLVideoElement | null;

export class WhepClient {
  private playbackUrl: string;
  private getVideoElement: GetVideoElement;
  private pc: RTCPeerConnection | null = null;
  private remoteStream: MediaStream | null = null;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private retryCount = 0;
  private abortController: AbortController | null = null;
  private stopped = false;
  private disconnectedGraceTimeout: ReturnType<typeof setTimeout> | null = null;

  constructor(playbackUrl: string, getVideoElement: GetVideoElement) {
    this.playbackUrl = playbackUrl;
    this.getVideoElement = getVideoElement;
  }

  start(): void {
    this.stopped = false;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    void this.connect();
  }

  async connect(): Promise<void> {
    if (!this.playbackUrl || this.stopped) return;
    await this.cleanupPeerConnection();

    const iceServers: RTCIceServer[] = [
      { urls: "stun:stun.l.google.com:19302" },
      { urls: "stun:stun1.l.google.com:19302" },
      { urls: "stun:stun.cloudflare.com:3478" },
    ];

    const pc = new RTCPeerConnection({ iceServers });
    this.pc = pc;

    const remoteStream = new MediaStream();
    this.remoteStream = remoteStream;
    const videoEl = this.getVideoElement();
    if (videoEl) {
      videoEl.srcObject = remoteStream;
    }

    pc.addEventListener("track", event => {
      const [stream] = event.streams;
      if (stream) {
        this.remoteStream = stream;
        const el = this.getVideoElement();
        if (el) {
          el.srcObject = stream;
          void el.play().catch(() => {});
        }
      } else if (this.remoteStream) {
        this.remoteStream.addTrack(event.track);
      }
    });

    pc.addEventListener("iceconnectionstatechange", () => {
      if (this.pc !== pc) return;
      const state = pc.iceConnectionState;
      if (state === "connected" || state === "completed") {
        if (this.disconnectedGraceTimeout) {
          clearTimeout(this.disconnectedGraceTimeout);
          this.disconnectedGraceTimeout = null;
        }
        return;
      }
      if (state === "disconnected") {
        if (this.disconnectedGraceTimeout) {
          clearTimeout(this.disconnectedGraceTimeout);
          this.disconnectedGraceTimeout = null;
        }
        try {
          pc.restartIce();
        } catch {}
        this.disconnectedGraceTimeout = setTimeout(() => {
          if (this.stopped || this.pc !== pc) return;
          const current = pc.iceConnectionState;
          if (current === "disconnected") {
            this.scheduleReconnect();
          }
        }, 2000);
        return;
      }
      if (state === "failed" || state === "closed") {
        if (this.disconnectedGraceTimeout) {
          clearTimeout(this.disconnectedGraceTimeout);
          this.disconnectedGraceTimeout = null;
        }
        this.scheduleReconnect();
      }
    });

    try {
      pc.addTransceiver("video", { direction: "recvonly" });
      pc.addTransceiver("audio", { direction: "recvonly" });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const localDesc = await this.waitForIceGatheringComplete(pc, 2000);
      if (!localDesc.sdp) throw new Error("No local SDP generated");

      const controller = new AbortController();
      this.abortController = controller;

      const response = await fetch(this.playbackUrl, {
        method: "POST",
        headers: { "Content-Type": "application/sdp" },
        body: localDesc.sdp,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => "");
        throw new Error(
          `WHEP request failed: ${response.status} ${response.statusText} ${errorText}`,
        );
      }

      const answerSdp = await response.text();
      await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });

      this.retryCount = 0;
    } catch {
      this.scheduleReconnect();
    }
  }

  async stop(): Promise<void> {
    this.stopped = true;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    await this.cleanupPeerConnection();
  }

  private scheduleReconnect(): void {
    if (this.stopped) return;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    const attempt = this.retryCount;
    const delay = this.calculateDelay(attempt);
    this.reconnectTimeout = setTimeout(() => {
      if (this.stopped) return;
      this.retryCount = attempt + 1;
      void this.connect();
    }, delay);
  }

  private async cleanupPeerConnection(): Promise<void> {
    if (this.disconnectedGraceTimeout) {
      clearTimeout(this.disconnectedGraceTimeout);
      this.disconnectedGraceTimeout = null;
    }
    if (this.abortController) {
      try {
        this.abortController.abort();
      } catch {}
      this.abortController = null;
    }
    if (this.pc) {
      const currentPc = this.pc;
      this.pc = null;
      try {
        currentPc.getTransceivers().forEach(t => {
          try {
            t.stop();
          } catch {}
        });
      } catch {}
      try {
        currentPc.close();
      } catch {}
    }
    const el = this.getVideoElement();
    if (el) {
      try {
        el.srcObject = null;
      } catch {}
    }
    this.remoteStream = null;
  }

  private calculateDelay(count: number): number {
    const initialDelay = 500;
    const linearPhaseDelay = 300;
    const linearPhaseEndCount = 10;
    const baseExponentialDelay = 500;
    const maxExponentialDelay = 60 * 1000;
    const exponentFactor = 2;
    if (count === 0) return initialDelay;
    if (count > 0 && count <= linearPhaseEndCount) return linearPhaseDelay;
    const exponentialAttemptNumber = count - linearPhaseEndCount;
    const delay =
      baseExponentialDelay *
      Math.pow(exponentFactor, exponentialAttemptNumber - 1);
    return Math.min(delay, maxExponentialDelay);
  }

  private waitForIceGatheringComplete(
    pc: RTCPeerConnection,
    timeoutMs: number,
  ): Promise<RTCSessionDescriptionInit> {
    return new Promise((resolve, reject) => {
      if (pc.iceGatheringState === "complete" && pc.localDescription) {
        resolve(pc.localDescription);
        return;
      }
      const onChange = () => {
        if (pc.iceGatheringState === "complete" && pc.localDescription) {
          cleanup();
          resolve(pc.localDescription);
        }
      };
      const onTimeout = () => {
        cleanup();
        if (pc.localDescription) {
          resolve(pc.localDescription);
        } else {
          reject(new Error("ICE gathering timeout without localDescription"));
        }
      };
      const cleanup = () => {
        pc.removeEventListener("icegatheringstatechange", onChange);
        clearTimeout(timerId);
      };
      pc.addEventListener("icegatheringstatechange", onChange);
      const timerId = setTimeout(onTimeout, timeoutMs);
    });
  }
}
