// from https://github.com/livepeer/pipelines/blob/main/apps/streamdiffusion/src/lib/WhipClient.ts
export type GetStream = () => MediaStream | null;

type AccessControlParams = {
  accessKey?: string;
  jwt?: string;
};

type WhipClientOptions = {
  whipUrl: string;
  getStream: GetStream;
  onConnectionStateChange?: (state: RTCPeerConnectionState) => void;
  onRetryLimitExceeded?: () => void;
  onPlaybackUrl?: (url: string) => void;
  iceServers?: RTCIceServer[];
  sdpTimeout?: number;
  enableRetry?: boolean;
  accessKey?: string;
  jwt?: string;
  debugStats?: boolean;
  initialMaxFramerate?: number;
  maxRetries?: number;
  retryDelayBaseMs?: number;
};

const VIDEO_BITRATE = 300000;
const AUDIO_BITRATE = 64000;

const MAX_REDIRECT_CACHE_SIZE = 10;
const redirectUrlCache = new Map<string, URL>();
const playbackIdPattern = /([/+])([^/+?]+)$/;
const REPLACE_PLACEHOLDER = "PLAYBACK_ID";

function getCachedTemplate(key: string): URL | undefined {
  const cachedItem = redirectUrlCache.get(key);
  if (cachedItem) {
    redirectUrlCache.delete(key);
    redirectUrlCache.set(key, cachedItem);
  }
  return cachedItem;
}

function setCachedTemplate(key: string, value: URL): void {
  if (redirectUrlCache.has(key)) {
    redirectUrlCache.delete(key);
  } else if (redirectUrlCache.size >= MAX_REDIRECT_CACHE_SIZE) {
    const oldestKey = redirectUrlCache.keys().next().value;
    if (oldestKey) redirectUrlCache.delete(oldestKey);
  }
  redirectUrlCache.set(key, value);
}

function preferCodec(sdp: string, codec: string): string {
  const lines = sdp.split("\r\n");
  const mLineIndex = lines.findIndex(line => line.startsWith("m=video"));
  if (mLineIndex === -1) return sdp;
  const codecRegex = new RegExp(`a=rtpmap:(\\d+) ${codec}(/\\d+)+`);
  const codecLine = lines.find(line => codecRegex.test(line));
  if (!codecLine) return sdp;
  const codecPayload = codecRegex.exec(codecLine)?.[1];
  if (!codecPayload) return sdp;
  const mLineElements = lines[mLineIndex].split(" ");
  const reorderedMLine = [
    ...mLineElements.slice(0, 3),
    codecPayload,
    ...mLineElements.slice(3).filter(payload => payload !== codecPayload),
  ];
  lines[mLineIndex] = reorderedMLine.join(" ");
  return lines.join("\r\n");
}

function getRTCPeerConnectionConstructor(): typeof RTCPeerConnection | null {
  if (typeof window === "undefined") return null;
  return (
    window.RTCPeerConnection ||
    (window as any).webkitRTCPeerConnection ||
    (window as any).mozRTCPeerConnection ||
    null
  );
}

async function waitToCompleteICEGathering(
  peerConnection: RTCPeerConnection,
  timeoutMs: number,
): Promise<RTCSessionDescription | null> {
  return new Promise<RTCSessionDescription | null>(resolve => {
    const timeout = setTimeout(() => {
      resolve(peerConnection.localDescription);
    }, timeoutMs);
    peerConnection.onicegatheringstatechange = () => {
      if (peerConnection.iceGatheringState === "complete") {
        clearTimeout(timeout);
        resolve(peerConnection.localDescription);
      }
    };
  });
}

async function constructClientOffer(
  peerConnection: RTCPeerConnection,
  noIceGathering?: boolean,
): Promise<RTCSessionDescription | null> {
  const originalOffer = await peerConnection.createOffer({
    offerToReceiveAudio: false,
    offerToReceiveVideo: false,
  });
  const enhancedOffer = new RTCSessionDescription({
    type: originalOffer.type,
    sdp: preferCodec(originalOffer.sdp || "", "H264"),
  });
  await peerConnection.setLocalDescription(enhancedOffer);
  if (noIceGathering) return peerConnection.localDescription;
  const ofr = await waitToCompleteICEGathering(peerConnection, 1000);
  if (!ofr) throw Error("Failed to gather ICE candidates for offer");
  return ofr;
}

async function postSDPOffer(
  endpoint: string,
  data: string,
  controller: AbortController,
  accessControl: AccessControlParams,
  sdpTimeout: number,
  onPlaybackUrl?: (url: string) => void,
): Promise<Response> {
  const id = setTimeout(() => controller.abort(), sdpTimeout);
  const urlForPost = new URL(endpoint);
  const parsedMatches = urlForPost.pathname.match(playbackIdPattern);
  const currentPlaybackId = parsedMatches?.[2];
  const cachedTemplateUrl = getCachedTemplate(endpoint);
  if (cachedTemplateUrl && currentPlaybackId) {
    urlForPost.host = cachedTemplateUrl.host;
    urlForPost.pathname = cachedTemplateUrl.pathname.replace(
      REPLACE_PLACEHOLDER,
      currentPlaybackId,
    );
    urlForPost.search = cachedTemplateUrl.search;
  }
  try {
    const response = await fetch(urlForPost.toString(), {
      method: "POST",
      mode: "cors",
      headers: {
        "Content-Type": "application/sdp",
        ...(accessControl.accessKey && {
          "Livepeer-Access-Key": accessControl.accessKey,
        }),
        ...(accessControl.jwt && {
          "Livepeer-Jwt": accessControl.jwt,
        }),
      },
      body: data,
      signal: controller.signal,
    });
    clearTimeout(id);
    const playbackUrl = response.headers.get("livepeer-playback-url");
    if (playbackUrl && onPlaybackUrl) onPlaybackUrl(playbackUrl);
    if (response.url !== urlForPost.toString()) {
      const actualRedirectedUrl = new URL(response.url);
      const templateForCache = new URL(actualRedirectedUrl);
      templateForCache.pathname = templateForCache.pathname.replace(
        playbackIdPattern,
        `$1${REPLACE_PLACEHOLDER}`,
      );
      if (
        !templateForCache.searchParams.has("ingestpb") ||
        templateForCache.searchParams.get("ingestpb") !== "true"
      ) {
        setCachedTemplate(endpoint, templateForCache);
      }
    }
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

async function applyBitrateConstraints(
  peerConnection: RTCPeerConnection,
  maxFramerate?: number,
) {
  const senders = peerConnection.getSenders();
  for (const sender of senders) {
    if (!sender.track) continue;
    const params = sender.getParameters();
    const trackKind = sender.track.kind;
    if (trackKind === "video") {
      if (!params.encodings) params.encodings = [{}];
      params.degradationPreference = "maintain-resolution";
      const encoding = params.encodings[0];
      encoding.maxBitrate = VIDEO_BITRATE;
      if (typeof maxFramerate === "number" && maxFramerate > 0) {
        encoding.maxFramerate = maxFramerate;
      }
      encoding.scaleResolutionDownBy = 1.0;
      encoding.priority = "high";
      encoding.networkPriority = "high";

      try {
        await sender.setParameters(params);
      } catch {}
    } else if (trackKind === "audio") {
      if (!params.encodings) params.encodings = [{}];
      const encoding = params.encodings[0];
      encoding.maxBitrate = AUDIO_BITRATE;
      encoding.priority = "medium";
      encoding.networkPriority = "medium";
      try {
        await sender.setParameters(params);
      } catch {}
    }
  }
}

function debounceBitrate(
  fn: (pc: RTCPeerConnection, fps?: number) => Promise<void>,
  wait: number,
) {
  let t: ReturnType<typeof setTimeout> | null = null;
  let resolvers: Array<() => void> = [];
  let lastArgs: [RTCPeerConnection, number?] | null = null;

  return (pc: RTCPeerConnection, fps?: number) => {
    lastArgs = [pc, fps];
    if (t) {
      clearTimeout(t);
      resolvers.forEach(r => r());
      resolvers = [];
    }
    return new Promise<void>(resolve => {
      resolvers.push(resolve);
      t = setTimeout(() => {
        const [p, f] = lastArgs!;
        fn(p, f)
          .catch(() => {})
          .finally(() => {
            resolvers.forEach(r => r());
            resolvers = [];
            t = null;
          });
      }, wait);
    });
  };
}

const applyBitrateConstraintsDebounced = debounceBitrate(
  applyBitrateConstraints,
  50,
);

export class WhipClient {
  private whipUrl: string;
  private getStream: GetStream;
  private onConnectionStateChange?: (state: RTCPeerConnectionState) => void;
  private onRetryLimitExceeded?: () => void;
  private onPlaybackUrl?: (url: string) => void;
  private iceServers?: RTCIceServer[];
  private sdpTimeout: number;
  private enableRetry: boolean;
  private accessKey?: string;
  private jwt?: string;
  private debugStats?: boolean;
  private pc: RTCPeerConnection | null = null;
  private videoSender: RTCRtpSender | null = null;
  private audioSender: RTCRtpSender | null = null;
  private videoTransceiver: RTCRtpTransceiver | null = null;
  // private audioTransceiver: RTCRtpTransceiver | null = null;
  private abortController: AbortController | null = null;
  private isConnecting = false;
  private connectionEstablished = false;
  private retryTimeout: ReturnType<typeof setTimeout> | null = null;
  private retryCount = 0;
  private maxRetries: number;
  private retryDelayBaseMs: number;
  private stopped = false;
  private disconnectedGraceTimeout: ReturnType<typeof setTimeout> | null = null;
  private maxFramerate?: number;

  constructor(options: WhipClientOptions) {
    this.whipUrl = options.whipUrl;
    this.getStream = options.getStream;
    this.onConnectionStateChange = options.onConnectionStateChange;
    this.onRetryLimitExceeded = options.onRetryLimitExceeded;
    this.onPlaybackUrl = options.onPlaybackUrl;
    this.iceServers = options.iceServers;
    this.sdpTimeout = options.sdpTimeout ?? 10000;
    this.enableRetry = options.enableRetry ?? true;
    this.accessKey = options.accessKey;
    this.jwt = options.jwt;
    this.debugStats = options.debugStats;
    this.maxRetries = options.maxRetries ?? 2;
    this.retryDelayBaseMs = options.retryDelayBaseMs ?? 1000;
    this.maxFramerate = options.initialMaxFramerate;
  }

  async connect(): Promise<void> {
    if (this.stopped) return;
    const stream = this.getStream();
    if (!stream || !stream.active) return;
    const v = stream.getVideoTracks()[0];
    if (!v || v.readyState !== "live") return;
    if (this.connectionEstablished && this.pc) {
      const replaced = await this.replaceTracks(stream).catch(() => false);
      if (replaced) return;
    }
    if (this.isConnecting) return;
    this.isConnecting = true;
    if (this.abortController) {
      try {
        this.abortController.abort();
      } catch {}
    }
    if (this.disconnectedGraceTimeout) {
      clearTimeout(this.disconnectedGraceTimeout);
      this.disconnectedGraceTimeout = null;
    }
    this.abortController = new AbortController();
    try {
      if (this.pc) {
        try {
          this.pc.close();
        } catch {}
        this.pc = null;
        this.connectionEstablished = false;
        this.videoSender = null;
        this.audioSender = null;
        this.videoTransceiver = null;
        // this.audioTransceiver = null;
      }
      const pc = this.createPeerConnection();
      this.pc = pc;
      await this.attachStream(stream, pc);
      try {
        if (this.videoTransceiver) {
          const caps = RTCRtpSender.getCapabilities("video");
          if (caps?.codecs?.length) {
            const h264 = caps.codecs.filter(c =>
              c.mimeType.toLowerCase().includes("h264"),
            );
            if (h264.length && this.videoTransceiver.setCodecPreferences) {
              this.videoTransceiver.setCodecPreferences(h264);
            }
          }
        }
      } catch {}
      if (this.pc)
        await applyBitrateConstraintsDebounced(this.pc, this.maxFramerate);
      const offer = await constructClientOffer(pc, false);
      if (!offer?.sdp) throw new Error("Failed to create valid offer");
      const accessControl: AccessControlParams = {
        accessKey: this.accessKey,
        jwt: this.jwt,
      };
      const response = await postSDPOffer(
        this.whipUrl,
        offer.sdp,
        this.abortController,
        accessControl,
        this.sdpTimeout,
        this.onPlaybackUrl,
      );
      if (!response.ok) {
        const msg = await response.text().catch(() => "");
        throw new Error(
          `WHIP request failed: ${response.status} ${response.statusText} - ${msg}`,
        );
      }
      const answerSdp = await response.text();
      await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
      if (this.pc)
        await applyBitrateConstraintsDebounced(this.pc, this.maxFramerate);
      this.retryCount = 0;
      this.connectionEstablished = true;
    } catch (e) {
      if (this.pc) {
        try {
          this.pc.close();
        } catch {}
        this.pc = null;
      }
      this.connectionEstablished = false;
      this.scheduleRetry();
    } finally {
      this.isConnecting = false;
    }
  }

  async stop(): Promise<void> {
    this.stopped = true;
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }
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
      try {
        this.pc.getTransceivers().forEach(t => {
          try {
            t.stop();
          } catch {}
        });
      } catch {}
      try {
        this.pc.close();
      } catch {}
      this.pc = null;
    }
    this.videoSender = null;
    this.audioSender = null;
    this.videoTransceiver = null;
    // this.audioTransceiver = null;
  }

  setMaxFramerate(fps?: number) {
    this.maxFramerate = fps;
    const pc = this.pc;
    if (!pc) return;
    void applyBitrateConstraintsDebounced(pc, this.maxFramerate);
  }

  setWhipUrl(url: string) {
    this.whipUrl = url;
  }

  async setStream(_: MediaStream | null) {
    if (!this.pc || !this.connectionEstablished) return;
    const s = this.getStream();
    if (!s) return;
    await this.replaceTracks(s).catch(() => {});
  }

  private createPeerConnection(): RTCPeerConnection {
    const RTCPeerConnectionConstructor = getRTCPeerConnectionConstructor();
    if (!RTCPeerConnectionConstructor)
      throw new Error("No RTCPeerConnection constructor");
    const defaultIceServers: RTCIceServer[] = [
      { urls: "stun:stun.l.google.com:19302" },
      { urls: "stun:stun1.l.google.com:19302" },
      { urls: "stun:stun.cloudflare.com:3478" },
    ];
    const pc = new RTCPeerConnectionConstructor({
      iceServers: this.iceServers || defaultIceServers,
      iceCandidatePoolSize: 10,
    });
    try {
      const v = pc.addTransceiver("video", { direction: "sendonly" });
      const a = pc.addTransceiver("audio", { direction: "sendonly" });
      this.videoTransceiver = v;
      // this.audioTransceiver = a;
      this.videoSender = v.sender || null;
      this.audioSender = a.sender || null;
    } catch {}
    pc.onconnectionstatechange = () => {
      this.onConnectionStateChange?.(pc.connectionState);
      if (this.pc !== pc) return;
      switch (pc.connectionState) {
        case "connected":
          this.connectionEstablished = true;
          if (this.disconnectedGraceTimeout) {
            clearTimeout(this.disconnectedGraceTimeout);
            this.disconnectedGraceTimeout = null;
          }
          break;
        case "disconnected":
          this.connectionEstablished = false;
          if (this.disconnectedGraceTimeout) {
            clearTimeout(this.disconnectedGraceTimeout);
            this.disconnectedGraceTimeout = null;
          }
          try {
            pc.restartIce();
          } catch {}
          this.disconnectedGraceTimeout = setTimeout(() => {
            if (this.stopped || this.pc !== pc) return;
            if (pc.connectionState === "disconnected") {
              this.scheduleRetry();
            }
          }, 2000);
          break;
        case "failed":
          this.connectionEstablished = false;
          this.isConnecting = false;
          if (this.disconnectedGraceTimeout) {
            clearTimeout(this.disconnectedGraceTimeout);
            this.disconnectedGraceTimeout = null;
          }
          this.scheduleRetry();
          break;
        case "closed":
          this.connectionEstablished = false;
          this.isConnecting = false;
          if (this.disconnectedGraceTimeout) {
            clearTimeout(this.disconnectedGraceTimeout);
            this.disconnectedGraceTimeout = null;
          }
          break;
      }
    };
    pc.oniceconnectionstatechange = () => {
      if (
        (pc.iceConnectionState === "failed" ||
          pc.iceConnectionState === "disconnected") &&
        this.enableRetry
      ) {
        try {
          pc.restartIce();
        } catch {}
      }
      void applyBitrateConstraintsDebounced(pc, this.maxFramerate);
    };
    if (this.debugStats) this.setupDebugStats(pc);
    return pc;
  }

  private async attachStream(mediaStream: MediaStream, _: RTCPeerConnection) {
    const videoTrack = mediaStream.getVideoTracks()[0] || null;
    const audioTrack = mediaStream.getAudioTracks()[0] || null;
    if (this.videoSender) {
      const current = this.videoSender.track || null;
      const same = current && videoTrack && current.id === videoTrack.id;
      if (!same) {
        try {
          await this.videoSender.replaceTrack(videoTrack);
        } catch {}
      }
      if (videoTrack && videoTrack.contentHint == null) {
        try {
          videoTrack.contentHint = "motion";
        } catch {}
      }
    }
    if (this.audioSender) {
      if (audioTrack) {
        const current = this.audioSender.track || null;
        const same = current && current.id === audioTrack.id;
        try {
          if (!same) await this.audioSender.replaceTrack(audioTrack);
        } catch {}
      }
    }
  }

  private async replaceTracks(newStream: MediaStream) {
    if (!this.pc) return false;
    const tasks: Promise<void>[] = [];
    const videoTrack = newStream.getVideoTracks()[0] || null;
    const audioTrack = newStream.getAudioTracks()[0] || null;
    if (this.videoSender) {
      tasks.push(this.videoSender.replaceTrack(videoTrack).catch(() => {}));
    }
    if (this.audioSender) {
      if (audioTrack) {
        const current = this.audioSender.track || null;
        const same = current && current.id === audioTrack.id;
        if (!same)
          tasks.push(this.audioSender.replaceTrack(audioTrack).catch(() => {}));
      }
    }
    await Promise.allSettled(tasks);
    if (this.pc)
      await applyBitrateConstraintsDebounced(this.pc, this.maxFramerate);
    return true;
  }

  private scheduleRetry() {
    if (!this.enableRetry) return;
    if (this.retryCount >= this.maxRetries) {
      this.onRetryLimitExceeded?.();
      return;
    }
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }
    const delay = this.retryDelayBaseMs * Math.pow(2, this.retryCount);
    this.retryTimeout = setTimeout(() => {
      if (this.stopped) return;
      this.retryCount++;
      void this.connect();
    }, delay);
  }

  private setupDebugStats(pc: RTCPeerConnection) {
    let interval: ReturnType<typeof setInterval> | null = null;
    let lastVideoBytes = 0;
    let lastAudioBytes = 0;
    let lastFrames = 0;
    let videoBaseFrames = 0;
    let videoBaseFramesSet = false;
    let videoFpsEma: number | null = null;
    const fpsAlpha = 0.3;
    let audioBaseDuration = 0;
    let audioBaseDurationSet = false;
    const tick = async () => {
      try {
        const senders = pc.getSenders();
        const videoSender = senders.find(s => s.track?.kind === "video");
        const audioSender = senders.find(s => s.track?.kind === "audio");
        let videoRelTimeSec: number | undefined;
        let audioRelTimeSec: number | undefined;
        if (videoSender?.getStats) {
          const stats = await videoSender.getStats();
          stats.forEach(report => {
            if (report.type === "outbound-rtp" && report.kind === "video") {
              const bytes = report.bytesSent || 0;
              const frames = report.framesEncoded || report.framesSent || 0;
              const width = report.frameWidth;
              const height = report.frameHeight;
              const rtt = report.roundTripTime;
              const bytesDelta = bytes - lastVideoBytes;
              const framesDelta = frames - lastFrames;
              lastVideoBytes = bytes;
              lastFrames = frames;
              if (!videoBaseFramesSet && frames > 0) {
                videoBaseFrames = frames;
                videoBaseFramesSet = true;
              }
              const relFrames = frames - videoBaseFrames;
              const fpsInstant = Math.max(0, framesDelta);
              videoFpsEma =
                videoFpsEma == null
                  ? fpsInstant
                  : videoFpsEma * (1 - fpsAlpha) + fpsInstant * fpsAlpha;
              if (videoFpsEma && videoFpsEma > 0) {
                videoRelTimeSec = relFrames / videoFpsEma;
              }
              if (bytesDelta >= 0 || framesDelta >= 0) {
                const size = width && height ? `${width}x${height}` : undefined;
                const visibility =
                  typeof document !== "undefined"
                    ? document.visibilityState
                    : undefined;
                console.log("WHIP video tx", {
                  kbps: Math.round((bytesDelta * 8) / 1000),
                  framesDelta,
                  size,
                  rtt,
                  visibility,
                });
              }
            }
          });
        }
        if (audioSender?.getStats) {
          const stats = await audioSender.getStats();
          stats.forEach(report => {
            if (report.type === "outbound-rtp" && report.kind === "audio") {
              const bytes = report.bytesSent || 0;
              const bytesDelta = bytes - lastAudioBytes;
              lastAudioBytes = bytes;
              console.debug("WHIP audio tx", {
                kbps: Math.round((bytesDelta * 8) / 1000),
              });
            }
            if (report.type === "media-source" && report.kind === "audio") {
              const dur = report.totalSamplesDuration;
              if (typeof dur === "number") {
                if (!audioBaseDurationSet) {
                  audioBaseDuration = dur;
                  audioBaseDurationSet = true;
                }
                audioRelTimeSec = dur - audioBaseDuration;
              }
            }
          });
        }
        if (
          typeof videoRelTimeSec === "number" &&
          typeof audioRelTimeSec === "number"
        ) {
          const delta = audioRelTimeSec - videoRelTimeSec;
          console.log("WHIP av sync", {
            audio_time_s: Math.round(audioRelTimeSec * 1000) / 1000,
            video_time_s: Math.round(videoRelTimeSec * 1000) / 1000,
            delta_s: Math.round(delta * 1000) / 1000,
          });
        }
      } catch {}
    };
    interval = setInterval(tick, 1000);
    const stop = () => {
      if (interval) clearInterval(interval);
    };
    const origClose = pc.close.bind(pc);
    pc.close = () => {
      stop();
      origClose();
    };
  }
}
