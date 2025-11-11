export interface StartWhipOptions {
  endpoint: string;
  token?: string;
  localStream?: MediaStream;
  iceServers?: RTCIceServer[];
  /** Max time to wait for ICE gathering before sending (ms). Defaults to 2500ms. */
  maxIceGatherMs?: number;
}

export async function startWhip(options: StartWhipOptions) {
  const { endpoint, token, localStream, iceServers, maxIceGatherMs } = options;

  const peerConnection = new RTCPeerConnection({
    iceServers: iceServers ?? [{ urls: "stun:stun.l.google.com:19302" }],
  });

  // Attach upstream if provided; otherwise be recv-only
  if (localStream) {
    localStream.getTracks().forEach(track => {
      peerConnection.addTrack(track, localStream);
    });
  } else {
    peerConnection.addTransceiver("video");
  }

  // Collect remote stream
  const remoteStream = new MediaStream();
  peerConnection.ontrack = evt => {
    if (evt.streams && evt.streams[0]) {
      evt.streams[0].getTracks().forEach(track => remoteStream.addTrack(track));
    } else {
      remoteStream.addTrack(evt.track);
    }
  };

  // Create offer and wait for ICE complete (non-trickle)
  const offer = await peerConnection.createOffer();
  await peerConnection.setLocalDescription(offer);
  await waitForIceGatheringComplete(peerConnection, maxIceGatherMs ?? 2500);

  const sdp = peerConnection.localDescription?.sdp || offer.sdp || "";
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/sdp",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: sdp,
  });

  if (!response.ok) {
    const text = await response.text();
    peerConnection.close();
    throw new Error(
      `WHIP offer failed: ${response.status} ${response.statusText}: ${text}`
    );
  }

  const resourceUrl = response.headers.get("location") || undefined;
  const answerSdp = await response.text();

  await peerConnection.setRemoteDescription({
    type: "answer",
    sdp: answerSdp,
  });

  const stop = async () => {
    try {
      peerConnection.close();
      if (resourceUrl) {
        await fetch(resourceUrl, {
          method: "DELETE",
          headers: {
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
        });
      }
    } catch {
      // ignore
    }
  };

  return { peerConnection, remoteStream, resourceUrl, stop };
}

async function waitForIceGatheringComplete(
  pc: RTCPeerConnection,
  timeoutMs: number
): Promise<void> {
  if (pc.iceGatheringState === "complete") return;

  await Promise.race<void>([
    new Promise<void>(resolve => {
      const listener = () => {
        if (pc.iceGatheringState === "complete") {
          pc.removeEventListener("icegatheringstatechange", listener);
          resolve();
        }
      };
      pc.addEventListener("icegatheringstatechange", listener);
    }),
    new Promise<void>(resolve => setTimeout(resolve, timeoutMs)),
  ]);
}
