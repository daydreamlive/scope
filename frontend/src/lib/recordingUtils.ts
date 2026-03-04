/**
 * Negotiate the best available video MIME type for MediaRecorder.
 * Prefers mp4 (rare), then vp9, h264, vp8, falling back to plain webm.
 */
export function negotiateMimeType(): {
  mimeType: string;
  fileExtension: "mp4" | "webm";
} {
  if (typeof MediaRecorder === "undefined") {
    return { mimeType: "video/webm", fileExtension: "webm" };
  }

  if (MediaRecorder.isTypeSupported("video/mp4")) {
    return { mimeType: "video/mp4", fileExtension: "mp4" };
  }
  if (MediaRecorder.isTypeSupported("video/webm;codecs=vp9")) {
    return { mimeType: "video/webm;codecs=vp9", fileExtension: "webm" };
  }
  if (MediaRecorder.isTypeSupported("video/webm;codecs=h264")) {
    return { mimeType: "video/webm;codecs=h264", fileExtension: "webm" };
  }
  if (MediaRecorder.isTypeSupported("video/webm;codecs=vp8")) {
    return { mimeType: "video/webm;codecs=vp8", fileExtension: "webm" };
  }

  return { mimeType: "video/webm", fileExtension: "webm" };
}

/**
 * Trigger a browser download for an in-memory Blob.
 * Creates a temporary anchor element, clicks it, and cleans up.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Get the intrinsic dimensions of a video or canvas element.
 * Falls back to 640x480 when dimensions are unavailable.
 */
export function getSourceDimensions(el: HTMLVideoElement | HTMLCanvasElement): {
  width: number;
  height: number;
} {
  if (el instanceof HTMLVideoElement) {
    return {
      width: el.videoWidth || 640,
      height: el.videoHeight || 480,
    };
  }
  return {
    width: el.width || 640,
    height: el.height || 480,
  };
}

/**
 * Build a timestamped filename for a recording download.
 */
export function buildFilename(prefix: string, extension: string): string {
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  return `${prefix}-${ts}.${extension}`;
}
