### Muxer

Lightweight canvas muxer for 2D compositing and smooth source crossfades.

_This muxer will be extracted into a separate library soon_

### Quick start

```tsx
import { PropsWithChildren, useEffect, useRef } from "react";
import { MuxerProvider, useMuxer } from "@/muxer";

function App({ children }: PropsWithChildren) {
  return (
    <MuxerProvider width={512} height={512} fps={30} sendFps={30}>
      <MySource />
      {children}
    </MuxerProvider>
  );
}

function MySource() {
  const { setCanvasSize, setSource } = useMuxer();
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!videoRef.current) return;
    setCanvasSize(1280, 720);
    setSource({
      kind: "video",
      element: videoRef.current,
      contentHint: "motion",
      fit: "cover",
    });
  }, [muxer]);

  return <video ref={videoRef} autoPlay muted />;
}
```

### API

- **stream**: `MediaStream`
- **setCanvasSize(w, h, dpr?)**
- **setFps(fps)**, **setSendFps(fps)**
- **setSource(source)**, **clearSource()**
