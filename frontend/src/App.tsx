import { StreamPage } from "./pages/StreamPage";
import { Toaster } from "./components/ui/sonner";
import "./index.css";
import { MuxerProvider } from "@/muxer";

function App() {
  return (
    <MuxerProvider width={512} height={512} fps={30} sendFps={30}>
      <>
        <StreamPage />
        <Toaster />
      </>
    </MuxerProvider>
  );
}

export default App;
