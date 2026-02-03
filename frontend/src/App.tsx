import { StreamPage } from "./pages/StreamPage";
import { Toaster } from "./components/ui/sonner";
import { CloudProvider } from "./lib/cloudContext";
import "./index.css";

// Get cloud WebSocket URL and API key from environment variables
// Set VITE_CLOUD_WS_URL to enable cloud mode, e.g.:
// VITE_CLOUD_WS_URL=wss://fal.run/your-username/scope-app/ws
// VITE_CLOUD_KEY=your-cloud-api-key
const CLOUD_WS_URL = import.meta.env.VITE_CLOUD_WS_URL as string | undefined;
const CLOUD_KEY = import.meta.env.VITE_CLOUD_KEY as string | undefined;

function App() {
  return (
    <CloudProvider wsUrl={CLOUD_WS_URL} apiKey={CLOUD_KEY}>
      <StreamPage />
      <Toaster />
    </CloudProvider>
  );
}

export default App;
