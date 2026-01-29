import { StreamPage } from "./pages/StreamPage";
import { Toaster } from "./components/ui/sonner";
import { FalProvider } from "./lib/falContext";
import "./index.css";

// Get fal WebSocket URL and API key from environment variables
// Set VITE_FAL_WS_URL to enable fal mode, e.g.:
// VITE_FAL_WS_URL=wss://fal.run/your-username/scope-app/ws
// VITE_FAL_KEY=your-fal-api-key
const FAL_WS_URL = import.meta.env.VITE_FAL_WS_URL as string | undefined;
const FAL_KEY = import.meta.env.VITE_FAL_KEY as string | undefined;

function App() {
  return (
    <FalProvider wsUrl={FAL_WS_URL} apiKey={FAL_KEY}>
      <StreamPage />
      <Toaster />
    </FalProvider>
  );
}

export default App;
