import { StreamPage } from "./pages/StreamPage";
import { Toaster } from "./components/ui/sonner";
import { PipelinesProvider } from "./contexts/PipelinesContext";
import "./index.css";

function App() {
  return (
    <PipelinesProvider>
      <StreamPage />
      <Toaster />
    </PipelinesProvider>
  );
}

export default App;
