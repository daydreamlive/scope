import { useState, useEffect } from "react";
import { StreamPage } from "./pages/StreamPage";
import { Toaster } from "./components/ui/sonner";
import { PIPELINES } from "./data/pipelines";
import { getPipelineDefaults } from "./lib/api";
import { setPipelineDefaults } from "./lib/utils";
import type { PipelineId } from "./types";
import "./index.css";

function App() {
  const [defaultsLoaded, setDefaultsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAllDefaults = async () => {
      try {
        const pipelineIds = Object.keys(PIPELINES);
        await Promise.all(
          pipelineIds.map(async id => {
            const defaults = await getPipelineDefaults(id);
            setPipelineDefaults(id as PipelineId, defaults);
          })
        );
        setDefaultsLoaded(true);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load pipeline defaults"
        );
      }
    };

    fetchAllDefaults();
  }, []);

  if (error) {
    return (
      <div style={{ padding: "2rem", textAlign: "center" }}>
        <h1>Error Loading Application</h1>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  if (!defaultsLoaded) {
    return (
      <div style={{ padding: "2rem", textAlign: "center" }}>
        <p>Loading pipeline defaults...</p>
      </div>
    );
  }

  return (
    <>
      <StreamPage />
      <Toaster />
    </>
  );
}

export default App;
