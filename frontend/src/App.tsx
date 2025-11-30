import { useEffect } from "react";
import { StreamPage } from "./pages/StreamPage";
import { Toaster } from "./components/ui/sonner";
import { prefetchAllPipelineSchemas } from "./lib/api";
import "./index.css";

function App() {
  // Pre-fetch all pipeline schemas on app load to avoid waiting
  // for schema fetches when user switches pipelines
  useEffect(() => {
    prefetchAllPipelineSchemas();
  }, []);

  return (
    <>
      <StreamPage />
      <Toaster />
    </>
  );
}

export default App;
