import { createContext, useContext, type ReactNode } from "react";
import { usePlugins, type UsePluginsReturn } from "@/hooks/usePlugins";

const PluginsContext = createContext<UsePluginsReturn | null>(null);

export function PluginsProvider({ children }: { children: ReactNode }) {
  const pluginsState = usePlugins();
  return (
    <PluginsContext.Provider value={pluginsState}>
      {children}
    </PluginsContext.Provider>
  );
}

export function usePluginsContext() {
  const context = useContext(PluginsContext);
  if (!context) {
    throw new Error("usePluginsContext must be used within PluginsProvider");
  }
  return context;
}
