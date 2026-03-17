import { createContext, useContext, type ReactNode } from "react";
import { useServerInfo, type UseServerInfoReturn } from "@/hooks/useServerInfo";

const ServerInfoContext = createContext<UseServerInfoReturn | null>(null);

export function ServerInfoProvider({ children }: { children: ReactNode }) {
  const serverInfoState = useServerInfo();
  return (
    <ServerInfoContext.Provider value={serverInfoState}>
      {children}
    </ServerInfoContext.Provider>
  );
}

export function useServerInfoContext() {
  const context = useContext(ServerInfoContext);
  if (!context) {
    throw new Error(
      "useServerInfoContext must be used within ServerInfoProvider"
    );
  }
  return context;
}
