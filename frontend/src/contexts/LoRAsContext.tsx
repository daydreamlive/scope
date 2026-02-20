import { createContext, useContext, type ReactNode } from "react";
import { useLoRAFiles, type UseLoRAFilesReturn } from "@/hooks/useLoRAFiles";

const LoRAsContext = createContext<UseLoRAFilesReturn | null>(null);

export function LoRAsProvider({ children }: { children: ReactNode }) {
  const loraFilesState = useLoRAFiles();
  return (
    <LoRAsContext.Provider value={loraFilesState}>
      {children}
    </LoRAsContext.Provider>
  );
}

export function useLoRAsContext() {
  const context = useContext(LoRAsContext);
  if (!context) {
    throw new Error("useLoRAsContext must be used within LoRAsProvider");
  }
  return context;
}
