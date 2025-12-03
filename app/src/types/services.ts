export interface AppState {
  mainWindow: Electron.BrowserWindow | null;
  isSettingUp: boolean;
  needsSetup: boolean;
  currentSetupStatus: string;
  serverProcess: any | null;
  isServerRunning: boolean;
}

export interface SetupService {
  isSetupNeeded(): boolean;
  checkUvInstalled(): Promise<boolean>;
  downloadAndInstallUv(): Promise<void>;
  runUvSync(): Promise<void>;
}

export interface PythonProcessService {
  startServer(): Promise<void>;
  stopServer(): void;
  isServerRunning(): boolean;
}
