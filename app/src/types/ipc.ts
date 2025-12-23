export const IPC_CHANNELS = {
  // Setup
  GET_SETUP_STATE: 'get-setup-state',
  GET_SETUP_STATUS: 'get-setup-status',
  SETUP_STATUS: 'setup-status',

  // Server
  GET_SERVER_STATUS: 'get-server-status',
  SERVER_STATUS: 'server-status',
  SERVER_ERROR: 'server-error',

  // Logs
  SHOW_LOGS: 'show-logs',
  GET_LOGS: 'get-logs',
  SHOW_CONTEXT_MENU: 'show-context-menu',
} as const;

export const SETUP_STATUS = {
  INITIALIZING: 'initializing',
  CHECKING_UV: 'checking-uv',
  DOWNLOADING_UV: 'downloading-uv',
  INSTALLING_UV: 'installing-uv',
  RUNNING_UV_SYNC: 'running-uv-sync',
  SETUP_DONE: 'setup-done',
  SETUP_ERROR: 'setup-error',
} as const;

export type SetupStatus = typeof SETUP_STATUS[keyof typeof SETUP_STATUS];
