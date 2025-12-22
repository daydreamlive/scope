import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import Setup from './components/Setup';
import ServerLoading from './components/ServerLoading';
import { ErrorBoundary } from './components/ErrorBoundary';

declare global {
  interface Window {
    scope: {
      getSetupState: () => Promise<{ needsSetup: boolean }>;
      onSetupStatus: (callback: (status: string) => void) => void;
      onServerStatus: (callback: (isRunning: boolean) => void) => void;
      onServerError: (callback: (error: string) => void) => void;
      showContextMenu: () => Promise<void>;
      getLogs: () => Promise<string>;
    };
  }
}

// Wait for window.scope to be available (preload script needs to run first)
const waitForScope = (): Promise<typeof window.scope> => {
  return new Promise((resolve) => {
    if (window.scope) {
      resolve(window.scope);
      return;
    }

    // Poll for window.scope to be available
    const checkInterval = setInterval(() => {
      if (window.scope) {
        clearInterval(checkInterval);
        resolve(window.scope);
      }
    }, 50);

    // Timeout after 5 seconds
    setTimeout(() => {
      clearInterval(checkInterval);
      // Return a mock scope object if it never becomes available
      resolve({
        getSetupState: () => Promise.resolve({ needsSetup: false }),
        getServerStatus: () => Promise.resolve({ isRunning: false }),
        onSetupStatus: () => {},
        onServerStatus: () => {},
        onServerError: () => {},
        showContextMenu: () => Promise.resolve(),
        getLogs: () => Promise.resolve(''),
      } as typeof window.scope);
    }, 5000);
  });
};

const rootEl = document.getElementById('root') || document.body;

// Add draggable title bar for macOS (always add - webkit-app-region only works on macOS)
const titleBar = document.createElement('div');
titleBar.style.cssText = `
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 32px;
  -webkit-app-region: drag;
  z-index: 10000;
  pointer-events: auto;
`;
document.body.appendChild(titleBar);

// Add right-click context menu to title bar
titleBar.addEventListener('contextmenu', async (e) => {
  e.preventDefault();
  try {
    const scope = await waitForScope();
    if (scope?.showContextMenu) {
      scope.showContextMenu();
    }
  } catch (err) {
    console.error('Error showing context menu:', err);
  }
});

const App = () => {
  const [needsSetup, setNeedsSetup] = useState<boolean | null>(null);
  const [setupComplete, setSetupComplete] = useState(false);
  const [serverRunning, setServerRunning] = useState<boolean | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;

    (async () => {
      try {
        // Wait for scope to be available
        const scope = await waitForScope();

        // Set a timeout to prevent getting stuck forever
        timeoutId = setTimeout(() => {
          console.warn('IPC calls timed out, using fallback values');
          setNeedsSetup(false);
          setServerRunning(false);
        }, 3000);

        // Check both setup state and server status in parallel
        const [setupState, serverStatus] = await Promise.all([
          scope.getSetupState ? scope.getSetupState().catch(() => ({ needsSetup: false })) : Promise.resolve({ needsSetup: false }),
          scope.getServerStatus ? scope.getServerStatus().catch(() => ({ isRunning: false })) : Promise.resolve({ isRunning: false }),
        ]);

        clearTimeout(timeoutId);
        setNeedsSetup(setupState?.needsSetup ?? false);
        setServerRunning(serverStatus?.isRunning ?? false);
      } catch (err) {
        console.error('Error checking initial state:', err);
        if (timeoutId) clearTimeout(timeoutId);
        // Fallback to default values
        setNeedsSetup(false);
        setServerRunning(false);
      }
    })();

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, []);

  useEffect(() => {
    (async () => {
      const scope = await waitForScope();
      if (scope?.onServerStatus) {
        scope.onServerStatus((isRunning: boolean) => {
          setServerRunning(isRunning);
        });
      }
      if (scope?.onSetupStatus) {
        scope.onSetupStatus((status: string) => {
          // Setup status updates are handled by Setup component
        });
      }
      if (scope?.onServerError) {
        scope.onServerError((error: string) => {
          setServerError(error);
        });
      }
    })();
  }, []);

  if (needsSetup === null || serverRunning === null) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: 'hsl(0, 0%, 6%)',
        color: 'hsl(0, 0%, 90%)',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        overflow: 'hidden',
      }}>
        <div style={{
          fontSize: 16,
          fontWeight: 400,
          color: 'hsl(0, 0%, 60%)',
        }}>
          Loading...
        </div>
      </div>
    );
  }

  // Show setup screen if needed
  if (needsSetup && !setupComplete) {
    return <Setup onComplete={() => setSetupComplete(true)} />;
  }

  // If server is already running, show a brief loading message
  // The main process will load the frontend directly
  if (serverRunning) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: 'hsl(0, 0%, 6%)',
        color: 'hsl(0, 0%, 90%)',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        overflow: 'hidden',
      }}>
        <div style={{
          fontSize: 16,
          fontWeight: 400,
          color: 'hsl(0, 0%, 60%)',
        }}>
          Loading...
        </div>
      </div>
    );
  }

  // Show server loading screen when setup is complete or not needed
  // This will be shown while the server is starting up
  if (setupComplete || !needsSetup) {
    return (
      <ServerLoading
        onComplete={() => {
          // Server is ready, main process will load the frontend
        }}
        onError={(error) => {
          setServerError(error);
        }}
      />
    );
  }

  // Fallback - should not reach here
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100vh',
      background: 'hsl(0, 0%, 6%)',
      color: 'hsl(0, 0%, 90%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      overflow: 'hidden',
    }}>
      <div style={{
        fontSize: 16,
        fontWeight: 400,
        color: 'hsl(0, 0%, 60%)',
      }}>
        Loading...
      </div>
    </div>
  );
};

createRoot(rootEl).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
