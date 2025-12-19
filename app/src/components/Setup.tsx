import React, { useEffect, useState } from 'react';

interface SetupProps {
  onComplete: () => void;
}

declare global {
  interface Window {
    scope: {
      onSetupStatus: (callback: (status: string) => void) => void;
      getSetupStatus: () => Promise<{ status: string }>;
      getSetupState: () => Promise<{ needsSetup: boolean }>;
    };
  }
}

const Setup: React.FC<SetupProps> = ({ onComplete }) => {
  const [setupStatus, setSetupStatus] = useState<string>('Initializing...');
  const [progress, setProgress] = useState<number>(0);
  const [done, setDone] = useState(false);

  useEffect(() => {
    const scope = window.scope;

    // Query the current setup status on mount
    if (scope?.getSetupStatus) {
      scope.getSetupStatus().then((result: { status: string }) => {
        updateStatus(result.status);
      });
    }

    if (scope?.onSetupStatus) {
      scope.onSetupStatus((status: string) => {
        updateStatus(status);
      });
    }
  }, []);

  const updateStatus = (status: string) => {
    let displayStatus = 'Initializing...';
    let newProgress = 0;

    switch (status) {
      case 'initializing':
        displayStatus = 'Initializing...';
        newProgress = 0;
        break;
      case 'checking-uv':
        displayStatus = 'Checking for uv...';
        newProgress = 10;
        break;
      case 'downloading-uv':
        displayStatus = 'Downloading uv...';
        newProgress = 25;
        break;
      case 'installing-uv':
        displayStatus = 'Installing uv...';
        newProgress = 45;
        break;
      case 'running-uv-sync':
        displayStatus = 'Installing dependencies';
        newProgress = 60;
        break;
      case 'setup-done':
        displayStatus = 'Setup complete!';
        newProgress = 100;
        setDone(true);
        break;
      case 'setup-error':
        displayStatus = 'Setup failed. Please check the logs.';
        break;
      default:
        displayStatus = status;
    }

    setSetupStatus(displayStatus);
    setProgress(newProgress);
  };

  useEffect(() => {
    if (done) {
      // Wait a moment to show completion
      setTimeout(() => {
        onComplete();
      }, 1000);
    }
  }, [done, onComplete]);

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: 'column',
      background: 'hsl(0, 0%, 6%)',
      color: 'hsl(0, 0%, 90%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      overflow: 'hidden',
    }}>
      <div style={{
        textAlign: 'center',
        maxWidth: 600,
        padding: '0 24px',
      }}>
        <h1 style={{
          marginBottom: 48,
          fontSize: 28,
          fontWeight: 500,
          color: 'hsl(0, 0%, 90%)',
          letterSpacing: '0.01em',
        }}>
          Daydream Scope
        </h1>
        <div style={{
          width: '100%',
          height: 6,
          backgroundColor: 'hsl(0, 0%, 18%)',
          borderRadius: 3,
          overflow: 'hidden',
          marginBottom: 24,
        }}>
          <div style={{
            width: `${progress}%`,
            height: '100%',
            backgroundColor: 'hsl(0, 0%, 90%)',
            transition: 'width 0.3s ease',
          }} />
        </div>
        <div style={{
          fontSize: 16,
          fontWeight: 400,
          color: 'hsl(0, 0%, 60%)',
        }}>
          {setupStatus}
        </div>
      </div>
    </div>
  );
};

export default Setup;
