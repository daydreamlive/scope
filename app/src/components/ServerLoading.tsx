import React, { useEffect, useState } from 'react';

interface ServerLoadingProps {
  onComplete: () => void;
  onError: (error: string) => void;
}

declare global {
  interface Window {
    scope: {
      onServerStatus: (callback: (isRunning: boolean) => void) => void;
      onServerError: (callback: (error: string) => void) => void;
      getServerStatus: () => Promise<{ isRunning: boolean }>;
    };
  }
}

const ServerLoading: React.FC<ServerLoadingProps> = ({ onComplete, onError }) => {
  const [serverRunning, setServerRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const scope = window.scope;

    // Query the current server status on mount
    if (scope?.getServerStatus) {
      scope.getServerStatus().then((result: { isRunning: boolean }) => {
        if (result.isRunning) {
          setServerRunning(true);
          onComplete();
        }
      });
    }

    if (scope?.onServerStatus) {
      scope.onServerStatus((isRunning: boolean) => {
        if (isRunning) {
          setServerRunning(true);
          onComplete();
        }
      });
    }

    if (scope?.onServerError) {
      scope.onServerError((errorMessage: string) => {
        setError(errorMessage);
        onError(errorMessage);
      });
    }
  }, [onComplete, onError]);

  if (error) {
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
            fontSize: 16,
            fontWeight: 500,
            color: 'hsl(0, 62%, 50%)',
            marginBottom: 24,
          }}>
            Server failed to start
          </div>
          <div style={{
            fontSize: 14,
            fontWeight: 400,
            color: 'hsl(0, 0%, 60%)',
            backgroundColor: 'hsl(0, 0%, 8%)',
            border: '1px solid hsl(0, 0%, 18%)',
            padding: 16,
            borderRadius: 8,
            textAlign: 'left',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}>
            {error}
          </div>
        </div>
      </div>
    );
  }

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
            width: '100%',
            height: '100%',
            backgroundColor: 'hsl(0, 0%, 90%)',
            animation: 'pulse 1.5s ease-in-out infinite',
          }} />
        </div>
        <div style={{
          fontSize: 16,
          fontWeight: 400,
          color: 'hsl(0, 0%, 60%)',
        }}>
          Starting server...
        </div>
      </div>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};

export default ServerLoading;
