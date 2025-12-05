import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * Error Boundary component to catch React errors
 * Prevents the entire app from crashing when a component throws an error
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error to console (in production, this could be sent to an error reporting service)
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    this.setState({
      error,
      errorInfo,
    });
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Render custom fallback UI or use provided fallback
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          background: 'hsl(0, 0%, 6%)',
          color: 'hsl(0, 0%, 90%)',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          padding: '20px',
          textAlign: 'center',
        }}>
          <h1 style={{
            fontSize: '24px',
            fontWeight: 600,
            marginBottom: '16px',
            color: 'hsl(0, 0%, 95%)',
          }}>
            Something went wrong
          </h1>
          <p style={{
            fontSize: '14px',
            color: 'hsl(0, 0%, 70%)',
            marginBottom: '24px',
            maxWidth: '500px',
          }}>
            An unexpected error occurred. Please try reloading the application.
          </p>
          {process.env.NODE_ENV === 'development' && this.state.error && (
            <details style={{
              marginBottom: '24px',
              textAlign: 'left',
              maxWidth: '600px',
              width: '100%',
            }}>
              <summary style={{
                cursor: 'pointer',
                marginBottom: '8px',
                color: 'hsl(0, 0%, 80%)',
              }}>
                Error Details (Development Only)
              </summary>
              <pre style={{
                background: 'hsl(0, 0%, 10%)',
                padding: '12px',
                borderRadius: '4px',
                overflow: 'auto',
                fontSize: '12px',
                color: 'hsl(0, 70%, 70%)',
                border: '1px solid hsl(0, 0%, 20%)',
              }}>
                {this.state.error.toString()}
                {this.state.errorInfo && (
                  <>
                    {'\n\n'}
                    {this.state.errorInfo.componentStack}
                  </>
                )}
              </pre>
            </details>
          )}
          <button
            onClick={this.handleReset}
            style={{
              background: 'hsl(210, 70%, 50%)',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '4px',
              fontSize: '14px',
              cursor: 'pointer',
              fontWeight: 500,
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'hsl(210, 70%, 45%)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'hsl(210, 70%, 50%)';
            }}
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
