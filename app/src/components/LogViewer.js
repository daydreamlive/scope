// LogViewer script - externalized for CSP compliance
(function() {
  'use strict';

  // Log level markers and their CSS classes
  const logLevels = [
    { marker: '[ERROR]', className: 'error' },
    { marker: '[WARN]', className: 'warn' },
    { marker: '[INFO]', className: 'info' },
    { marker: '[SERVER]', className: 'server' },
  ];

  // Global function to update logs (called from main process via executeJavaScript)
  window.updateLogs = function(logPath, logContent) {
    // Display log path
    const logPathElement = document.getElementById('logPath');
    if (logPathElement) {
      logPathElement.textContent = 'Log file: ' + logPath;
    }

    // Render log lines with syntax highlighting
    const pre = document.getElementById('logs');
    if (pre) {
      const fragment = document.createDocumentFragment();
      const lines = logContent.split('\n');

      lines.forEach((line) => {
        const lineDiv = document.createElement('div');
        let foundLevel = false;

        // Check for log level markers
        for (const { marker, className } of logLevels) {
          if (line.includes(marker)) {
            const parts = line.split(marker);
            if (parts[0]) lineDiv.appendChild(document.createTextNode(parts[0]));
            const span = document.createElement('span');
            span.className = className;
            span.textContent = marker;
            lineDiv.appendChild(span);
            if (parts[1]) lineDiv.appendChild(document.createTextNode(parts[1]));
            foundLevel = true;
            break;
          }
        }

        // No log level marker found, display as plain text
        if (!foundLevel) {
          lineDiv.textContent = line;
        }

        fragment.appendChild(lineDiv);
      });

      pre.textContent = '';
      pre.appendChild(fragment);
    }

    // Scroll to bottom
    window.scrollTo(0, document.body.scrollHeight);
  };

  // Initial display message
  const pre = document.getElementById('logs');
  if (pre) {
    pre.textContent = 'Loading logs...';
  }
})();
