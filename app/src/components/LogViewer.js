// LogViewer script - externalized for CSP compliance
(function() {
  'use strict';

  // Get log data from query params
  const params = new URLSearchParams(window.location.search);
  const logPath = decodeURIComponent(params.get('path') || 'Unknown');
  const logContent = decodeURIComponent(params.get('content') || '');

  // Display log path safely using textContent
  const logPathElement = document.getElementById('logPath');
  if (logPathElement) {
    logPathElement.textContent = 'Log file: ' + logPath;
  }

  // Display logs with syntax highlighting using safer DOM manipulation
  const pre = document.getElementById('logs');
  if (!pre) return;

  // Process log content safely using textContent and DOM manipulation
  // Split into lines and process each line
  const lines = logContent.split('\n');
  const fragment = document.createDocumentFragment();

  lines.forEach((line) => {
    const lineDiv = document.createElement('div');

    // Check for log level markers and apply highlighting safely
    if (line.includes('[INFO]')) {
      const parts = line.split('[INFO]');
      if (parts[0]) lineDiv.appendChild(document.createTextNode(parts[0]));
      const infoSpan = document.createElement('span');
      infoSpan.className = 'info';
      infoSpan.textContent = '[INFO]';
      lineDiv.appendChild(infoSpan);
      if (parts[1]) lineDiv.appendChild(document.createTextNode(parts[1]));
    } else if (line.includes('[ERROR]')) {
      const parts = line.split('[ERROR]');
      if (parts[0]) lineDiv.appendChild(document.createTextNode(parts[0]));
      const errorSpan = document.createElement('span');
      errorSpan.className = 'error';
      errorSpan.textContent = '[ERROR]';
      lineDiv.appendChild(errorSpan);
      if (parts[1]) lineDiv.appendChild(document.createTextNode(parts[1]));
    } else if (line.includes('[WARN]')) {
      const parts = line.split('[WARN]');
      if (parts[0]) lineDiv.appendChild(document.createTextNode(parts[0]));
      const warnSpan = document.createElement('span');
      warnSpan.className = 'warn';
      warnSpan.textContent = '[WARN]';
      lineDiv.appendChild(warnSpan);
      if (parts[1]) lineDiv.appendChild(document.createTextNode(parts[1]));
    } else if (line.includes('[SERVER]')) {
      const parts = line.split('[SERVER]');
      if (parts[0]) lineDiv.appendChild(document.createTextNode(parts[0]));
      const serverSpan = document.createElement('span');
      serverSpan.className = 'server';
      serverSpan.textContent = '[SERVER]';
      lineDiv.appendChild(serverSpan);
      if (parts[1]) lineDiv.appendChild(document.createTextNode(parts[1]));
    } else {
      // No special markers, just add as text (safely escaped via textContent)
      lineDiv.textContent = line;
    }

    fragment.appendChild(lineDiv);
  });

  // Clear and append fragment (textContent automatically escapes HTML)
  pre.textContent = '';
  pre.appendChild(fragment);

  // Setup refresh button
  const refreshButton = document.getElementById('refreshButton');
  if (refreshButton) {
    refreshButton.addEventListener('click', () => {
      location.reload();
    });
  }

  // Scroll to bottom
  window.scrollTo(0, document.body.scrollHeight);
})();
