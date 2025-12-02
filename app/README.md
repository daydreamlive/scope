# Daydream Scope Desktop App

This is the Electron wrapper for Daydream Scope.

## Development

```bash
cd app
npm install
npm start
```

## Building

To build for the current platform:

```bash
npm run package
```

To build distributables for all platforms:

```bash
npm run make
```

## Structure

- `src/main.ts` - Main Electron process
- `src/preload.ts` - Preload script for secure IPC
- `src/renderer.tsx` - React renderer for setup/loading screen
- `src/services/` - Services for setup, Python process management, etc.
- `src/components/` - React components
- `src/types/` - TypeScript type definitions
- `src/utils/` - Utility functions

## How it works

1. On first run, the app checks if `uv` is installed
2. If not, it downloads and installs `uv` to the user data directory
3. It runs `uv sync` to install Python dependencies
4. It starts the Python backend server using `uv run daydream-scope`
5. Once the server is ready, it loads the frontend from `http://127.0.0.1:8000`
