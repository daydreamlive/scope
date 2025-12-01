# Daydream Scope - Electron App

This directory contains the Electron wrapper for Daydream Scope, packaging the Vite frontend and Python backend into a desktop application.

## Structure

- `main.js` - Electron main process (manages Python server, system tray, windows)
- `preload.js` - Preload script for secure IPC communication
- `install-python-deps.js` - Python dependency installer utility
- `package.json` - Electron app configuration and build settings
- `vite.config.electron.js` - Vite config for Electron builds
- `assets/` - Application icons and assets
- `build/` - Build resources (icons for packaging)

## Development

### Prerequisites

- Node.js 18+ and npm
- Python 3.10+
- uv (Python package manager)
- The frontend must be built first: `cd frontend && npm run build`

### Running in Development

1. **Build the frontend** (if not already built):
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

2. **Install Electron dependencies**:
   ```bash
   cd electron
   npm install
   ```

3. **Run Electron app**:
   ```bash
   npm run dev
   ```

   This will:
   - Start the Electron app
   - Check for Python dependencies
   - Start the Python server automatically
   - Load the frontend from the built dist folder

### Development with Vite Dev Server

For faster frontend development, you can run Vite dev server separately:

1. **Terminal 1 - Start Vite dev server**:
   ```bash
   cd frontend
   npm run dev
   ```

2. **Terminal 2 - Start Electron**:
   ```bash
   cd electron
   NODE_ENV=development npm run dev
   ```

   The Electron app will connect to `http://localhost:5173` in development mode.

## Building for Production

### Build Frontend

First, build the frontend for production:

```bash
cd frontend
npm run build
cd ..
```

### Build Electron App

```bash
cd electron
npm run build
```

This will create platform-specific installers in `electron/dist/`:
- **Windows**: `.exe` installer (NSIS)
- **Linux**: `.AppImage` and `.deb` packages
- **macOS**: `.dmg` installer

### Platform-Specific Builds

```bash
# Windows only
npm run build:win

# Linux only
npm run build:linux

# macOS only
npm run build:mac
```

## How It Works

### Architecture

1. **Main Process** (`main.js`):
   - Manages the Python server lifecycle
   - Creates and manages the application window
   - Handles system tray icon and menu
   - Provides IPC handlers for renderer communication

2. **Renderer Process** (Vite frontend):
   - Runs the React UI
   - Communicates with Python API server via HTTP (localhost:8000)
   - Uses Electron IPC for app-specific features (server status, navigation)

3. **Python Server**:
   - Runs in background via child process
   - Serves API endpoints on localhost:8000
   - Managed by Electron main process

### System Tray

The app runs in the system tray with a context menu:
- **Open Scope** - Opens/restores the main window
- **Settings** - Opens window and navigates to settings
- **Logs** - Opens the logs directory in file manager
- **Quit** - Stops Python server and quits the app

### Python Dependency Management

On first launch, the app checks for:
1. Python 3.10+ installation
2. uv package manager
3. Required Python dependencies

If any are missing, it will attempt to install them or prompt the user.

## Configuration

### Environment Variables

- `NODE_ENV=development` - Run in development mode (uses Vite dev server)
- `DAYDREAM_SCOPE_LOGS_DIR` - Override logs directory location

### Server Configuration

The Python server runs on:
- Host: `127.0.0.1` (localhost only)
- Port: `8000`

These are hardcoded in `main.js` but can be changed if needed.

## Troubleshooting

### Python Server Won't Start

1. Check that Python dependencies are installed:
   ```bash
   uv sync
   ```

2. Verify Python executable path in `main.js`

3. Check server logs in `~/.daydream-scope/logs/`

### Frontend Not Loading

1. Ensure frontend is built:
   ```bash
   cd frontend && npm run build
   ```

2. Check that `frontend/dist/index.html` exists

3. In development, ensure Vite dev server is running on port 5173

### Build Failures

1. Ensure all dependencies are installed:
   ```bash
   cd electron && npm install
   ```

2. Check that icons exist in `electron/build/`:
   - `icon.png` (512x512)
   - `icon.ico` (Windows)
   - `icon.icns` (macOS)

3. Verify electron-builder is properly configured in `package.json`

## App Icons

See `assets/README.md` for icon requirements and creation instructions.

The app currently uses placeholder icons. Replace them with final designs before release.

## Notes

- The app runs the Python server in the background, so it's always available for API access
- The system tray icon allows quick access without keeping a window open
- Logs are stored in `~/.daydream-scope/logs/` (same as CLI version)
- Models are stored in `~/.daydream-scope/models/` (same as CLI version)
