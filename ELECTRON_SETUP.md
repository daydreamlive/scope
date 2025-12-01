# Electron App Setup - Daydream Scope

This document describes the Electron app packaging for Daydream Scope, which wraps the Vite frontend and Python backend into a desktop application.

## Overview

The Electron app provides:
- **Desktop application** - Native installers for Windows, Linux, and macOS
- **System tray integration** - App runs in background with tray icon
- **Automatic Python server management** - Starts/stops Python API server automatically
- **Dependency management** - Checks and installs Python dependencies on first launch
- **One-click installation** - Users don't need to install Git, Python, or manage dependencies manually

## Architecture

```
┌─────────────────────────────────────────┐
│         Electron Main Process           │
│  - Manages Python server lifecycle      │
│  - System tray icon & menu              │
│  - Window management                    │
│  - IPC handlers                         │
└─────────────────────────────────────────┘
           │                    │
           │                    │
    ┌──────▼──────┐      ┌──────▼──────┐
    │   Python    │      │   Electron  │
    │   Server    │      │   Renderer  │
    │ (port 8000) │◄─────┤  (Vite UI)  │
    └─────────────┘      └─────────────┘
         │                      │
         │                      │
    ┌────▼──────────────────────▼────┐
    │      HTTP API Requests         │
    │  (localhost:8000/api/v1/...)   │
    └────────────────────────────────┘
```

## File Structure

```
electron/
├── main.js                    # Main Electron process
├── preload.js                 # Preload script for secure IPC
├── install-python-deps.js     # Python dependency installer
├── package.json               # Electron app config & build settings
├── vite.config.electron.js    # Vite config for Electron builds
├── build.sh                   # Build script (frontend + Electron)
├── create-icons.sh            # Icon generation script
├── README.md                  # Electron-specific documentation
├── assets/                    # App icons (source)
│   ├── icon.png              # Main icon (512x512)
│   ├── tray-icon.png         # Tray icon (32x32)
│   └── README.md             # Icon creation guide
└── build/                     # Build resources
    ├── icon.png              # Build icon (512x512)
    ├── icon.ico              # Windows icon
    └── icon.icns             # macOS icon bundle
```

## Key Features

### 1. System Tray Integration

The app runs in the system tray with a context menu:
- **Open Scope** - Opens/restores the main window (Editor page)
- **Settings** - Opens window and focuses settings panel
- **Logs** - Opens logs directory (`~/.daydream-scope/logs/`) in file manager
- **Quit** - Stops Python server and exits app

### 2. Python Server Management

- Automatically starts Python server on app launch
- Monitors server health and restarts if crashed
- Stops server cleanly on app quit
- Runs server on `127.0.0.1:8000` (localhost only)

### 3. Dependency Management

On first launch, the app:
1. Checks for Python 3.10+ installation
2. Checks for `uv` package manager
3. Installs Python dependencies if needed
4. Prompts user if manual installation required

### 4. Development vs Production

**Development Mode** (`NODE_ENV=development`):
- Loads frontend from Vite dev server (`http://localhost:5173`)
- Uses local Python venv
- Opens DevTools automatically

**Production Mode**:
- Loads frontend from built `frontend/dist/` folder
- Uses system Python or bundled Python
- No DevTools

## Building the App

### Prerequisites

1. **Node.js 18+** and npm
2. **Python 3.10+**
3. **uv** (Python package manager)
4. **ImageMagick** (optional, for icon generation)

### Build Steps

1. **Build frontend**:
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

2. **Create icons** (if not already created):
   ```bash
   cd electron
   ./create-icons.sh
   ```

3. **Install Electron dependencies**:
   ```bash
   cd electron
   npm install
   ```

4. **Build Electron app**:
   ```bash
   npm run build
   ```

   Or use the convenience script:
   ```bash
   ./build.sh
   ```

### Build Outputs

After building, installers are created in `electron/dist/`:
- **Windows**: `Daydream Scope Setup x.x.x.exe` (NSIS installer)
- **Linux**: `daydream-scope-x.x.x.AppImage` and `daydream-scope-x.x.x.deb`
- **macOS**: `Daydream Scope-x.x.x.dmg`

## Running in Development

### Option 1: Electron with Built Frontend

```bash
cd electron
npm install
npm run dev
```

### Option 2: Electron with Vite Dev Server

**Terminal 1** - Start Vite:
```bash
cd frontend
npm run dev
```

**Terminal 2** - Start Electron:
```bash
cd electron
NODE_ENV=development npm run dev
```

## Configuration

### Environment Variables

- `NODE_ENV=development` - Enable development mode
- `DAYDREAM_SCOPE_LOGS_DIR` - Override logs directory

### Server Configuration

The Python server configuration is in `electron/main.js`:
- Host: `127.0.0.1` (localhost only for security)
- Port: `8000`

### Python Path Detection

The app tries multiple methods to find Python:
1. `uv run` (preferred, works in dev and prod)
2. `.venv/bin/python` (development venv)
3. System `python3` (fallback)

## App Icons

Icons are stored in:
- `electron/assets/` - Source icons
- `electron/build/` - Build resources (used by electron-builder)

**Required icons**:
- `icon.png` (512x512) - Main app icon
- `icon.ico` (256x256) - Windows icon
- `icon.icns` - macOS icon bundle
- `tray-icon.png` (32x32) - System tray icon

See `electron/assets/README.md` for icon creation instructions.

## Packaging Python Dependencies

**Current Approach**: The app expects Python and `uv` to be installed on the user's system. On first launch, it installs dependencies using `uv sync`.

**Future Enhancement**: Bundle Python runtime and dependencies:
- Use `pyinstaller` or similar to create standalone Python executable
- Bundle Python runtime with the Electron app
- Include pre-installed dependencies

## Troubleshooting

### Python Server Won't Start

1. Check Python installation:
   ```bash
   python3 --version  # Should be 3.10+
   ```

2. Check uv installation:
   ```bash
   uv --version
   ```

3. Install Python dependencies manually:
   ```bash
   uv sync
   ```

4. Check server logs:
   ```bash
   cat ~/.daydream-scope/logs/scope-logs-*.log
   ```

### Frontend Not Loading

1. Ensure frontend is built:
   ```bash
   cd frontend && npm run build
   ```

2. Check `frontend/dist/index.html` exists

3. In dev mode, ensure Vite is running on port 5173

### Build Failures

1. Ensure all dependencies installed:
   ```bash
   cd electron && npm install
   ```

2. Check icons exist:
   ```bash
   ls electron/build/icon.*
   ```

3. Verify electron-builder config in `package.json`

### App Crashes on Launch

1. Check Electron console for errors (DevTools)
2. Check Python server logs in `~/.daydream-scope/logs/`
3. Verify Python dependencies are installed
4. Try running Python server manually:
   ```bash
   uv run daydream-scope --no-browser
   ```

## Future Enhancements

1. **Bundle Python Runtime**: Include Python and dependencies in the app bundle
2. **Auto-updates**: Implement Electron auto-updater
3. **Crash Reporting**: Add crash reporting (Sentry, etc.)
4. **Settings UI**: Add Electron-specific settings window
5. **Update Notifications**: Notify users of new versions
6. **Offline Mode**: Handle offline scenarios gracefully

## Notes

- The app runs the Python server in the background, so the API is always available
- Logs and models use the same directories as the CLI version (`~/.daydream-scope/`)
- The system tray allows quick access without keeping a window open
- On macOS, the app stays in the dock even when windows are closed (standard macOS behavior)

## Related Documentation

- [Electron README](./electron/README.md) - Detailed Electron setup
- [Main README](./README.md) - Project overview and CLI usage
- [Server Documentation](./docs/server.md) - Python server details
