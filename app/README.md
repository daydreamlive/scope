# Daydream Scope Desktop App

Electron-based desktop application for Daydream Scope with auto-update support.

## Quick Start

### Development

```bash
cd app
npm install
npm start
```

This starts the development server with hot reloading.

### Building

```bash
# Build for current platform
npm run dist

# Build for specific platforms
npm run dist:mac      # macOS (DMG + ZIP)
npm run dist:win      # Windows (NSIS installer)
npm run dist:linux    # Linux (AppImage, deb, rpm)
npm run dist:all      # All platforms
```

Distributables are created in `app/dist/`.

### Publishing

```bash
# Set your GitHub token
export GH_TOKEN=your-github-token

# Create and push a tag
git tag v0.1.0-alpha.8
git push origin v0.1.0-alpha.8

# Publish to GitHub Releases
npm run publish:github
```

## Structure

- `src/main.ts` - Main Electron process (with auto-updater)
- `src/preload.ts` - Preload script for secure IPC
- `src/renderer.tsx` - React renderer for setup/loading screen
- `src/services/` - Services for setup, Python process management, etc.
- `src/components/` - React components
- `src/types/` - TypeScript type definitions
- `src/utils/` - Utility functions
- `build/` - Build scripts and configuration
- `electron-builder.yml` - electron-builder configuration

## How it works

1. On first run, the app checks if `uv` is installed
2. If not, it downloads and installs `uv` to the user data directory
3. It runs `uv sync` to install Python dependencies
4. It starts the Python backend server using `uv run daydream-scope`
5. Once the server is ready, it loads the frontend from `http://127.0.0.1:8000`
6. In production, checks for updates on startup and every 4 hours

## Key Features

- **Auto-Updates**: Automatic updates via GitHub Releases using electron-updater
- **Cross-Platform**: Windows, macOS (Intel + Apple Silicon), and Linux support
- **Setup Automation**: Automatically installs and configures Python dependencies
- **Process Management**: Manages Python server lifecycle
- **System Tray**: Minimize to system tray
- **IPC Communication**: Secure communication between processes

## Installation Methods

### Windows
- **NSIS Installer**: Per-user installation, no admin rights required
- Auto-creates desktop and Start Menu shortcuts

### macOS
- **DMG**: Drag-and-drop installation
- **ZIP**: Extract and run
- Universal binary for Intel and Apple Silicon

### Linux
- **AppImage**: Universal, no installation required
- **DEB**: For Debian/Ubuntu
- **RPM**: For Fedora/RHEL

## Documentation

See **[BUILD.md](BUILD.md)** for comprehensive building, code signing, and publishing documentation.

## Technologies

- Electron 32
- Vite 7
- React 19
- TypeScript 5
- electron-builder
- electron-updater
- Tailwind CSS
