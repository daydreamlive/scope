# Building and Distribution Guide

This guide explains how to build, package, and distribute Daydream Scope using electron-builder.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Development](#development)
- [Building](#building)
- [Code Signing](#code-signing)
- [Auto-Updates](#auto-updates)
- [Publishing Releases](#publishing-releases)
- [Platform-Specific Notes](#platform-specific-notes)

## Prerequisites

1. **Node.js** (v18 or later)
2. **npm** or **yarn**
3. Install dependencies:
   ```bash
   cd app
   npm install
   ```

## Development

Start the development server:

```bash
npm start
# or
npm run dev
```

This will:
- Start Vite dev server for hot reloading
- Launch Electron with the app
- Watch for changes in main, preload, and renderer processes

## Building

### Build for Current Platform

```bash
npm run dist
```

This compiles the TypeScript and creates a distributable package for your current platform.

### Build for Specific Platforms

```bash
# macOS (DMG and ZIP)
npm run dist:mac

# Windows (NSIS installer)
npm run dist:win

# Linux (AppImage, deb, and rpm)
npm run dist:linux

# All platforms
npm run dist:all
```

**Note:** Cross-platform builds have limitations:
- macOS can only be built on macOS (due to code signing requirements)
- Windows can be built on Windows or Linux
- Linux can be built on any platform

### Build Output

Distributables are created in `app/dist/`:
- **macOS:** `.dmg` and `.zip` files
- **Windows:** `*-Setup.exe` (NSIS installer)
- **Linux:** `.AppImage`, `.deb`, and `.rpm` packages

## Code Signing

Code signing is crucial for:
- **macOS:** Required for auto-updates and Gatekeeper
- **Windows:** Recommended to avoid SmartScreen warnings
- **Linux:** Optional

### macOS Code Signing

1. **Get a Developer ID Certificate:**
   - Enroll in the Apple Developer Program ($99/year)
   - Create a Developer ID Application certificate in Xcode or Apple Developer portal

2. **Set environment variables:**
   ```bash
   export CSC_LINK=/path/to/certificate.p12
   export CSC_KEY_PASSWORD=your-certificate-password
   export APPLE_ID=your@apple-id.com
   export APPLE_APP_SPECIFIC_PASSWORD=app-specific-password
   ```

3. **Notarization:**
   electron-builder will automatically notarize your app if the credentials are set.

### Windows Code Signing

1. **Get a Code Signing Certificate:**
   - Purchase from providers like DigiCert, Sectigo, etc.
   - Or use a self-signed certificate for testing (not recommended for production)

2. **Set environment variables:**
   ```bash
   export CSC_LINK=/path/to/certificate.pfx
   export CSC_KEY_PASSWORD=your-certificate-password
   ```

### Disabling Code Signing (for testing)

Set in your environment:
```bash
export CSC_IDENTITY_AUTO_DISCOVERY=false
```

Or build with:
```bash
npm run dist -- --publish never
```

## Auto-Updates

The app includes electron-updater for automatic updates.

### How It Works

1. **Check for updates:** The app checks for updates on startup and every 4 hours
2. **User notification:** Users are prompted to download when an update is available
3. **Background download:** Updates download in the background
4. **Install on quit:** Updates are installed when the user quits the app

### Configuration

Update settings are in `electron-builder.yml`:

```yaml
publish:
  - provider: github
    owner: daydream-mx
    repo: scope
    private: false
```

### Update Channels

You can use different channels for releases:
- `latest` (default): Stable releases
- `beta`: Beta releases
- `alpha`: Alpha releases

Users can switch channels by modifying the update channel in the app settings.

## Publishing Releases

### GitHub Releases (Recommended)

1. **Create a GitHub Personal Access Token:**
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Create a token with `repo` scope
   - Save the token securely

2. **Set the token:**
   ```bash
   export GH_TOKEN=your-github-token
   ```

3. **Create a git tag:**
   ```bash
   git tag v0.1.0-alpha.8
   git push origin v0.1.0-alpha.8
   ```

4. **Publish the release:**
   ```bash
   npm run publish:github
   ```

   This will:
   - Build the app for all platforms
   - Upload artifacts to GitHub Releases
   - Create release files for auto-update

### Manual Publishing

If you prefer manual publishing:

1. Build for all platforms:
   ```bash
   npm run dist:all
   ```

2. Create a GitHub Release manually
3. Upload the following files from `app/dist/`:
   - All `.dmg`, `.zip`, `.exe`, `.AppImage`, `.deb`, `.rpm` files
   - All `.blockmap` files (required for delta updates)
   - `latest-mac.yml` and `latest.yml` (update manifests)

### Release Versioning

Follow semantic versioning:
- `v1.0.0` - Stable release
- `v1.0.0-beta.1` - Beta release
- `v1.0.0-alpha.1` - Alpha release

Update version in `package.json` before building.

## Platform-Specific Notes

### macOS

**DMG Configuration:**
- Background image and window size are configured in `electron-builder.yml`
- Universal builds for both Intel (x64) and Apple Silicon (arm64)

**Entitlements:**
- Required for hardened runtime
- Configured in `build/entitlements.mac.plist`

**Gatekeeper:**
- Code signing and notarization are required for distribution
- Users will see a warning for unsigned apps

### Windows

**NSIS Installer (Best Practice):**
- Per-user installation (no admin rights required)
- Configurable installation directory
- Desktop and Start Menu shortcuts
- Automatic uninstaller

**SmartScreen:**
- Unsigned apps will show a warning
- Code signing eliminates this warning
- Build reputation over time with Microsoft

**Architecture:**
- x64 (64-bit) - Most common
- ia32 (32-bit) - For older systems

### Linux

**Supported Formats:**
- **AppImage:** Universal, runs on most distributions
- **deb:** Debian, Ubuntu, and derivatives
- **rpm:** Fedora, RHEL, CentOS, and derivatives

**Dependencies:**
- Listed in `electron-builder.yml` for deb and rpm packages
- AppImage bundles all dependencies

**Installation:**
```bash
# AppImage (no installation required)
chmod +x Daydream-Scope-*.AppImage
./Daydream-Scope-*.AppImage

# Debian/Ubuntu
sudo dpkg -i daydream-scope_*.deb
sudo apt-get install -f  # Install dependencies if needed

# Fedora/RHEL
sudo rpm -i daydream-scope-*.rpm
```

## Troubleshooting

### Build Fails

1. **Check Node.js version:** Ensure you're using Node 18+
2. **Clear cache:**
   ```bash
   rm -rf node_modules .vite dist
   npm install
   ```
3. **Check logs:** electron-builder logs are in `~/.electron-builder/`

### Code Signing Issues

1. **Verify certificate:**
   ```bash
   security find-identity -v -p codesigning  # macOS
   ```
2. **Check environment variables:** Ensure `CSC_LINK` and `CSC_KEY_PASSWORD` are set
3. **Test without signing:** Set `CSC_IDENTITY_AUTO_DISCOVERY=false`

### Auto-Update Not Working

1. **Check GitHub release:** Ensure `latest-mac.yml`/`latest.yml` are uploaded
2. **Verify app is signed:** Unsigned apps can't auto-update on macOS
3. **Check logs:** Auto-updater logs are in the app's log directory
4. **Test in development:**
   ```bash
   export ELECTRON_IS_DEV=0
   npm start
   ```

## Additional Resources

- [electron-builder Documentation](https://www.electron.build/)
- [electron-updater Documentation](https://www.electron.build/auto-update)
- [Code Signing Guide](https://www.electron.build/code-signing)
- [Publishing Guide](https://www.electron.build/configuration/publish)

## CI/CD Integration

Consider using GitHub Actions for automated builds:

```yaml
name: Build and Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [macos-latest, windows-latest, ubuntu-latest]

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18

      - name: Install dependencies
        run: |
          cd app
          npm install

      - name: Build and publish
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          cd app
          npm run publish:github
```

Save this as `.github/workflows/build.yml` for automated releases on tag push.
