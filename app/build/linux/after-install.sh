#!/bin/bash
# Post-installation script for Linux
# This script downloads UV and sets up the Python environment during installation

set -e

INSTALL_DIR="/opt/DaydreamScope"

# Fix chrome-sandbox permissions (required for Electron sandbox)
if [ -f "$INSTALL_DIR/chrome-sandbox" ]; then
    chmod 4755 "$INSTALL_DIR/chrome-sandbox"
    echo "Set chrome-sandbox permissions to 4755"
fi

# Update desktop database
if command -v update-desktop-database > /dev/null 2>&1; then
    update-desktop-database -q
fi

# Update icon cache
if command -v gtk-update-icon-cache > /dev/null 2>&1; then
    gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
fi

# Update MIME database
if command -v update-mime-database > /dev/null 2>&1; then
    update-mime-database /usr/share/mime || true
fi

# Detect the user who is installing (when using sudo)
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)

# If we couldn't detect the user, skip Python setup (it will happen on first launch)
if [ -z "$REAL_USER" ] || [ "$REAL_USER" = "root" ] || [ -z "$REAL_HOME" ]; then
    echo "Could not detect installing user. Python environment will be set up on first launch."
    exit 0
fi

echo "Setting up Python environment for user: $REAL_USER"

# Define paths (matching Electron's app.getPath('userData') for Linux)
USER_DATA_DIR="$REAL_HOME/.config/DaydreamScope"
UV_DIR="$USER_DATA_DIR/uv"
PROJECT_DIR="$USER_DATA_DIR/python-project"
RESOURCES_DIR="$INSTALL_DIR/resources"

# Create directories (logs directory for Electron app logs only, Python logs go to ~/.daydream-scope/logs)
su - "$REAL_USER" -c "mkdir -p '$UV_DIR' '$PROJECT_DIR' '$USER_DATA_DIR/logs'" || {
    echo "Warning: Failed to create directories. Setup will continue on first launch."
    exit 0
}

# Download UV
echo "Downloading UV package manager..."
UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
TEMP_DIR=$(mktemp -d)
UV_ARCHIVE="$TEMP_DIR/uv.tar.gz"

if command -v curl > /dev/null 2>&1; then
    curl -L -o "$UV_ARCHIVE" "$UV_URL" 2>/dev/null || {
        echo "Warning: Failed to download UV. It will be downloaded on first launch."
        rm -rf "$TEMP_DIR"
        exit 0
    }
elif command -v wget > /dev/null 2>&1; then
    wget -q -O "$UV_ARCHIVE" "$UV_URL" || {
        echo "Warning: Failed to download UV. It will be downloaded on first launch."
        rm -rf "$TEMP_DIR"
        exit 0
    }
else
    echo "Warning: curl or wget not found. UV will be downloaded on first launch."
    rm -rf "$TEMP_DIR"
    exit 0
fi

# Extract UV
echo "Extracting UV..."
tar -xzf "$UV_ARCHIVE" -C "$TEMP_DIR" 2>/dev/null || {
    echo "Warning: Failed to extract UV. It will be downloaded on first launch."
    rm -rf "$TEMP_DIR"
    exit 0
}

# Find and copy uv binary
UV_BINARY=$(find "$TEMP_DIR" -name "uv" -type f | head -n 1)
if [ -n "$UV_BINARY" ]; then
    cp "$UV_BINARY" "$UV_DIR/uv"
    chmod +x "$UV_DIR/uv"
    chown "$REAL_USER:$REAL_USER" "$UV_DIR/uv"
    echo "UV installed successfully"
else
    echo "Warning: Could not find uv binary. It will be downloaded on first launch."
    rm -rf "$TEMP_DIR"
    exit 0
fi

# Cleanup temp files
rm -rf "$TEMP_DIR"

# Copy Python project files from resources to user data directory
echo "Copying Python project files..."
if [ -d "$RESOURCES_DIR" ] && [ -f "$RESOURCES_DIR/pyproject.toml" ]; then
    su - "$REAL_USER" -c "
        cp -r '$RESOURCES_DIR/src' '$PROJECT_DIR/src' 2>/dev/null || true
        cp '$RESOURCES_DIR/pyproject.toml' '$PROJECT_DIR/pyproject.toml' 2>/dev/null || true
        cp '$RESOURCES_DIR/uv.lock' '$PROJECT_DIR/uv.lock' 2>/dev/null || true
        cp '$RESOURCES_DIR/.python-version' '$PROJECT_DIR/.python-version' 2>/dev/null || true
        cp '$RESOURCES_DIR/README.md' '$PROJECT_DIR/README.md' 2>/dev/null || true
        cp '$RESOURCES_DIR/LICENSE.md' '$PROJECT_DIR/LICENSE.md' 2>/dev/null || true
        if [ -d '$RESOURCES_DIR/frontend' ]; then
            cp -r '$RESOURCES_DIR/frontend' '$PROJECT_DIR/frontend' 2>/dev/null || true
        fi
    " || {
        echo "Warning: Failed to copy Python project files. Setup will continue on first launch."
        exit 0
    }
    echo "Python project files copied"
else
    echo "Warning: Python project files not found in resources. Setup will continue on first launch."
    exit 0
fi

# Run uv sync to install Python dependencies
echo "Installing Python dependencies (this may take a few minutes)..."
if [ -f "$UV_DIR/uv" ] && [ -f "$PROJECT_DIR/pyproject.toml" ]; then
    su - "$REAL_USER" -c "cd '$PROJECT_DIR' && '$UV_DIR/uv' sync" 2>/dev/null && {
        echo "Python dependencies installed successfully"
    } || {
        echo "Warning: uv sync failed. Dependencies will be installed on first launch."
    }
else
    echo "Warning: UV or pyproject.toml not found. Dependencies will be installed on first launch."
fi

echo "Setup complete!"
exit 0
