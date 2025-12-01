#!/bin/bash
# Build script for Electron app
# This script builds the frontend and then packages the Electron app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
ELECTRON_DIR="$SCRIPT_DIR"

echo "Building Daydream Scope Electron App..."
echo ""

# Step 1: Build frontend
echo "Step 1: Building frontend..."
cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi
npm run build
echo "✅ Frontend build complete"
echo ""

# Step 2: Build standalone Python environment
echo "Step 2: Building standalone Python environment..."
cd "$ELECTRON_DIR"
if [ ! -d "python-env/.venv" ]; then
    echo "Creating Python environment with all dependencies..."
    ./build-python-env.sh || {
        echo "Error: Failed to build Python environment"
        echo "Make sure uv is installed: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    }
else
    echo "Python environment already exists, skipping..."
fi
echo "✅ Python environment ready"
echo ""

# Step 3: Create icons (if needed)
echo "Step 3: Checking icons..."
cd "$ELECTRON_DIR"
if [ ! -f "assets/icon.png" ] || [ ! -f "build/icon.png" ]; then
    echo "Creating placeholder icons..."
    ./create-icons.sh || echo "Warning: Icon creation failed. Please create icons manually."
fi
echo "✅ Icons ready"
echo ""

# Step 4: Download uv binaries (for fallback, though we'll use bundled Python)
echo "Step 4: Downloading uv binaries (for fallback)..."
cd "$ELECTRON_DIR"
if [ ! -d "uv-binaries" ] || [ -z "$(ls -A uv-binaries 2>/dev/null)" ]; then
    echo "Downloading uv binaries for all platforms..."
    ./download-uv.sh || echo "Warning: uv binary download failed. Will use bundled Python only."
fi
echo "✅ uv binaries ready"
echo ""

# Step 5: Install Electron dependencies
echo "Step 5: Installing Electron dependencies..."
if [ ! -d "node_modules" ]; then
    npm install
fi
echo "✅ Electron dependencies installed"
echo ""

# Step 6: Build Electron app
echo "Step 6: Building Electron app..."
npm run build
echo ""

echo "✅ Build complete!"
echo "Installers are available in: $ELECTRON_DIR/dist/"
