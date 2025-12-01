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

# Step 2: Create icons (if needed)
echo "Step 2: Checking icons..."
cd "$ELECTRON_DIR"
if [ ! -f "assets/icon.png" ] || [ ! -f "build/icon.png" ]; then
    echo "Creating placeholder icons..."
    ./create-icons.sh || echo "Warning: Icon creation failed. Please create icons manually."
fi
echo "✅ Icons ready"
echo ""

# Step 3: Install Electron dependencies
echo "Step 3: Installing Electron dependencies..."
if [ ! -d "node_modules" ]; then
    npm install
fi
echo "✅ Electron dependencies installed"
echo ""

# Step 4: Build Electron app
echo "Step 4: Building Electron app..."
npm run build
echo ""

echo "✅ Build complete!"
echo "Installers are available in: $ELECTRON_DIR/dist/"
