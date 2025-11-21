#!/bin/bash

# Build script for daydream-scope
# This script builds the distribution files needed for publishing

set -e  # Exit on any error

echo "🔨 Building daydream-scope..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Build frontend first (required for the package)
echo "📦 Building frontend..."
if [ -d "frontend" ]; then
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "📥 Installing frontend dependencies..."
        npm install
    fi
    echo "🔨 Building frontend assets..."
    npm run build
    cd ..
else
    echo "⚠️  Warning: frontend directory not found. Skipping frontend build."
fi

# Build the Python package distribution
echo "📦 Building Python package..."
uv build

# Verify build output
if [ ! -d "dist" ]; then
    echo "❌ Error: dist directory not created. Build may have failed."
    exit 1
fi

if ! ls dist/*.whl 1> /dev/null 2>&1; then
    echo "❌ Error: No wheel files found in dist directory."
    exit 1
fi

echo "✅ Build completed successfully!"
echo "📦 Distribution files are in the dist/ directory"
echo ""
echo "Next steps:"
echo "  1. Set TestPyPI credentials:"
echo "     export TWINE_USERNAME=__token__"
echo "     export TWINE_PASSWORD=your_testpypi_token_here"
echo "  2. Run ./publish.sh to upload to TestPyPI"
