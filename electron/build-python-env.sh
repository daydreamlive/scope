#!/bin/bash
# Build script to create a standalone Python environment with all dependencies
# This environment will be bundled with the Electron app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_ENV_DIR="$SCRIPT_DIR/python-env"

echo "Building standalone Python environment..."
echo ""

# Detect platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)
            echo "linux"
            ;;
        Darwin*)
            if [ "$(uname -m)" = "arm64" ]; then
                echo "macos-arm64"
            else
                echo "macos-x64"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "windows"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

PLATFORM=$(detect_platform)
echo "Platform: $PLATFORM"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Using uv: $(which uv)"
echo "uv version: $(uv --version)"
echo ""

# Clean previous build
if [ -d "$PYTHON_ENV_DIR" ]; then
    echo "Cleaning previous Python environment..."
    rm -rf "$PYTHON_ENV_DIR"
fi

mkdir -p "$PYTHON_ENV_DIR"

# Copy project files needed for uv sync
echo "Preparing project files..."
cp "$PROJECT_ROOT/pyproject.toml" "$PYTHON_ENV_DIR/"
cp "$PROJECT_ROOT/uv.lock" "$PYTHON_ENV_DIR/"
cp "$PROJECT_ROOT/README.md" "$PYTHON_ENV_DIR/"
cp "$PROJECT_ROOT/LICENSE.md" "$PYTHON_ENV_DIR/"
cp "$PROJECT_ROOT/.python-version" "$PYTHON_ENV_DIR/" 2>/dev/null || true

# Copy source code
echo "Copying source code..."
cp -r "$PROJECT_ROOT/src" "$PYTHON_ENV_DIR/"

# Create Python environment with all dependencies
echo ""
echo "Creating Python environment with all dependencies..."
echo "This may take several minutes..."

cd "$PYTHON_ENV_DIR"

# Use uv sync to create a complete environment
# This will download Python and install all dependencies
# Note: --no-dev excludes dev dependencies to reduce size
uv sync --frozen --no-dev

# Verify the environment was created and Python is available

# Verify the environment was created
if [ ! -d ".venv" ]; then
    echo "Error: Failed to create Python environment"
    exit 1
fi

# Find Python executable
if [ "$PLATFORM" = "windows" ]; then
    PYTHON_EXE=".venv/Scripts/python.exe"
else
    PYTHON_EXE=".venv/bin/python"
fi

if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python executable not found at $PYTHON_EXE"
    exit 1
fi

echo ""
echo "✅ Python environment created successfully!"
echo "Python version: $($PYTHON_EXE --version)"
echo "Location: $PYTHON_ENV_DIR/.venv"
echo ""

# Verify the package is installed
echo "Verifying installation..."
if [ "$PLATFORM" = "windows" ]; then
    "$PYTHON_EXE" -m scope.server.app --help > /dev/null 2>&1 || echo "Warning: Could not verify scope.server.app"
else
    "$PYTHON_EXE" -m scope.server.app --help > /dev/null 2>&1 || echo "Warning: Could not verify scope.server.app"
fi

echo ""
echo "✅ Standalone Python environment ready for bundling!"
