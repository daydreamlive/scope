#!/bin/bash
# Download uv binaries for different platforms
# This script downloads uv binaries that will be bundled with the Electron app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_DIR="$SCRIPT_DIR/uv-binaries"

# Create directory for uv binaries
mkdir -p "$UV_DIR"

# Detect current platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)
            echo "unknown-linux-gnu"
            ;;
        Darwin*)
            if [ "$(uname -m)" = "arm64" ]; then
                echo "aarch64-apple-darwin"
            else
                echo "x86_64-apple-darwin"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "x86_64-pc-windows-msvc"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Download uv for a specific platform
download_uv() {
    local platform=$1
    local output_dir="$UV_DIR/$platform"

    mkdir -p "$output_dir"

    echo "Downloading uv for $platform..."

    if [ "$platform" = "x86_64-pc-windows-msvc" ]; then
        local url="https://github.com/astral-sh/uv/releases/latest/download/uv-${platform}.zip"
        local output_file="$output_dir/uv.zip"
        curl -L -o "$output_file" "$url" || return 1
        unzip -q "$output_file" -d "$output_dir" || return 1
        rm "$output_file"
        # On Windows, the binary is uv.exe
        if [ ! -f "$output_dir/uv.exe" ]; then
            echo "Warning: uv.exe not found after extraction"
            return 1
        fi
        chmod +x "$output_dir/uv.exe" 2>/dev/null || true
    else
        local url="https://github.com/astral-sh/uv/releases/latest/download/uv-${platform}.tar.gz"
        local output_file="$output_dir/uv.tar.gz"
        curl -L -o "$output_file" "$url" || return 1
        tar -xzf "$output_file" -C "$output_dir" || return 1
        rm "$output_file"
        # Find the uv binary (might be in a subdirectory)
        if [ -f "$output_dir/uv" ]; then
            chmod +x "$output_dir/uv"
        elif [ -f "$output_dir/uv-${platform}/uv" ]; then
            mv "$output_dir/uv-${platform}/uv" "$output_dir/uv"
            chmod +x "$output_dir/uv"
        else
            echo "Warning: uv binary not found after extraction"
            find "$output_dir" -name "uv" -type f
            return 1
        fi
    fi

    echo "✅ Downloaded uv for $platform"
}

# Download uv for all target platforms
PLATFORMS=(
    "x86_64-apple-darwin"
    "aarch64-apple-darwin"
    "x86_64-pc-windows-msvc"
    "x86_64-unknown-linux-gnu"
)

echo "Downloading uv binaries..."
for platform in "${PLATFORMS[@]}"; do
    download_uv "$platform" || echo "Failed to download uv for $platform (this is OK if building for different platform)"
done

echo ""
echo "✅ uv binaries downloaded to: $UV_DIR"
echo "Available platforms:"
ls -la "$UV_DIR" 2>/dev/null || echo "No binaries downloaded yet"
