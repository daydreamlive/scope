#!/bin/bash
# Simple script to create placeholder icons from the existing icon.svg
# This is a temporary solution until final icons are designed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="$SCRIPT_DIR/assets"
BUILD_DIR="$SCRIPT_DIR/build"
FRONTEND_PUBLIC="$SCRIPT_DIR/../frontend/public"

# Create directories if they don't exist
mkdir -p "$ASSETS_DIR"
mkdir -p "$BUILD_DIR"

# Check if ImageMagick is available
if command -v convert &> /dev/null; then
    echo "Using ImageMagick to create icons..."

    # Create PNG icons from SVG
    if [ -f "$FRONTEND_PUBLIC/icon.svg" ]; then
        # Main icon (512x512)
        convert "$FRONTEND_PUBLIC/icon.svg" -resize 512x512 "$ASSETS_DIR/icon.png"
        cp "$ASSETS_DIR/icon.png" "$BUILD_DIR/icon.png"

        # Tray icon (32x32)
        convert "$FRONTEND_PUBLIC/icon.svg" -resize 32x32 "$ASSETS_DIR/tray-icon.png"

        # Windows ICO (256x256)
        convert "$FRONTEND_PUBLIC/icon.svg" -resize 256x256 "$BUILD_DIR/icon.ico"

        echo "Icons created successfully!"
    else
        echo "Warning: icon.svg not found at $FRONTEND_PUBLIC/icon.svg"
        echo "Creating placeholder icons..."

        # Create a simple placeholder
        convert -size 512x512 xc:black -fill white -gravity center -pointsize 200 -annotate +0+0 "DS" "$ASSETS_DIR/icon.png"
        cp "$ASSETS_DIR/icon.png" "$BUILD_DIR/icon.png"
        convert -size 32x32 xc:black -fill white -gravity center -pointsize 20 -annotate +0+0 "DS" "$ASSETS_DIR/tray-icon.png"
        convert -size 256x256 xc:black -fill white -gravity center -pointsize 100 -annotate +0+0 "DS" "$BUILD_DIR/icon.ico"
    fi
else
    echo "ImageMagick not found. Creating simple placeholder icons using basic tools..."

    # Create a simple text-based placeholder using a basic approach
    # This is a fallback - ideally you'd use ImageMagick or a design tool
    echo "Please install ImageMagick (convert) or create icons manually:"
    echo "  - $ASSETS_DIR/icon.png (512x512)"
    echo "  - $ASSETS_DIR/tray-icon.png (32x32)"
    echo "  - $BUILD_DIR/icon.png (512x512)"
    echo "  - $BUILD_DIR/icon.ico (256x256, Windows)"
    echo "  - $BUILD_DIR/icon.icns (macOS bundle)"
    echo ""
    echo "You can use online tools or design software to create these from icon.svg"
fi
