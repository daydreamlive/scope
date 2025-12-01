# App Icons

This directory contains the application icons for Daydream Scope.

## Required Icons

For a complete Electron app, you'll need:

1. **icon.png** (512x512) - Main application icon (PNG format)
2. **icon.ico** (256x256) - Windows icon
3. **icon.icns** - macOS icon bundle
4. **tray-icon.png** (16x16 or 32x32) - System tray icon

## Icon Design Guidelines

Based on the PRD, the icon should:
- Align with the Daydream brand
- Convey a sense of exploration, scouting new territory
- Represent "scoping out what is possible"
- Show "seeing what you otherwise might not see"

## Creating Icons

### From SVG (if you have icon.svg)

1. **PNG (512x512)**:
   ```bash
   # Using ImageMagick or similar
   convert icon.svg -resize 512x512 icon.png
   ```

2. **ICO (Windows)**:
   ```bash
   # Using ImageMagick
   convert icon.svg -resize 256x256 icon.ico
   ```

3. **ICNS (macOS)**:
   ```bash
   # Using iconutil (macOS only)
   mkdir icon.iconset
   # Create various sizes and convert
   iconutil -c icns icon.iconset -o icon.icns
   ```

### Tray Icon

The tray icon should be a simplified version, typically:
- 16x16 or 32x32 pixels
- Monochrome or simple design
- Visible on both light and dark backgrounds

## Placeholder

Until the final icon is designed, you can use a simple placeholder or the existing icon.svg converted to PNG format.
