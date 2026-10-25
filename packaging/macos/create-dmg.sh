#!/usr/bin/env bash
set -euo pipefail

APP="dist/XAI-32k.app"
DMG="XAI-32k.dmg"
STAGING="dist/dmg-staging"
BG="packaging/macos/dmg-background.png"

if [ ! -d "$APP" ]; then
  echo "FATAL: $APP not found. Run pyinstaller first."
  exit 1
fi

# ensure create-dmg is installed
if ! command -v create-dmg >/dev/null; then
  echo "Installing create-dmg via Homebrew…"
  brew install create-dmg
fi

rm -rf "$STAGING" "$DMG"
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/"

create-dmg \
  --volname "32k-XAI" \
  --volicon "assets/app-icon.icns" \
  --background "$BG" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "XAI-32k.app" 150 200 \
  --hide-extension "XAI-32k.app" \
  --app-drop-link 450 200 \
  --no-internet-enable \
  "$DMG" "$STAGING"

echo ""
echo "Built: $DMG"
ls -lh "$DMG"
