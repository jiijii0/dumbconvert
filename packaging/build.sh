#!/bin/bash
# Build DumbConvert.app (and a zip to send) with PyInstaller.   Run:  ./packaging/build.sh
# The app is for the Mac it is built on: Apple Silicon (arm64) here.
set -e
cd "$(dirname "$0")/.."
N=vendor/negpy
rm -rf build dist
.venv/bin/pyinstaller --noconfirm --windowed --name DumbConvert \
  --icon packaging/DumbConvert.icns \
  --osx-bundle-identifier app.dumbconvert \
  --paths . --paths "$N" \
  --collect-submodules negpy.features --collect-submodules negpy.services \
  --collect-submodules negpy.kernel --collect-submodules negpy.domain --collect-submodules negpy.infrastructure \
  --collect-all negpy_core --collect-all rawpy --collect-all imagecodecs --collect-all pillow_heif \
  --collect-all tifffile --collect-all wgpu \
  --exclude-module PyQt6 --exclude-module pypdf --exclude-module pdfminer \
  --exclude-module matplotlib --exclude-module tkinter --exclude-module cryptography \
  --add-data "$N/negpy/features/exposure/shaders:negpy/features/exposure/shaders" \
  --add-data "$N/negpy/features/exposure/print_luts:negpy/features/exposure/print_luts" \
  --add-data "$N/negpy/features/geometry/shaders:negpy/features/geometry/shaders" \
  --add-data "$N/negpy/features/toning/shaders:negpy/features/toning/shaders" \
  --add-data "$N/negpy/features/lab/shaders:negpy/features/lab/shaders" \
  --add-data "$N/negpy/features/lith/shaders:negpy/features/lith/shaders" \
  --add-data "$N/negpy/features/cyanotype/shaders:negpy/features/cyanotype/shaders" \
  --add-data "$N/negpy/features/sabattier/shaders:negpy/features/sabattier/shaders" \
  --add-data "$N/negpy/features/finish/shaders:negpy/features/finish/shaders" \
  --add-data "$N/negpy/features/transparency/shaders:negpy/features/transparency/shaders" \
  --add-data "$N/icc:icc" --add-data "$N/crosstalk:crosstalk" --add-data "$N/gear:gear" \
  --add-data "$N/VERSION:." \
  --add-data "LICENSE:." \
  --add-data "simpleneg/icons:simpleneg/icons" \
  packaging/dumbconvert.py
# What the app never uses (unused image codecs, Qt's PDF/QML/virtual keyboard); stops if anything still needs it
.venv/bin/python packaging/slim.py dist/DumbConvert.app
# Memory: macOS gives freed picture memory back at once instead of caching gigabytes of it
# (see main() in simpleneg/app.py). Must be in the Info.plist: it is read when the app starts.
VERSION=$(.venv/bin/python -c "import simpleneg; print(simpleneg.__version__)")
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" dist/DumbConvert.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :LSEnvironment dict" dist/DumbConvert.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :LSEnvironment:MallocSpaceEfficient string 1" dist/DumbConvert.app/Contents/Info.plist
# ad-hoc signature: an unsigned Apple Silicon app is refused outright as "damaged"
codesign --force --deep --sign - dist/DumbConvert.app
# zip with ditto, which keeps the app's symlinks and signature intact
ditto -c -k --keepParent dist/DumbConvert.app "dist/DumbConvert-$VERSION-macos-arm64.zip"
echo "Built: dist/DumbConvert.app  ($(du -sh dist/DumbConvert.app | cut -f1))   zip: dist/DumbConvert-$VERSION-macos-arm64.zip"
