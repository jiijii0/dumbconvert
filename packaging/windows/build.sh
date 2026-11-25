#!/bin/bash
# Build the Windows (x86-64) version of DumbConvert on a Mac.   Run:  ./packaging/windows/build.sh
#
# Windows apps are normally built on Windows. This one is put together from parts that do not need it:
#   - NegPy's Rust core, cross-compiled for Windows (Rust + zig, kept in build/windows-tools, nothing
#     is installed on the Mac itself),
#   - the official Windows "embeddable" Python from python.org, with Windows wheels from PyPI,
#   - the app's own Python code, unchanged,
#   - DumbConvert.exe, a tiny launcher (launcher.c) that starts that Python.
# Result: dist/windows/DumbConvert/ and dist/windows/DumbConvert-<version>-windows-x64.zip
set -e
cd "$(dirname "$0")/../.."
ROOT=$PWD
HERE=$ROOT/packaging/windows
TOOLS=${TOOLS:-$ROOT/build/windows-tools}
OUT=$ROOT/dist/windows
APP=$OUT/DumbConvert
PY_VERSION=3.14.10                     # the same Python line as the Mac app (3.14)
TARGET=x86_64-pc-windows-gnu
VERSION=$(.venv/bin/python -c "import simpleneg; print(simpleneg.__version__)")
# The same package versions as the Mac build
WHEELS="PySide6-Essentials==6.12.0 shiboken6==6.12.0 numpy==2.5.4 pillow==13.0.0 rawpy==0.27.0 tifffile==2026.6.1
        imagecodecs==2026.11.11 imageio==2.38.1 piexif==1.1.3 wgpu==0.31.1 pillow-heif==1.8.1"

mkdir -p "$TOOLS" "$OUT"
export RUSTUP_HOME=$TOOLS/rustup CARGO_HOME=$TOOLS/cargo CARGO_TARGET_DIR=$TOOLS/target
export PATH=$CARGO_HOME/bin:$TOOLS/venv/bin:$PATH

echo "== 1. Tools (first run only)"
if [ ! -x "$CARGO_HOME/bin/rustup" ]; then
  curl -sSfL -o "$TOOLS/rustup-init" "https://static.rust-lang.org/rustup/dist/$(uname -m | sed 's/arm64/aarch64/')-apple-darwin/rustup-init"
  chmod +x "$TOOLS/rustup-init"
  "$TOOLS/rustup-init" -y -q --no-modify-path --profile minimal --default-toolchain none
fi
CHANNEL=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' vendor/negpy/negpy_core/rust-toolchain.toml)
rustup toolchain install "$CHANNEL" --profile minimal --target "$TARGET" >/dev/null
if [ ! -x "$TOOLS/venv/bin/maturin" ]; then
  .venv/bin/python -m venv "$TOOLS/venv"
  "$TOOLS/venv/bin/pip" install -q maturin ziglang
fi

echo "== 2. NegPy's Rust core for Windows"
rm -rf "$TOOLS/negpy_core" "$TOOLS/wheels"
rsync -a --exclude target vendor/negpy/negpy_core/ "$TOOLS/negpy_core/"     # built from a copy, so vendor/negpy stays clean
(cd "$TOOLS/negpy_core" && maturin build -q --release --target "$TARGET" --zig \
   --features python,pyo3/generate-import-lib -o "$TOOLS/wheels")

echo "== 3. Windows Python $PY_VERSION and the Windows packages"
EMBED=$TOOLS/python-$PY_VERSION-embed-amd64.zip
[ -f "$EMBED" ] || curl -sSfL -o "$EMBED" "https://www.python.org/ftp/python/$PY_VERSION/python-$PY_VERSION-embed-amd64.zip"
"$TOOLS/venv/bin/pip" download -q --dest "$TOOLS/wheels" --platform win_amd64 --python-version 3.14 \
  --implementation cp --only-binary=:all: $WHEELS

echo "== 4. Putting it together"
rm -rf "$APP"
mkdir -p "$APP/python/Lib/site-packages" "$APP/app/vendor/negpy"
(cd "$APP/python" && unzip -q "$EMBED")
for whl in "$TOOLS"/wheels/*.whl; do
  case "$(basename "$whl")" in rubicon_objc-*) continue ;; esac            # macOS only
  (cd "$APP/python/Lib/site-packages" && unzip -q -o "$whl")
done
SITE=$APP/python/Lib/site-packages
for data in "$SITE"/*.data; do                 # what pip would install from a wheel's .data folder
  for kind in platlib purelib; do
    [ -d "$data/$kind" ] && cp -R "$data/$kind/." "$SITE/"
  done
  rm -rf "$data"
done
# Qt's developer tools and the QML / Quick parts: the app uses plain widgets only
Q=$SITE/PySide6
rm -rf "$Q"/*.exe "$Q"/qml "$Q"/translations "$Q"/metatypes "$Q"/include "$Q"/typesystems "$Q"/glue \
       "$Q"/plugins/designer "$Q"/plugins/qmltooling "$Q"/Qt6Designer*.dll "$Q"/Qt6Quick*.dll "$Q"/Qt6Qml*.dll \
       "$Q"/QtQuick*.pyd "$Q"/QtQml.pyd "$Q"/QtDesigner.pyd "$Q"/QtUiTools.pyd "$Q"/Qt6UiTools.dll \
       "$Q"/Qt6Labs*.dll "$Q"/Qt6Lottie*.dll "$Q"/plugins/vectorimageformats "$Q"/plugins/sqldrivers \
       "$Q"/plugins/imageformats/qpdf.dll "$Q"/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll \
       "$Q"/QtLabs*.pyd "$Q"/QtQml*.pyd "$Q"/pyside6qml*.dll
# Where this Python looks for code: its own library, the packages, the app and NegPy
PTH=$(ls "$APP"/python/python3*._pth)
printf 'python314.zip\r\n.\r\nLib\\site-packages\r\n..\\app\r\n..\\app\\vendor\\negpy\r\nimport site\r\n' > "$PTH"
rsync -a --exclude __pycache__ simpleneg "$APP/app/"
for part in negpy gear icc crosstalk VERSION; do
  rsync -a --exclude __pycache__ --exclude '*.pyc' "vendor/negpy/$part" "$APP/app/vendor/negpy/"
done
cp LICENSE "$APP/"
cp "$HERE/README.txt" "$APP/README.txt"

echo "== 5. DumbConvert.exe"
.venv/bin/python -c "
from PIL import Image
Image.open('packaging/DumbConvert.icns').save('$TOOLS/DumbConvert.ico', sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])"
cp "$HERE/launcher.rc" "$TOOLS/launcher.rc"
(cd "$TOOLS" && "$TOOLS/venv/bin/python" -m ziglang cc -target x86_64-windows-gnu -O2 -s -Wl,--subsystem,windows \
   "$HERE/launcher.c" launcher.rc -o "$APP/DumbConvert.exe")

rm -f "$OUT/DumbConvert-$VERSION-windows-x64.zip"
(cd "$OUT" && zip -q -r "DumbConvert-$VERSION-windows-x64.zip" DumbConvert)
echo "Built: $APP  ($(du -sh "$APP" | cut -f1))   zip: $OUT/DumbConvert-$VERSION-windows-x64.zip"
