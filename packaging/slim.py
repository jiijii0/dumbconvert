"""Remove from a built DumbConvert.app what it never uses, then prove nothing left links to it.

Run by packaging/build.sh after PyInstaller and before signing:  python packaging/slim.py dist/DumbConvert.app
Exits with an error (and the build stops) if any binary still in the app needs a library removed here.
"""

import re
import subprocess
import sys
from pathlib import Path

# imagecodecs: the codecs DumbConvert and NegPy use (JPEG XL, colour management, brotli) and the ones
# tifffile needs to read compressed TIFF scans; the scientific and exotic codecs go. AVIF and WebP
# exports are written by Pillow, which brings its own libraries.
KEEP_CODECS = {"_imcd", "_shared", "_shared_cython", "_jpegxl", "_cms", "_brotli", "_deflate", "_zlib",
               "_zlibng", "_zstd", "_jpeg8", "_jpeg", "_ljpeg", "_jpegsof3", "_lzma", "_bz2", "_webp",
               "_png", "_spng", "_lzw", "_packbits", "_delta", "_floatpred"}

# Qt: PDF, QML / Quick and the virtual keyboard come in through plugins the app never loads
QT_FRAMEWORKS = ("QtPdf", "QtQuick", "QtQml", "QtQmlModels", "QtQmlMeta", "QtQmlWorkerScript",
                 "QtVirtualKeyboard", "QtVirtualKeyboardQml", "QtQuickControls2", "QtQuickTemplates2")
QT_PLUGINS = ("imageformats/libqpdf.dylib", "platforminputcontexts/libqtvirtualkeyboardplugin.dylib")


def links(binary: Path) -> list[str]:
    """The libraries a binary loads (a library's own install name, which otool lists too, left out)."""
    out = subprocess.run(["otool", "-L", str(binary)], capture_output=True, text=True).stdout
    own = subprocess.run(["otool", "-D", str(binary)], capture_output=True, text=True).stdout.splitlines()[1:]
    return [m.group(1) for m in re.finditer(r"^\s+(\S+) \(compat", out, re.M) if m.group(1) not in own]


def is_macho(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(4) in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf", b"\xbe\xba\xfe\xca")
    except OSError:
        return False


def remove(path: Path, removed: list):
    if path.is_symlink() or path.is_file():
        removed.append(path)
        path.unlink()
    elif path.is_dir():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_symlink() or child.is_file():
                removed.append(child)
                child.unlink()
            else:
                child.rmdir()
        path.rmdir()


def main(app: Path) -> int:
    frameworks = app / "Contents" / "Frameworks"
    size_before = sum(f.stat().st_size for f in app.rglob("*") if f.is_file() and not f.is_symlink())
    removed: list[Path] = []

    # imagecodecs: unused codec modules, then the shared libraries no kept module needs
    codecs = frameworks / "imagecodecs"
    dylibs = codecs / "__dot__dylibs"
    if codecs.is_dir():
        kept = [so for so in codecs.glob("_*.abi3.so") if so.name.split(".")[0] in KEEP_CODECS]
        for so in codecs.glob("_*.abi3.so"):
            if so not in kept:
                remove(so, removed)
        needed, todo = set(), list(kept)
        while todo:                                  # every library the kept ones load, and theirs
            for ref in links(todo.pop()):
                name = Path(ref).name
                lib = dylibs / name
                if lib.exists() and name not in needed:
                    needed.add(name)
                    todo.append(lib)
        for lib in list(dylibs.glob("*")):
            if lib.name not in needed:
                remove(lib, removed)

    # Qt
    qt = frameworks / "PySide6" / "Qt"
    for fw in QT_FRAMEWORKS:
        remove(qt / "lib" / f"{fw}.framework", removed)
    for plugin in QT_PLUGINS:
        remove(qt / "plugins" / plugin, removed)
    remove(qt / "qml", removed)
    for binding in (frameworks / "PySide6").glob("Qt*.abi3.so"):
        if any(binding.name.startswith(fw + ".") for fw in QT_FRAMEWORKS):
            remove(binding, removed)

    # proof: nothing left in the app links to what was removed
    # (a library removed here may also ship elsewhere in the app under its name, e.g. Pillow's own libavif:
    # a link is broken only when nothing of that name is left)
    left = {f.name for f in app.rglob("*") if f.is_file() or f.is_symlink()}
    gone_frameworks = {f"{fw}.framework" for fw in QT_FRAMEWORKS}
    broken = []
    for f in app.rglob("*"):
        if f.is_file() and not f.is_symlink() and is_macho(f):
            for ref in links(f):
                name = Path(ref).name
                if set(Path(ref).parts) & gone_frameworks or (ref.startswith("@") and name not in left):
                    broken.append(f"{f.relative_to(app)} -> {ref}")
    size_after = sum(f.stat().st_size for f in app.rglob("*") if f.is_file() and not f.is_symlink())
    print(f"slim: removed {len(removed)} files, {(size_before - size_after) / 1e6:.1f} MB "
          f"({size_before / 1e6:.0f} -> {size_after / 1e6:.0f} MB)")
    if broken:
        print("slim: these still link to something removed:\n  " + "\n  ".join(sorted(set(broken))), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
