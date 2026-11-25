"""Shared test setup: a headless app, settings kept away from yours, and test negatives made in code.

Nothing here touches your own key bindings, presets or recovery file: QSettings and the recovery file
are pointed at a temporary folder before the app module is used. Test scans are generated (no real
scans, no licensed sample packs in the repository).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "vendor" / "negpy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dumbconvert-tests-"))
_INI = str(TMP / "settings.ini")


def _isolated_settings(*_args, **_kwargs):
    return QSettings(_INI, QSettings.IniFormat)


from simpleneg import app, engine, exif_panel, keys  # noqa: E402, F401  (engine: for the tests)

for _module in (app, keys, exif_panel):
    _module.QSettings = _isolated_settings
app.MainWindow._recovery_path = staticmethod(lambda: TMP / "recovery.json")

qapp = QApplication.instance() or QApplication([])
qapp.setStyleSheet(app.make_style())


# ---------------------------------------------------------------------------
# Test negatives
# ---------------------------------------------------------------------------
# A colour negative as a camera scan sees it: bright light around the film, the orange film base as a
# border, and a picture inside (a grey ramp, colour patches, a sky-to-ground gradient). Dust specks are
# dark dots: dust blocks the light.
DUST = ((0.42, 0.55), (0.62, 0.40), (0.30, 0.70))      # in 0..1 of the whole scan
PATCHES = {                                             # name -> (scene colour 0..1, where in the picture)
    "grey_dark": ((0.12, 0.12, 0.12), (0.10, 0.75)),
    "grey_mid": ((0.35, 0.35, 0.35), (0.30, 0.75)),
    "grey_light": ((0.75, 0.75, 0.75), (0.50, 0.75)),
    "red": ((0.60, 0.15, 0.12), (0.70, 0.75)),
    "green": ((0.15, 0.50, 0.18), (0.90, 0.75)),
    "blue": ((0.12, 0.20, 0.60), (0.10, 0.25)),
    "skin": ((0.70, 0.50, 0.40), (0.30, 0.25)),
    "yellow": ((0.80, 0.70, 0.15), (0.50, 0.25)),
}
_BASE = np.array([0.78, 0.47, 0.30])                   # orange mask: what clear film passes
_GAMMA = np.array([0.55, 0.62, 0.70])                  # dye layers of different contrast (like real film)
_BW_BASE = np.array([0.55, 0.54, 0.50])                # B&W base plus fog, under a slightly warm light
_BW_GAMMA = np.array([0.65, 0.65, 0.65])
_FILES: dict = {}


def scene(h: int, w: int) -> np.ndarray:
    """The positive picture: a soft gradient with patches of known colour."""
    y, x = np.mgrid[0:h, 0:w]
    img = np.empty((h, w, 3), np.float64)
    img[:] = (0.25 + 0.35 * (y / h))[..., None] * np.array([0.9, 1.0, 1.1]) + 0.1 * (x / w)[..., None]
    ph, pw = int(h * 0.18), int(w * 0.14)
    for colour, (cx, cy) in PATCHES.values():
        y0, x0 = int(cy * h - ph / 2), int(cx * w - pw / 2)
        img[y0:y0 + ph, x0:x0 + pw] = colour
    return np.clip(img, 0.02, 1.0)


def negative(path: Path, size=(1200, 1800), portrait=False, exposure=1.0, dust=True, border=True,
             flat=1.0, bw=False) -> Path:
    """Write a 16-bit TIFF negative of `scene` and return its path.
    `flat` below 1 squeezes the scene's tones toward a mid grey, like fog or an overcast day.
    `bw`: a B&W negative, a grey film base (base plus fog) instead of the orange mask."""
    base, gamma = (_BW_BASE, _BW_GAMMA) if bw else (_BASE, _GAMMA)
    h, w = size
    if portrait:
        h, w = w, h
    film = np.empty((h, w, 3))
    film[:] = base
    m = (int(h * 0.08), int(w * 0.07)) if border else (0, 0)          # film border around the picture
    ih, iw = h - 2 * m[0], w - 2 * m[1]
    pic = scene(ih, iw)
    pic = 0.35 * (pic / 0.35) ** flat * exposure
    if bw:
        pic = pic.mean(-1, keepdims=True)
    film[m[0]:m[0] + ih, m[1]:m[1] + iw] = base * np.clip(pic, 1e-3, None) ** (-gamma) * 0.08
    # grain: colour speckle like a real scan, the same every run
    film *= 1.0 + 0.04 * np.random.default_rng(7).standard_normal((h, w, 3))
    film = np.clip(film, 0.002, base)
    scan = np.ones((h, w, 3)) if border else film                      # the light panel around the film
    if border:
        e = (int(h * 0.04), int(w * 0.035))
        scan[e[0]:h - e[0], e[1]:w - e[1]] = film[e[0]:h - e[0], e[1]:w - e[1]]
    if dust:
        yy, xx = np.mgrid[0:h, 0:w]
        for cx, cy in DUST:
            scan[(xx - cx * w) ** 2 + (yy - cy * h) ** 2 < (h / 300) ** 2] = 0.01
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, (scan * 65535).astype(np.uint16), photometric="rgb")
    return path


def test_file(name: str = "frame", **kwargs) -> Path:
    """A test negative, made once per test run."""
    key = (name, tuple(sorted(kwargs.items())))
    if key not in _FILES:
        _FILES[key] = negative(TMP / "scans" / f"{name}-{len(_FILES)}.tif", **kwargs)
    return _FILES[key]


def test_roll(frames: int = 4, name: str = "Test roll") -> Path:
    """A folder of test negatives (exposed a little differently, like a real roll)."""
    folder = TMP / "rolls" / name
    if not folder.exists():
        for i in range(frames):
            negative(folder / f"frame{i + 1:02d}.tif", exposure=0.8 + 0.15 * i)
    return folder


# ---------------------------------------------------------------------------
# Driving the window
# ---------------------------------------------------------------------------
def pump(seconds: float = 0.1):
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.005)


def idle(win, timeout: float = 120.0) -> bool:
    """Wait until every render and background job of the window has finished."""
    end = time.time() + timeout
    calm = 0
    while time.time() < end:
        qapp.processEvents()
        busy = (win.preview_busy or win.render_timer.isActive() or win._settle_timer.isActive()
                or win.pool.activeThreadCount() or win.preview_pool.activeThreadCount())
        calm = 0 if busy else calm + 1
        if calm >= 10:
            return True
        time.sleep(0.01)
    raise TimeoutError("the window kept working for too long")


_OPEN: list = []


def window(width: int = 1440, height: int = 900):
    win = app.MainWindow()
    _OPEN.append(win)
    win.spotlight_enabled = False      # a roll without a film darkens the window; tests that click turn it off
    win.resize(width, height)
    win.show()
    win.activateWindow()
    pump(0.2)
    return win


def close(win):
    import shiboken6
    if not shiboken6.isValid(win):
        return
    if win in _OPEN:
        _OPEN.remove(win)
    idle(win)
    win._unsaved_ok = True if hasattr(win, "_unsaved_ok") else None
    win.hide()
    win.deleteLater()
    pump(0.1)
    # processEvents() never runs deferred deletes: without this every test's window stayed alive,
    # its app-wide event filters with it, and the suite slowed down test after test.
    from PySide6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def shown_picture(win) -> np.ndarray:
    """The preview as an RGB array."""
    img = win.preview.source.toImage().convertToFormat(win.preview.source.toImage().Format.Format_RGB888)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())
    return a[:, :img.width() * 3].reshape(img.height(), img.width(), 3).copy()


def close_all():
    """Every window a test opened and left open."""
    for win in list(_OPEN):
        close(win)
    _OPEN.clear()
