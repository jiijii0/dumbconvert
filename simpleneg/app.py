"""The desktop app (Qt / PySide6). The conversion itself is NegPy's (see engine.py).

Layout:
    +----------------+---------------------------+----------------------+
    | rolls & frames |   "Image 3 of 6: x.dng"   | Add roll / frames .. |
    | roll details   |                           | L  Mirror  R         |
    |                |       big preview         | Contrast Density Sat |
    |                |                           | C  M  Y  + Auto WB   |
    |                |                           | > Advanced           |
    |                |   Undo Redo ... Edit Crop | Prev/Next  Export    |
    +----------------+---------------------------+----------------------+
    | filmstrip: [1 D+1 M-2] [2 0 0 0 0] [3 ...]                        |
    +-------------------------------------------------------------------+

Once frames are loaded, the whole workflow also runs from the keyboard,
like a lab scanner. Which key does what is set in the "Keys" tab
(see keys.py).

Key ideas for learning Qt:
  * Widgets are nested in layouts (QHBoxLayout / QVBoxLayout / QGridLayout).
  * Signals & slots: e.g. `button.clicked.connect(self.on_click)`.
  * Slow work (decoding, rendering, exporting) runs on thread pools so the
    window never freezes. Workers report back with signals, which Qt delivers
    safely on the main (UI) thread.
"""

import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import Property, QEasingCurve, QEvent, QVariantAnimation, QItemSelectionModel, QPropertyAnimation, QSettings, QObject, QPoint, QPointF, QRect, QRectF, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import (QBrush, QColor, QCursor, QDesktopServices, QFont, QIcon, QImage, QKeySequence, QLinearGradient,
                           QRadialGradient,
                           QPainter, QPainterPath, QPen, QPolygonF,
                           QPixmap, QShortcut, QTransform)
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QCompleter, QDoubleSpinBox, QFormLayout, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLayout, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget,
    QStyledItemDelegate, QToolButton, QToolTip, QVBoxLayout, QWidget,
)

from . import engine, exif_panel, keys

APP_NAME = "DumbConvert"
PREVIEW_SIZE = 1600   # long edge of the live preview
PREVIEW_DRAFT = 800   # while a slider is moving: ~3x faster, replaced by a full render on pause
# Priority in the worker pool for work someone is waiting for on screen (healing a click, a crop,
# white balance), ahead of the thumbnails of a roll that is still loading
URGENT = 2
THUMB_SIZE = 120     # filmstrip thumbnails: fit in THUMB_SIZE x THUMB_H
THUMB_H = 80

def corrections(st: "engine.Settings") -> dict[str, int]:
    """Corrections in keypad steps from the automatic conversion, as a lab scanner shows them."""
    steps = {"density": 0.05, "cyan": 0.02, "magenta": 0.02, "yellow": 0.02, "contrast": 0.05, "saturation": 0.05}
    return {k: round((getattr(st, k) - 1.0) / v) for k, v in steps.items()}


def correction_text(st: "engine.Settings", short: bool = True) -> str:
    c = corrections(st)
    f = lambda v: f"{v:+d}" if v else "0"
    text = f"D{f(c['density'])} C{f(c['cyan'])} M{f(c['magenta'])} Y{f(c['yellow'])}"
    if not short:
        text = f"D {f(c['density'])}   C {f(c['cyan'])}   M {f(c['magenta'])}   Y {f(c['yellow'])}"
        text += f"     Contrast {f(c['contrast'])}   Sat {f(c['saturation'])}"
    return text


@dataclass(eq=False)
class Photo:
    path: Path
    settings: engine.Settings = field(default_factory=engine.Settings)
    item: QListWidgetItem | None = None    # only while its group is on screen
    thumb: np.ndarray | None = None
    ok: bool = False          # first render succeeded
    note: str | None = None   # a warning about the scan itself (engine.scan_note); None = not checked yet
    failed: bool = False
    group: "Group | None" = None
    history: list = field(default_factory=list)   # settings snapshots for undo/redo
    hindex: int = -1
    exif: dict = field(default_factory=dict)      # this frame's own EXIF values (else the roll's)
    exported: str = ""        # the settings it was last exported with (to know what would be lost on quit)
    levelled: float = 0.0     # the angle Auto crop straightened it by (taken back when Auto goes off)


@dataclass(eq=False)
class Group:
    """A roll (frames of one film, analysed together) or the loose single frames."""
    kind: str                 # "roll" or "frames"
    name: str
    photos: list = field(default_factory=list)
    stock_id: str = ""        # film stock (NegPy catalogue id), rolls only
    no_stock: bool = False    # "No film stock" chosen on purpose: the roll converts without one
    camera: str = ""          # camera the film was shot with, rolls only
    lens: str = ""
    date: str = ""            # "2024", "2024-06" or "2024-06-21"
    location: str = ""        # GPS coordinates "lat, lon" for the whole roll
    location_label: str = ""  # the place picked in the search, shown instead of the numbers
    push_pull: int = 0        # stops: -3 .. +3
    analyzed: bool = False
    analyzing: bool = False
    progress: float = 0.0     # 0 .. 1 while analysing
    analysis_id: int = 0      # the analysis that counts: a newer one cancels the one running
    outliers: int = 0
    card: "RollCard | None" = None

    @property
    def stock(self) -> dict | None:
        return engine.stock_by_id(self.stock_id) if self.stock_id else None

    @property
    def film_chosen(self) -> bool:
        """A film stock was picked, or "No film stock": the roll can be analysed."""
        return bool(self.stock_id) or self.no_stock

    def label(self) -> tuple[str, str]:
        """(title, subtitle) for its card in the left column."""
        n = len(self.photos)
        if self.kind == "frames":
            return "Single frames", f"{n} frame{'s' if n != 1 else ''}"
        film = self.stock["displayName"] if self.stock else "No film stock" if self.no_stock else FILM_NEEDED
        state = "  ·  analysing…" if self.analyzing else ""
        return self.name, f"{film}  ·  {n} frame{"s" if n != 1 else ""}{state}"


FILM_NEEDED = "Choose the film stock"   # a roll's card until it has one: colours and Lab print follow it


# Settings that come from measuring or from the scanning setup, not from editing a frame.
NOT_EDITS = ("roll_floors", "roll_ceils", "roll_axis", "roll_slope", "raw_wb", "flat_field", "light", "film", "grayscale",  # from the setup / film stock
             "auto_rect", "auto_key")                                                     # where Auto crop found the frame
# Settings that stay with one frame when copying settings to other frames.
PER_FRAME = ("rotation", "fine_rotation", "mirror", "crop", "auto_crop", "auto_rect", "auto_key", "analysis_rect",
             "heal_spots", "heal_areas")


def edit_key(st: engine.Settings) -> str:
    """A frame's adjustments as text (what was measured or set up is left out)."""
    return repr([getattr(st, f) for f in st.__dataclass_fields__ if f not in NOT_EDITS])


def same_edits(a: engine.Settings, b: engine.Settings) -> bool:
    return all(getattr(a, f) == getattr(b, f) for f in a.__dataclass_fields__ if f not in NOT_EDITS)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------
class RollProgress(QWidget):
    """At the foot of the window: a roll's analysis, "Roll: analysing… 42 %" over a thin bar in the
    accent colour that fills smoothly. Says "roll analysed" with the bar full for a moment, then goes."""

    BAR_W, BAR_H = 120, 3

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 0, 0)      # in line with the status bar's messages
        lay.setSpacing(10)
        self.label = QLabel()
        self.label.setStyleSheet(f"color: {COLORS['text']}; font-size: 11px;")
        lay.addWidget(self.label)
        self.bar = QWidget()
        self.bar.setFixedSize(self.BAR_W, 12)
        self.bar.paintEvent = self._paint_bar
        lay.addWidget(self.bar)
        self._value = 0.0
        self.waiting = False                      # asking for the film stock, no bar
        self._anim = QVariantAnimation(self, duration=260, easingCurve=QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._set_drawn)
        self._hide = QTimer(self, singleShot=True, interval=2500, timeout=self.hide)
        self.hide()

    def value(self) -> float:
        return self._value

    def _set_drawn(self, v):
        self._value = float(v)
        self.bar.update()

    def _appear(self):
        """Show, taking the status bar's place: a message there would be drawn over the line."""
        if not self.isVisible():
            bar = self.parentWidget()
            if hasattr(bar, "clearMessage"):
                bar.clearMessage()
            self.show()

    def show_waiting(self, name: str):
        """No analysis until the roll's film is chosen: say so, in the accent, without a bar."""
        self._hide.stop()
        self.waiting = True
        self.label.setText(f"{name}: choose the film stock to analyse the roll")
        self.label.setStyleSheet(f"color: {ACCENT2}; font-size: 11px;")
        self.bar.hide()
        self._appear()

    def _working(self):
        if self.waiting:
            self.waiting = False
            self.label.setStyleSheet(f"color: {COLORS['text']}; font-size: 11px;")
            self.bar.show()

    def show_progress(self, name: str, frac: float):
        self._hide.stop()
        if self.waiting:
            self._working()
            self.hide()
        frac = min(max(frac, 0.0), 1.0)
        self.label.setText(f"{name}: analysing…  {round(frac * 100)} %")
        if not self.isVisible():
            self._anim.stop()
            self._set_drawn(0.0)
            self._appear()
        self._glide(frac)

    def show_done(self, name: str):
        self._working()
        self.label.setText(f"{name}: roll analysed")
        self._glide(1.0)
        self._appear()
        self._hide.start()

    def _glide(self, to: float):
        if to < self._value:                      # a new analysis: start again from empty
            self._anim.stop()
            self._set_drawn(to)
            return
        self._anim.stop()
        self._anim.setStartValue(self._value)
        self._anim.setEndValue(to)
        self._anim.start()

    def _paint_bar(self, _event):
        p = QPainter(self.bar)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        y = (self.bar.height() - self.BAR_H) / 2
        r = self.BAR_H / 2
        p.setBrush(QColor(COLORS["line"]))
        p.drawRoundedRect(QRectF(0, y, self.BAR_W, self.BAR_H), r, r)
        if self._value > 0:
            p.setBrush(QColor(ACCENT))
            p.drawRoundedRect(QRectF(0, y, max(self.BAR_H, self.BAR_W * self._value), self.BAR_H), r, r)
        p.end()


class AccentPicker(QWidget):
    """Settings › Accent: a round swatch per accent; the chosen one has a ring."""
    chosen = Signal(str)
    D, GAP = 20, 10

    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        self.current = current if current in ACCENTS else "purple"
        self._hover: str | None = None
        n = len(ACCENTS)
        self.setFixedSize(n * self.D + (n - 1) * self.GAP + 8, self.D + 8)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def _rects(self):
        for i, key in enumerate(ACCENTS):
            yield key, QRectF(4 + i * (self.D + self.GAP), 4, self.D, self.D)

    def _at(self, pos) -> str | None:
        return next((k for k, r in self._rects() if r.adjusted(-3, -3, 3, 3).contains(QPointF(pos))), None)

    def mouseMoveEvent(self, e):
        k = self._at(e.position())
        if k != self._hover:
            self._hover = k
            self.setToolTip(ACCENTS[k][0] if k else "")
            self.update()

    def leaveEvent(self, e):
        self._hover = None
        self.update()

    def mousePressEvent(self, e):
        k = self._at(e.position())
        if k and k != self.current:
            self.current = k
            self.update()
            self.chosen.emit(k)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        for key, r in self._rects():
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACCENTS[key][2 if key == self._hover else 1]))
            p.drawEllipse(r.adjusted(3, 3, -3, -3) if key == self.current else r.adjusted(1, 1, -1, -1))
            if key == self.current:                   # a ring around the chosen one
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(COLORS["text"]), 1.4))
                p.drawEllipse(r.adjusted(0.7, 0.7, -0.7, -0.7))
        p.end()


class InfoDot(QWidget):
    """An "i" in a ring beside the file name: hovering it shows the frame's details at once.
    In the warning colour when the scan itself has a problem."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self.info_html = ""
        self.warn = False
        self._hover = False

    def set_info(self, html: str, warn: bool):
        self.info_html, self.warn = html, warn
        self.update()

    def enterEvent(self, e):
        self._hover = True
        self.update()
        if self.info_html:
            QToolTip.showText(self.mapToGlobal(QPoint(self.width() // 2, self.height() + 6)), self.info_html, self)

    def leaveEvent(self, e):
        self._hover = False
        self.update()
        QToolTip.hideText()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(COLORS["warn"] if self.warn else COLORS["text"] if self._hover else COLORS["muted"])
        p.setPen(QPen(c, 1.2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(2.5, 2.5, 13, 13))
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawEllipse(QRectF(8.25, 5.6, 1.5, 1.5))     # the dot of the i
        p.drawRoundedRect(QRectF(8.3, 8.0, 1.4, 5.0), 0.7, 0.7)
        p.end()


class AnalysisCancelled(Exception):
    """A roll's analysis stopping because a newer one started."""


class WorkerSignals(QObject):
    thumb = Signal(object, object)          # (photo, uint8 array or "ERROR: ...")
    uv = Signal(object, object, object)     # (photo, geometry key, uv grid): click -> raw scan position
    hires = Signal(object, object, object)  # (photo, settings key, full-resolution array): for zooming in
    preview = Signal(object, object, int)   # (photo, uint8 array or "ERROR: ...", request id)
    wb = Signal(object, object)             # (photo, (c, m, y) or "ERROR: ...")
    exported = Signal(object, object)       # (photo, output path or "ERROR: ...")
    sheet = Signal(object, object)          # (group, the contact sheet's files or "ERROR: ...")
    sheet_progress = Signal(object, int, int)   # (group, frames rendered, frames in all)
    roll = Signal(object, object)           # (group, (floors, ceils, outliers) or "ERROR: ...")
    roll_progress = Signal(object, float)   # (group, 0 .. 1 of the roll analysed)
    straighten = Signal(object, object)     # (photo, angle | None | "ERROR: ...")
    level = Signal(object, object, bool)    # (photo, angle | None | "ERROR: ...", part of the automatic conversion?)
    autocrop = Signal(object, object, object)   # (photo, key asked for, (rect, key) | None | "ERROR: ...")
    note = Signal(object, object)           # (photo, a warning about the scan itself, or "")


class Job(QRunnable):
    """Run `fn()` on a pool thread, then call `emit(result)` (which emits a signal)."""

    def __init__(self, fn, emit):
        super().__init__()
        self.fn, self.emit = fn, emit

    def run(self):
        try:
            result = self.fn()
        except Exception as exc:
            traceback.print_exc()
            result = f"ERROR: {exc}"
        self.emit(result)


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------
def to_pixmap(rgb: np.ndarray) -> QPixmap:
    """uint8 RGB numpy array -> QPixmap."""
    data = np.ascontiguousarray(rgb[..., :3])
    h, w = data.shape[:2]
    img = QImage(data.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())   # copy: QImage must not outlive `data`


class FlowLayout(QLayout):
    """Lays widgets out in a row and wraps to the next row when space runs out
    (like words in a paragraph). Used for the toolbar so the window can get narrow."""

    def __init__(self, parent=None, spacing: int = 8):
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        return size

    def _arrange(self, rect, move: bool) -> int:
        """Fill rows left to right, then centre each row (it sits under the centred photo)."""
        rows, row, row_w = [], [], 0
        for it in self._items:
            if not it.widget() or it.widget().isHidden():    # hidden on purpose, not just off-screen
                continue
            w = it.sizeHint().width()
            if row and row_w + self.spacing() + w > rect.width():   # wrap
                rows.append((row, row_w))
                row, row_w = [], 0
            row_w += (self.spacing() if row else 0) + w
            row.append(it)
        if row:
            rows.append((row, row_w))
        y = rect.y()
        for items, width in rows:
            x = rect.x() + max(0, (rect.width() - width) // 2)
            row_h = max(it.sizeHint().height() for it in items)
            for it in items:
                if move:
                    it.setGeometry(QRect(QPoint(x, y), it.sizeHint()))
                x += it.sizeHint().width() + self.spacing()
            y += row_h + self.spacing()
        return max(0, y - self.spacing() - rect.y())


def panel() -> tuple[QFrame, QVBoxLayout]:
    """A section: a darker, borderless box that groups controls."""
    frame = QFrame()
    frame.setObjectName("panel")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(18, 16, 18, 16)
    lay.setSpacing(12)
    return frame, lay


FADE_MS = 200      # hover and selection changes glide instead of snapping


def mix(a: str, b: str, t: float) -> str:
    """The colour a fraction `t` of the way from `a` to `b` (hex strings)."""
    ca, cb = QColor(a), QColor(b)
    f = lambda x, y: round(x + (y - x) * t)
    return QColor(f(ca.red(), cb.red()), f(ca.green(), cb.green()), f(ca.blue(), cb.blue())).name()


class Fade(QVariantAnimation):
    """A value gliding between 0 and 1 in FADE_MS; `on_change(t)` is called on the way."""

    def __init__(self, parent, on_change):
        super().__init__(parent)
        self.t = 0.0
        self.setDuration(FADE_MS)
        self.setEasingCurve(QEasingCurve.OutCubic)
        self._on_change = on_change
        self.valueChanged.connect(self._step)

    def _step(self, v):
        self.t = float(v)
        self._on_change(self.t)

    def to(self, target: float):
        if abs(target - self.t) < 1e-3 and self.state() != QVariantAnimation.Running:
            return
        self.stop()
        self.setStartValue(self.t)
        self.setEndValue(float(target))
        self.start()


class FadeButton(QPushButton):
    """A button whose text glides to its hover colour and back (a stylesheet can only snap).
    While gliding it carries its own in-between colour; at rest the app's stylesheet rules."""

    QUIET = ("toggle", "bar", "tab", "fold", "tool")   # these rest in the muted colour
    TO_TEXT = ("tab", "fold")                     # and these light up to white, not neon

    def __init__(self, text: str = ""):
        super().__init__(text)
        self._fade = Fade(self, self._tint)

    def enterEvent(self, e):
        super().enterEvent(e)
        if self.isEnabled():
            self._fade.to(1.0)

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self._fade.to(0.0)

    def _tint(self, t: float):
        name = self.objectName()
        if t <= 0.001 or t >= 0.999 or self.isChecked() or self.property("on") or name in ("primary", "export"):
            if self.styleSheet():
                self.setStyleSheet("")
            return
        rest = COLORS["muted"] if name in self.QUIET else COLORS["text"]
        lit = COLORS["text"] if name in self.TO_TEXT else ACCENT
        self.setStyleSheet(f"QPushButton:!pressed {{ color: {mix(rest, lit, t)}; }}")


# What the system's file browser is called, for "Show in … when done"
FILE_MANAGER = {"darwin": "Finder", "win32": "Explorer"}.get(sys.platform, "file manager")


# What the system's bin is called, for "Move originals to the … after export"
TRASH_NAME = {"win32": "Recycle Bin"}.get(sys.platform, "Trash")


def move_to_trash(path: Path) -> bool:
    """Move a file to the system's Trash, where it can be put back. False when the system has no
    Trash for it (some network drives, a sandbox): the file is then left where it is, never deleted."""
    from PySide6.QtCore import QFile
    try:
        moved = QFile.moveToTrash(str(path))
    except Exception:
        return False
    moved = moved[0] if isinstance(moved, tuple) else moved
    return bool(moved) and not Path(path).exists()


def show_in_file_manager(path: str):
    """Open `path` (a folder, or the folder a file is in) in the system's file browser.

    macOS and Windows open a folder URL in Finder / Explorer. On Linux Qt hands it to xdg-open, which
    a Flatpak sandbox does not have: there the desktop portal is asked to open the folder."""
    import subprocess
    if sys.platform.startswith("linux"):
        if Path("/.flatpak-info").exists():
            try:
                from PySide6.QtDBus import QDBusConnection, QDBusMessage, QDBusUnixFileDescriptor
                fd = os.open(path, os.O_RDONLY)
                try:
                    msg = QDBusMessage.createMethodCall("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop",
                                                        "org.freedesktop.portal.OpenURI", "OpenDirectory")
                    msg.setArguments(["", QDBusUnixFileDescriptor(fd), {}])
                    reply = QDBusConnection.sessionBus().call(msg)
                    if reply.type() != QDBusMessage.ErrorMessage:
                        return
                    print("Show in file manager: the portal said", reply.errorMessage(), file=sys.stderr)
                finally:
                    os.close(fd)
            except Exception as exc:                  # no D-Bus: try the usual way below
                print("Show in file manager:", exc, file=sys.stderr)
        else:
            folder = path if os.path.isdir(path) else os.path.dirname(path)
            try:
                subprocess.Popen(["xdg-open", folder], start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                pass
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def process_alive(pid: int) -> bool:
    """Is a process with this id running? (Only asks: on Windows os.kill would end it.)"""
    if sys.platform == "win32":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259                               # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def button(text: str, slot=None, name: str = "") -> QPushButton:
    b = FadeButton(text)
    if name:
        b.setObjectName(name)
    if slot:
        # clicked sends a "checked" flag; a slot with an optional argument (toggle_crop's `apply`)
        # would take it as that argument, so the flag is dropped here
        b.clicked.connect(lambda _checked=False: slot())
    return b


# Preferences: the few choices that outlive a session (the Settings tab). Stored by macOS in
# ~/Library/Preferences, next to the key bindings and the camera / lens presets.
PREF_DEFAULTS = {"gpu": True, "border": True, "border_px": 80, "sync": "", "auto_crop_new": False, "scan_warnings": True, "flat_field": "", "light": "white", "export": "", "sheet": "", "accent": "purple"}


def pref(key: str):
    v = QSettings("SimpleNeg", "SimpleNeg").value("pref/" + key, PREF_DEFAULTS[key])
    if isinstance(PREF_DEFAULTS[key], bool):       # stored as text: "true" / "false"
        return v if isinstance(v, bool) else str(v).lower() == "true"
    if isinstance(PREF_DEFAULTS[key], int):
        try:
            return int(v)
        except (TypeError, ValueError):
            return PREF_DEFAULTS[key]
    return v or ""


def set_pref(key: str, value):
    QSettings("SimpleNeg", "SimpleNeg").setValue("pref/" + key, value)


class GearButton(FadeButton):
    """The Settings tab: a drawn gear instead of a word."""

    def __init__(self):
        super().__init__("")
        self._t = 0.0
        self.setFixedWidth(40)
        self.setToolTip("Settings")

    def _tint(self, t: float):
        self._t = t
        self.update()

    def paintEvent(self, e):
        import math
        super().paintEvent(e)                     # the tab's plate and underline, from the stylesheet
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = COLORS["text"] if self.isChecked() else mix(COLORS["muted"], COLORS["text"], self._t)
        p.setPen(QPen(QColor(color), 1.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.translate(self.width() / 2, self.height() / 2 - 0.5)
        ring = QPainterPath()
        for i in range(8):                        # eight teeth
            a = i * 45.0
            for da, r in ((-13, 5.0), (-8, 6.9), (8, 6.9), (13, 5.0)):
                x, y = r * math.cos(math.radians(a + da)), r * math.sin(math.radians(a + da))
                ring.lineTo(x, y) if (i or da != -13) else ring.moveTo(x, y)
        ring.closeSubpath()
        p.drawPath(ring)
        p.drawEllipse(QPointF(0, 0), 2.3, 2.3)
        p.end()


class Chevron(QToolButton):
    """The one fold mark: a thin chevron that points right when closed and turns down when open."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(22, 22)
        self.setCursor(Qt.PointingHandCursor)
        self._angle = 0.0
        self.on_light = False                     # sits on a white plate: drawn in black
        self._shown = False                       # the state the chevron is drawn (or turning) for
        self._anim = QPropertyAnimation(self, b"angle", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def _get_angle(self) -> float:
        return self._angle

    def _set_angle(self, a: float):
        self._angle = a
        self.update()

    angle = Property(float, _get_angle, _set_angle)

    def paintEvent(self, e):
        if self.isChecked() != self._shown:       # also when checked from code, without a signal
            self._shown = self.isChecked()
            self._anim.stop()
            self._anim.setStartValue(self._angle)
            self._anim.setEndValue(90.0 if self._shown else 0.0)
            self._anim.start()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = (COLORS["on_select"] if self.on_light else
                 COLORS["text"] if self.underMouse() or self.isChecked() else COLORS["muted"])
        p.setPen(QPen(QColor(color), 1.3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.translate(self.width() / 2, self.height() / 2)
        p.rotate(self._angle)
        p.drawPolyline([QPointF(-1.8, -3.5), QPointF(1.8, 0), QPointF(-1.8, 3.5)])
        p.end()


def _icon_rotate(p: QPainter, clockwise: bool):
    """An open circle ending in a solid arrowhead (drawn in a 20 x 20 box). The line stops at
    the arrowhead's base, so the two never cross."""
    import math
    if clockwise:                                 # the same drawing, mirrored
        p.translate(20, 0)
        p.scale(-1, 1)
    cx, cy, r = 10.0, 10.3, 5.8
    start, end = 205.0, 462.0                     # degrees, counter-clockwise: a gap at the upper left
    path = QPainterPath()
    path.arcMoveTo(cx - r, cy - r, 2 * r, 2 * r, start)
    path.arcTo(cx - r, cy - r, 2 * r, 2 * r, start, end - start)
    p.drawPath(path)
    a = math.radians(end)
    base = QPointF(cx + r * math.cos(a), cy - r * math.sin(a))     # where the line ends
    tx, ty = -math.sin(a), -math.cos(a)                             # direction of travel there
    head = QPainterPath()
    head.moveTo(base.x() + 4.4 * tx, base.y() + 4.4 * ty)           # the tip
    head.lineTo(base.x() + 2.7 * ty, base.y() - 2.7 * tx)
    head.lineTo(base.x() - 2.7 * ty, base.y() + 2.7 * tx)
    head.closeSubpath()
    pen = p.pen()
    p.setBrush(pen.color())
    p.setPen(QPen(pen.color(), 0.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPath(head)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)


def _icon_mirror(p: QPainter, _=None):
    """Two triangles facing each other across a centre line."""
    p.drawLine(QPointF(10, 3.5), QPointF(10, 16.5))
    p.drawPolygon([QPointF(7.2, 6), QPointF(7.2, 14), QPointF(2.5, 10)])
    p.drawPolygon([QPointF(12.8, 6), QPointF(12.8, 14), QPointF(17.5, 10)])


class IconButton(FadeButton):
    """A flat tile with a thin line icon (drawn, so it is sharp at any scale); neon on hover."""

    def __init__(self, draw, arg=None, slot=None, tip: str = ""):
        super().__init__()
        self._draw, self._arg = draw, arg
        self._t = 0.0
        self.setObjectName("tile")
        self.setMinimumHeight(30)
        self.setToolTip(tip)
        self.rest_color: str | None = None        # set by a parent that changes colour (e.g. a white plate)
        self.scale = 0.9                          # the 20 x 20 drawing's size on the button
        if slot:
            self.clicked.connect(lambda _checked=False: slot())

    def paintEvent(self, e):
        super().paintEvent(e)                     # the tile's background, from the stylesheet
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rest = self.rest_color or (COLORS["muted"] if self.objectName() in ("bar", "tool") else COLORS["soft_solid"])
        if self.property("on"):                   # a mode it started is on (Pick grey waiting for a click)
            rest = ACCENT
        color = (COLORS["faint"] if not self.isEnabled() else COLORS["text"] if self.isDown() else
                 mix(rest, ACCENT, self._t))
        p.setPen(QPen(QColor(color), 1.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.translate(self.width() / 2, self.height() / 2)
        p.scale(self.scale, self.scale)
        p.translate(-10, -10)
        self._draw(p, self._arg)
        p.end()


def _icon_wand(p: QPainter, _arg=None):
    """Auto: a wand with a small spark."""
    p.drawLine(QPointF(4.5, 15.5), QPointF(12.0, 8.0))
    p.drawLine(QPointF(13.5, 3.5), QPointF(13.5, 6.5))
    p.drawLine(QPointF(12.0, 5.0), QPointF(15.0, 5.0))
    p.drawLine(QPointF(16.5, 8.5), QPointF(16.5, 10.5))
    p.drawLine(QPointF(15.5, 9.5), QPointF(17.5, 9.5))


def _icon_dropper(p: QPainter, _arg=None):
    """Pick: an eyedropper."""
    tube = QPainterPath()
    tube.moveTo(12.0, 6.0)
    tube.lineTo(14.0, 8.0)
    tube.lineTo(7.0, 15.0)
    tube.lineTo(4.5, 15.5)
    tube.lineTo(5.0, 13.0)
    tube.closeSubpath()
    p.drawPath(tube)
    p.drawEllipse(QPointF(14.6, 5.4), 1.9, 1.9)
    p.drawLine(QPointF(11.0, 5.0), QPointF(15.0, 9.0))


def _icon_copy(p: QPainter, _arg=None):
    p.drawRoundedRect(QRectF(4.5, 6.5, 8.5, 9.5), 1.6, 1.6)
    p.drawPolyline([QPointF(7.5, 6.5), QPointF(7.5, 4.0), QPointF(15.5, 4.0), QPointF(15.5, 13.0), QPointF(13.0, 13.0)])


def _icon_paste(p: QPainter, _arg=None):
    p.drawRoundedRect(QRectF(5.0, 5.0, 10.0, 11.5), 1.6, 1.6)
    p.drawRoundedRect(QRectF(7.5, 3.5, 5.0, 3.0), 1.0, 1.0)
    p.drawLine(QPointF(7.5, 10.5), QPointF(12.5, 10.5))
    p.drawLine(QPointF(7.5, 13.0), QPointF(11.0, 13.0))


def _icon_sync(p: QPainter, _arg=None):
    """Two arrows chasing round a circle."""
    path = QPainterPath()
    path.arcMoveTo(4.5, 4.5, 11, 11, 20)
    path.arcTo(4.5, 4.5, 11, 11, 20, 140)
    path.arcMoveTo(4.5, 4.5, 11, 11, 200)
    path.arcTo(4.5, 4.5, 11, 11, 200, 140)
    p.drawPath(path)
    p.drawPolyline([QPointF(5.6, 5.4), QPointF(5.4, 8.4), QPointF(8.3, 8.0)])
    p.drawPolyline([QPointF(14.4, 14.6), QPointF(14.6, 11.6), QPointF(11.7, 12.0)])


def _icon_heal(p: QPainter, _arg=None):
    """A spot: a ring around a dot."""
    p.drawEllipse(QPointF(10, 10), 5.5, 5.5)
    p.drawEllipse(QPointF(10, 10), 1.0, 1.0)


def _icon_area(p: QPainter, _arg=None):
    """A lasso: a dashed loop."""
    pen = p.pen()
    pen.setDashPattern([2.0, 1.6])
    p.setPen(pen)
    p.drawEllipse(QRectF(3.5, 5.0, 13.0, 9.0))


def _icon_reset(p: QPainter, _arg=None):
    """Back to the start: a hooked arrow (↩), so it never reads as a turn."""
    path = QPainterPath()
    path.moveTo(5.5, 8.0)
    path.lineTo(12.0, 8.0)
    path.arcTo(QRectF(8.5, 8.0, 7.0, 7.0), 90.0, -180.0)
    path.lineTo(8.5, 15.0)
    p.drawPath(path)
    p.drawPolyline([QPointF(8.5, 5.0), QPointF(5.5, 8.0), QPointF(8.5, 11.0)])


def _icon_autocrop(p: QPainter, _arg=None):
    """Find the picture edges again: four crop corners around a small sparkle."""
    for x, y, dx, dy in ((3.5, 3.5, 1, 1), (16.5, 3.5, -1, 1), (3.5, 16.5, 1, -1), (16.5, 16.5, -1, -1)):
        p.drawPolyline([QPointF(x, y + 4.0 * dy), QPointF(x, y), QPointF(x + 4.0 * dx, y)])
    star = QPainterPath()
    star.moveTo(10.0, 6.3)
    for px, py in ((10.9, 9.1), (13.7, 10.0), (10.9, 10.9), (10.0, 13.7), (9.1, 10.9), (6.3, 10.0), (9.1, 9.1)):
        star.lineTo(px, py)
    star.closeSubpath()
    pen = p.pen()
    p.setBrush(pen.color())
    p.setPen(QPen(pen.color(), 0.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPath(star)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)


def _icon_more(p: QPainter, _arg=None):
    """Options: a small down chevron."""
    p.drawPolyline([QPointF(6.0, 8.5), QPointF(10.0, 12.5), QPointF(14.0, 8.5)])


def _icon_orientation(p: QPainter, _arg=None):
    """Turn the crop: an upright frame and a lying one, a small arc between them."""
    p.drawRoundedRect(QRectF(3.5, 4.0, 6.0, 9.0), 1.0, 1.0)              # portrait
    p.drawRoundedRect(QRectF(8.5, 10.0, 9.0, 6.0), 1.0, 1.0)             # landscape
    arc = QPainterPath()
    arc.moveTo(12.0, 4.5)
    arc.quadTo(16.0, 4.5, 16.0, 8.0)
    p.drawPath(arc)
    p.drawPolyline([QPointF(14.6, 6.9), QPointF(16.0, 8.4), QPointF(17.4, 6.9)])


def _icon_negative(p: QPainter, _arg=None):
    """The original scan: a film frame with its two sprocket rows."""
    p.drawRoundedRect(QRectF(3.5, 5.0, 13.0, 10.0), 1.2, 1.2)
    for x in (6.0, 9.0, 12.0, 15.0):
        p.drawPoint(QPointF(x - 0.5, 6.8))
        p.drawPoint(QPointF(x - 0.5, 13.2))
    p.drawRect(QRectF(6.0, 8.5, 8.0, 3.0))


def tool_plate(groups: list[list[QWidget]]) -> QFrame:
    """A toolbar: one plate, its buttons flat on it, groups split by a thin divider."""
    plate = QFrame()
    plate.setObjectName("toolPlate")
    row = QHBoxLayout(plate)
    row.setContentsMargins(3, 3, 3, 3)
    row.setSpacing(1)
    for i, group in enumerate(groups):
        if i:
            div = QFrame()
            div.setObjectName("toolDiv")
            div.setFixedSize(1, 18)
            row.addSpacing(4)
            row.addWidget(div, 0, Qt.AlignVCenter)
            row.addSpacing(4)
        for w in group:
            row.addWidget(w)
    return plate


def line_icon(draw, arg=None, size: int = 14) -> QIcon:
    """A drawn line icon as a QIcon (sharp on Retina), in the bar's text colour; dimmed when disabled."""
    icon = QIcon()
    for mode, colour in ((QIcon.Normal, COLORS["muted"]), (QIcon.Disabled, COLORS["faint"])):
        pix = QPixmap(size * 2, size * 2)
        pix.setDevicePixelRatio(2.0)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(colour), 1.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.scale(size / 20, size / 20)
        draw(p, arg)
        p.end()
        icon.addPixmap(pix, mode)
    return icon


def _icon_gear(p: QPainter, _arg=None):
    """A small cog (drawn in a 20 x 20 box): eight square teeth, one straight up, a hole in the middle."""
    import math
    outline = QPainterPath()
    r_in, r_out = 4.0, 5.6
    for k in range(8):
        c = k * 45.0 - 90.0                       # the first tooth points straight up
        for i, (da, r) in enumerate(((-17.0, r_in), (-10.0, r_out), (10.0, r_out), (17.0, r_in))):
            a = math.radians(c + da)
            pt = QPointF(10 + r * math.cos(a), 10 + r * math.sin(a))
            if k == 0 and i == 0:
                outline.moveTo(pt)
            else:
                outline.lineTo(pt)
    outline.closeSubpath()
    p.drawPath(outline)
    p.drawEllipse(QPointF(10, 10), 1.6, 1.6)


class Toast(QLabel):
    """A short notice on top of the photo ("Updating 3 photos") that fades away by itself: a dark
    pill with a warning sign, so it reads as a notice and not as a button."""

    H, PAD, ICON = 34, 14, 15

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        f = QFont()
        f.setPixelSize(12)
        f.setWeight(QFont.DemiBold)
        self.setFont(f)
        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self._fade = QPropertyAnimation(self._fx, b"opacity", self)
        self._fade.setDuration(400)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.hide)
        self._stay = QTimer(self, singleShot=True, interval=1400)
        self._stay.timeout.connect(self._fade.start)
        self.hide()

    def show_text(self, text: str):
        self.setText(text)
        w = self.fontMetrics().horizontalAdvance(text) + 2 * self.PAD + self.ICON + 10
        self.setFixedSize(w, self.H)
        self.move((self.parentWidget().width() - self.width()) // 2, 14)
        self._fade.stop()
        self._fx.setOpacity(1.0)
        self.show()
        self.raise_()
        self._stay.start()                         # a new notice keeps it up a little longer

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        box = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        edge = QColor(COLORS["warn"])
        edge.setAlpha(110)
        p.setPen(QPen(edge, 1))
        p.setBrush(QColor(16, 17, 19, 235))
        p.drawRoundedRect(box, self.H / 2, self.H / 2)
        # the warning sign: a rounded triangle with an exclamation mark
        x0, cy, s = self.PAD, self.H / 2, self.ICON
        tri = QPainterPath()
        tri.moveTo(x0 + s / 2, cy - s / 2 + 0.5)
        tri.lineTo(x0 + s, cy + s / 2 - 1)
        tri.lineTo(x0, cy + s / 2 - 1)
        tri.closeSubpath()
        warn = QColor(COLORS["warn"])
        p.setPen(QPen(warn, 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        p.drawPath(tri)
        p.setPen(QPen(warn, 1.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(x0 + s / 2, cy - 2.2), QPointF(x0 + s / 2, cy + 1.8))
        p.drawPoint(QPointF(x0 + s / 2, cy + 4.6))
        p.setPen(QColor(COLORS["text"]))
        p.setFont(self.font())
        p.drawText(QRectF(x0 + s + 10, 0, self.width() - x0 - s - 10 - self.PAD, self.H),
                   Qt.AlignVCenter | Qt.AlignLeft, self.text())
        p.end()


# What Sync Roll can send to the other frames, by group: key -> (label, settings)
SYNC_GROUPS = {
    "exposure": ("Exposure, density and contrast", ("exposure", "density", "contrast")),
    "saturation": ("Saturation", ("saturation",)),
    "colour": ("Colour (C / M / Y, temp, tint)", ("cyan", "magenta", "yellow", "temp", "tint")),
    "tone": ("Highlights and shadows", ("highlight_tone", "shadow_tone")),
    "refine": ("Sharpen, colour noise, glow", ("sharpen", "color_noise", "glow")),
    "paper": ("Print look", ("paper",)),
    "points": ("Whites and blacks", ("white_point", "black_point")),
    "mixer": ("Colour mixer", ("hsl_hue", "hsl_saturation", "hsl_luminance")),
    "framing": ("Crop and rotation", ("rotation", "mirror", "fine_rotation", "crop", "auto_crop")),
}
SYNC_DEFAULT = ["exposure", "saturation", "colour", "tone", "refine", "points", "mixer", "paper"]    # framing differs frame to frame


class LineSlider(QWidget):
    """One control as a thin horizontal line: its name, the line with a small marker, the value.

    A two-sided control (soft ↔ hard, cool ↔ warm) has its zero in the middle and can tint each
    side; an amount (sharpen, glow) starts at the left. Drag or click the line; double-click resets.
    """

    changed = Signal(float)
    LABEL_W, VALUE_W = 92, 36

    def __init__(self, title: str, lo: float, hi: float, default: float = 0.0, step: float = 0.05,
                 unipolar: bool = False, minus: str = "", plus: str = ""):
        super().__init__()
        self.title, self.lo, self.hi, self.default, self.step = title, lo, hi, default, step
        self.unipolar = unipolar
        self.origin = lo if unipolar else 0.0          # where the filled part starts
        self.minus_color, self.plus_color = minus or COLORS["text"], plus or COLORS["text"]
        self.gradient: list | None = None              # a colour track, left to right (the colour mixer)
        self._value = default
        self._drag = False
        self._hover = False
        self.setFixedHeight(30)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def value(self) -> float:
        return self._value

    def set_value(self, v: float):
        self._set(v, emit=False)

    def reset(self):
        self._set(self.default, emit=True)

    def _set(self, v: float, emit: bool = False):
        # whole steps away from the default (which need not be a round number: noise starts at 0.33)
        v = self.default + round((min(max(v, self.lo), self.hi) - self.default) / self.step) * self.step
        v = round(min(max(v, self.lo), self.hi), 2)
        if v == self._value and emit:
            return
        self._value = v
        self.update()
        if emit:
            self.changed.emit(v)

    def _track(self) -> tuple[float, float]:
        return self.LABEL_W + 6.0, self.width() - self.VALUE_W - 8.0

    def _x(self, v: float) -> float:
        x0, x1 = self._track()
        return x0 + (v - self.lo) / (self.hi - self.lo) * (x1 - x0)

    def paintEvent(self, e):
        """Lightroom-like: a thin track with ticks above it, a fill from zero, a small pointer on the track."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h = self.height()
        ty = h / 2 + 2                                  # the track sits a little low: the ticks go above it
        f = p.font()
        f.setPixelSize(12)
        p.setFont(f)
        moved = abs(self._value - self.default) > 1e-6
        lit = self._drag or self._hover
        p.setPen(QColor(COLORS["text"] if moved or lit else COLORS["muted"]))
        p.drawText(QRectF(0, 0, self.LABEL_W, h), Qt.AlignVCenter | Qt.AlignLeft, self.title)
        x0, x1 = self._track()
        xo, xv = self._x(self.origin), self._x(self._value)
        c = self.plus_color if self._value >= self.origin else self.minus_color
        paint_slider(p, x0, x1, ty, xo, xv, 8,
                     QColor(ACCENT if lit else COLORS["pointer_on"] if moved else COLORS["pointer"]),
                     QColor(COLORS["fill"] if c == COLORS["text"] else c), gradient=self.gradient, lit=lit)
        # the value: "+ 20", "– 44", "0"
        if self.unipolar:
            n = round(100 * (self._value - self.lo) / (self.hi - self.lo))
            text = f"{n}"
        else:
            n = round((self._value - self.origin) / self.step)
            text = f"+ {n}" if n > 0 else f"– {-n}" if n < 0 else "0"
        p.setPen(QColor(COLORS["text"] if moved else COLORS["muted"]))
        p.drawText(QRectF(self.width() - self.VALUE_W - 6, 0, self.VALUE_W + 6, h), Qt.AlignVCenter | Qt.AlignRight, text)
        p.end()

    def _value_at(self, x: float) -> float:
        x0, x1 = self._track()
        return self.lo + (min(max(x, x0), x1) - x0) / (x1 - x0) * (self.hi - self.lo)

    def mousePressEvent(self, e):
        """Only the line takes a click: the name and the number are labels (double-click still resets)."""
        x0, x1 = self._track()
        if e.button() == Qt.LeftButton and x0 - 8 <= e.position().x() <= x1 + 8:
            self._drag = True
            self._set(self._value_at(e.position().x()), emit=True)
            self.update()

    def mouseMoveEvent(self, e):
        if self._drag:
            self._set(self._value_at(e.position().x()), emit=True)

    def mouseReleaseEvent(self, e):
        self._drag = False
        self.update()

    def mouseDoubleClickEvent(self, e):
        self._drag = False
        self.reset()


def _icon_tint(self, t: float):
    self._t = t
    self.update()


IconButton._tint = _icon_tint


class ThumbDelegate(QStyledItemDelegate):
    """Draws a filmstrip frame: just the picture. A selected frame is a little larger with a
    thin neon border, and never tinted (Qt's own selection washes the picture with colour).
    The frame under the mouse lights up. Both changes glide (FADE_MS)."""

    GROW = 1.1
    DIM = 0.80            # the frames around the chosen one step back a little

    def __init__(self, view: QListWidget):
        super().__init__(view)
        self.view = view
        self._sel: dict[int, float] = {}          # row -> how far "selected" has come in (0..1)
        self._hot: dict[int, float] = {}          # row -> the same for "under the mouse"
        self._hover_row = -1
        self.edited = lambda row: False           # set by the window: was this frame adjusted?
        self._timer = QTimer(self, interval=16)
        self._timer.timeout.connect(self._tick)
        view.setMouseTracking(True)
        view.viewport().installEventFilter(self)
        view.selectionModel().selectionChanged.connect(lambda *_: self._timer.start())
        view.model().modelReset.connect(self._forget)
        view.model().rowsRemoved.connect(self._forget)

    def _forget(self, *_):
        self._sel.clear()
        self._hot.clear()
        self._hover_row = -1

    def eventFilter(self, obj, e):
        t = e.type()
        if t == QEvent.MouseMove:
            row = self.view.indexAt(e.position().toPoint()).row()
        elif t in (QEvent.Leave, QEvent.HoverLeave):
            row = -1
        else:
            return False
        if row != self._hover_row:
            self._hover_row = row
            self._timer.start()
        return False

    def _tick(self):
        step, moving = self._timer.interval() / FADE_MS, False
        selected = {i.row() for i in self.view.selectedIndexes()}
        for row in range(self.view.count()):
            for state, target in ((self._sel, 1.0 if row in selected else 0.0),
                                  (self._hot, 1.0 if row == self._hover_row else 0.0)):
                v = state.get(row, 0.0)
                if v != target:
                    state[row] = min(target, v + step) if target > v else max(target, v - step)
                    moving = True
        self.view.viewport().update()
        if not moving:
            self._timer.stop()

    def paint(self, p: QPainter, option, index):
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(option.rect)
        ease = lambda t: 1 - (1 - t) ** 3
        sel = ease(self._sel.get(index.row(), 0.0))
        hot = ease(self._hot.get(index.row(), 0.0))
        icon = index.data(Qt.DecorationRole)
        text = index.data(Qt.DisplayRole)
        if text:                                  # a file that could not be read: its name, no picture
            p.setPen(QColor(COLORS["warn"]))
            f = p.font()
            f.setPixelSize(10)
            p.setFont(f)
            p.drawText(r.adjusted(4, 0, -4, 0), Qt.AlignCenter | Qt.TextWrapAnywhere, text)
        elif isinstance(icon, QIcon) and not icon.isNull():
            pix = icon.pixmap(QSize(THUMB_SIZE, THUMB_H), QIcon.Normal, QIcon.Off)
            size = pix.deviceIndependentSize()
            k = 1.0 + (self.GROW - 1.0) * sel
            w, h = size.width() * k, size.height() * k
            target = QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h)
            p.setOpacity(self.DIM + (1.0 - self.DIM) * max(sel, hot))
            p.drawPixmap(target, pix, QRectF(pix.rect()))
            p.setOpacity(1.0)
            p.setBrush(Qt.NoBrush)
            edge = target.adjusted(-0.75, -0.75, 0.75, 0.75)
            if hot > 0.01 and sel < 0.99:         # under the mouse: a faint white edge
                c = QColor(COLORS["text"])
                c.setAlphaF(0.45 * hot * (1.0 - sel))
                p.setPen(QPen(c, 1.0))
                p.drawRect(edge)
            if sel > 0.01:
                c = QColor(ACCENT)
                c.setAlphaF(sel)
                p.setPen(QPen(c, 1.5))
                p.drawRect(edge)
            if self.edited(index.row()):          # adjusted by hand: a small accent mark under its corner
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(ACCENT2))
                p.drawRect(QRectF(target.right() - 10, target.bottom() + 3, 10, 2))
        p.restore()

    def sizeHint(self, option, index):
        return QSize(THUMB_SIZE + 22, THUMB_H + 18)


# Fader colours: the top (+) and bottom (−) end of each control
FADER_COLORS = {"plus": "#c4c4c8", "minus": "#6b6d73",
                "dens+": "#7e8086", "dens-": "#c4c4c8", "con+": "#c8c8cc", "con-": "#7e8086",
                "sat+": "#d99a4e", "sat-": "#8a8c92",
                "warm+": "#d9a04e", "cool-": "#5b86c9", "mag+": "#c45a8e", "grn-": "#4f9a62",
                "fx+": "#c4c4c8", "fx-": "#55575d",
                # muted like the colour mixer's dots: a hint of the direction, not a signal
                "plusC": "#4aa3c4", "minusR": "#cf5a52",
                "plusM": "#c45a8e", "minusG": "#4f9a62",
                "plusY": "#d4bd4c", "minusB": "#5a78cf"}


class FaderTrack(QWidget):
    """The painted part of a fader: a track from the + colour (top) to the − colour (bottom),
    a zero line in the middle, a bar growing from zero, and a knob at the current value.

    Drag the knob, scroll, or double-click to reset. It shows ±SPAN steps; beyond that the
    knob waits at the end and a small arrow shows the value goes further.
    """

    SPAN = 20                                   # steps from the middle to each end

    def __init__(self, fader: "Stepper", top: str, bottom: str, unipolar: bool = False,
                 span: int = 20):
        super().__init__()
        self.SPAN = span                 # steps from the zero line to the end of the track
        self.fader, self.top, self.bottom = fader, QColor(top), QColor(bottom)
        self.unipolar = unipolar       # zero at the bottom: for controls that only go up (Glow)
        self.active = False
        self._hover = self._drag = False
        self.setFixedSize(42, 116)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip("Drag, scroll, or double-click to reset")

    def _geometry(self):
        pad = 8
        h = self.height() - 2 * pad
        if self.unipolar:
            return pad, h, pad + h, h / self.SPAN               # zero sits at the bottom
        return pad, h, self.height() / 2, h / 2 / self.SPAN    # top, height, centre y, px per step

    def paintEvent(self, e):
        """The main controls, like a mixing desk's faders in the Refine sliders' quiet style: a recessed
        groove with the track inside it, tinted from the − colour (bottom) to the + colour (top), ticks
        on both sides, a fill from zero and a small raised cap. Colour comes up when the fader is touched
        (hover, drag or its key held)."""
        pad, h, cy, px = self._geometry()
        cx = round(self.width() / 2) + 0.5
        steps = self.fader.steps()
        lit = self._hover or self._drag or self.active
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        shown = max(0 if self.unipolar else -self.SPAN, min(self.SPAN, steps))
        y = cy - shown * px
        # ticks every 5 steps on both sides, a longer one at zero
        for k in range(0 if self.unipolar else -self.SPAN, self.SPAN + 1, 5):
            ty = round(cy - k * px - 0.5) + 0.5
            long_ = k == 0
            p.setPen(QPen(QColor(COLORS["text"] if long_ and lit else COLORS["tick"]), 1))
            for side in (-1, 1):
                p.drawLine(QPointF(cx + side * 6.0, ty), QPointF(cx + side * (12.0 if long_ else 9.0), ty))
        # the groove, recessed into the panel
        groove = QRectF(cx - 2.5, pad - 2, 5, h + 4)
        p.setPen(QPen(QColor(COLORS["edge_lo"]), 1))
        p.setBrush(QColor(COLORS["field"]))
        p.drawRoundedRect(groove, 2.5, 2.5)
        # the track inside it: the two directions' colours meeting in the track colour at zero
        grad = QLinearGradient(0, pad, 0, pad + h)
        top, bottom = QColor(self.top), QColor(self.bottom)
        for c in (top, bottom):
            c.setAlphaF(0.95 if lit else 0.55)
        zero = (cy - pad) / h
        grad.setColorAt(0, top)
        grad.setColorAt(min(max(zero, 0.0), 1.0), QColor(COLORS["track"]))
        grad.setColorAt(1, bottom)
        p.setPen(QPen(QBrush(grad), 1.5, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cx, pad), QPointF(cx, pad + h))
        if steps:                                   # from zero to the value, in its direction's colour
            p.setPen(QPen(QColor(self.top if steps > 0 else self.bottom), 3, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cx, cy), QPointF(cx, y))
        # the cap: small and raised, a line across it that lights up
        cap = QRectF(cx - 10, y - 4.5, 20, 9)
        g = QLinearGradient(0, cap.top(), 0, cap.bottom())
        g.setColorAt(0, QColor(COLORS["button_hover_top"] if lit else COLORS["button_top"]))
        g.setColorAt(1, QColor(COLORS["button"]))
        p.setPen(QPen(QColor(COLORS["edge_lo"]), 1))
        p.setBrush(QBrush(g))
        p.drawRoundedRect(cap, 2.5, 2.5)
        p.setPen(QPen(QColor(COLORS["edge_hi"]), 1))
        p.drawLine(QPointF(cap.left() + 2.5, cap.top() + 1.0), QPointF(cap.right() - 2.5, cap.top() + 1.0))
        p.setPen(QPen(QColor(ACCENT if lit else COLORS["text"] if steps else COLORS["muted"]), 1.4, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cx - 5.5, y), QPointF(cx + 5.5, y))
        if abs(steps) > self.SPAN:                  # goes further than the track shows
            d = -1 if steps > 0 else 1
            p.setPen(QPen(QColor(COLORS["text"]), 1.2, Qt.SolidLine, Qt.RoundCap))
            p.drawPolyline([QPointF(cx - 3.5, y + d * 8), QPointF(cx, y + d * 11.5), QPointF(cx + 3.5, y + d * 8)])
        p.end()

    def _steps_at(self, y: float) -> int:
        _, _, cy, px = self._geometry()
        return round((cy - y) / px)

    def mousePressEvent(self, e):
        self._drag = True
        self.fader.set_steps(self._steps_at(e.position().y()))
        self.update()

    def mouseReleaseEvent(self, e):
        self._drag = False
        self.update()

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.fader.set_steps(self._steps_at(e.position().y()))

    def mouseDoubleClickEvent(self, e):
        self.fader._set(self.fader.default, emit=True)

    def wheelEvent(self, e):
        d = 1 if e.angleDelta().y() > 0 else -1
        self.fader.set_steps(self.fader.steps() + d)


class Stepper(QWidget):
    """One control as a vertical fader: ▲ (the + colour), the track, ▼ (the − colour), the value.

    The value is shown in steps from the automatic conversion (+3 = three clicks up),
    the same numbers as the filmstrip. Hold ▲ / ▼ to repeat; double-click to reset.
    """

    changed = Signal(float)

    def __init__(self, title: str, plus_style: str, minus_style: str,
                 step: float = 0.05, lo: float = 0.0, hi: float = 3.0, default: float = 1.0,
                 unipolar: bool = False, span: int = 20):
        super().__init__()
        self.step, self.lo, self.hi, self.default = step, lo, hi, default
        self.unipolar = unipolar
        self.origin = lo if unipolar else default      # where the track's zero sits
        self._value = default
        top, bottom = FADER_COLORS[plus_style], FADER_COLORS[minus_style]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("faderTitle")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        self.plus = self._arrow("▲", top, +1)
        self.minus = self._arrow("▼", bottom, -1)
        self.track = FaderTrack(self, top, bottom, unipolar, span)
        self.value_lbl = QLabel()
        self.value_lbl.setObjectName("faderValue")
        self.value_lbl.setAlignment(Qt.AlignCenter)
        self.value_lbl.mouseDoubleClickEvent = lambda e: self._set(self.default, emit=True)
        for w in (self.title_lbl, self.plus, self.track, self.minus, self.value_lbl):
            lay.addWidget(w, 0, Qt.AlignHCenter)
        self._set(default)

    def _arrow(self, text: str, color: str, direction: int) -> QToolButton:
        b = QToolButton()
        b.setText(text)
        b.setFixedSize(30, 20)
        b.setStyleSheet(f"QToolButton {{ color: {COLORS['faint']}; background: transparent; border: none; font-size: 8px; }}"
                        f"QToolButton:hover {{ color: {color}; }}"          # its colour only when you reach for it
                        f"QToolButton:pressed {{ color: {COLORS['text']}; }}")
        b.setAutoRepeat(True)
        b.setAutoRepeatDelay(350)
        b.setAutoRepeatInterval(90)
        b.clicked.connect(lambda: self._set(self._value + direction * self.step, emit=True))
        return b

    def value(self) -> float:
        return self._value

    def set_value(self, v: float):
        self._set(v, emit=False)

    def steps(self) -> int:
        return round((self._value - self.origin) / self.step)

    def set_steps(self, n: int):
        if n != self.steps():
            self._set(self.origin + n * self.step, emit=True)

    def set_active(self, on: bool):
        """Highlighted while its key is held (keyboard adjusting)."""
        self.title_lbl.setStyleSheet(f"color: {ACCENT};" if on else "")
        self.track.active = on
        self.track.update()

    def _set(self, v: float, emit: bool = False):
        self._value = round(min(max(v, self.lo), self.hi), 2)
        n = self.steps()
        if self.unipolar:                        # an amount: 0 .. 100
            self.value_lbl.setText(f"{round(100 * (self._value - self.lo) / (self.hi - self.lo))}")
        else:                                    # steps from the automatic result
            self.value_lbl.setText(f"{n:+d}" if n else "0")
        self.value_lbl.setToolTip(f"{self._value:.2f}  ·  double-click to reset")
        edited = abs(self._value - (self.lo if self.unipolar else self.default)) > 1e-6
        if self.value_lbl.property("edited") != edited:
            self.value_lbl.setProperty("edited", edited)
            self.value_lbl.style().unpolish(self.value_lbl)
            self.value_lbl.style().polish(self.value_lbl)
        self.track.update()
        if emit:
            self.changed.emit(self._value)


class Backdrop(QWidget):
    """The window background: a quiet grid (dots or lines), painted from one repeated tile.

    The tile is drawn at the screen's real resolution (2x on Retina) and each dot fills
    whole device pixels with no smoothing: a smoothed dot that straddles pixels turns
    into a soft blob that looks shaded.
    """

    PATTERN = "mat"   # "dots", "lines", or "mat": a fine grid of thin lines, a stronger one every MAJOR cells
    SPACING = 8       # px between dots / lines
    MAJOR = 4         # mat: a stronger line every 4 cells, like a cutting mat
    DOT = 0.5         # dot size or line width in px (0.5 = one device pixel on Retina, the thinnest there is)

    def __init__(self):
        super().__init__()
        self._brush, self._dpr = None, None
        self.photo: "PreviewLabel | None" = None      # when it is zoomed in, it fills the whole window
        self.black = False                            # full screen: nothing but the photo on black
        self._press: QPointF | None = None
        self._panned = False

    def _tile(self, dpr: float) -> QBrush:
        n = round(self.SPACING * dpr)                 # device pixels
        dot = max(1, round(self.DOT * dpr))
        if self.PATTERN == "mat":                     # one major cell: a fine line per cell, a stronger one on its edge
            big = n * self.MAJOR
            tile = QPixmap(big, big)
            tile.fill(QColor(COLORS["bg"]))
            p = QPainter(tile)
            fine = QColor(COLORS["grid_fine"])
            for i in range(1, self.MAJOR):
                p.fillRect(QRect(0, i * n, big, dot), fine)
                p.fillRect(QRect(i * n, 0, dot, big), fine)
            p.fillRect(QRect(0, 0, big, dot), QColor(COLORS["grid"]))
            p.fillRect(QRect(0, 0, dot, big), QColor(COLORS["grid"]))
            p.end()
            return QBrush(tile)
        tile = QPixmap(n, n)
        tile.fill(QColor(COLORS["bg"]))
        p = QPainter(tile)
        c, colour = (n - dot) // 2, QColor(COLORS["dots"])
        if self.PATTERN == "lines":                   # one horizontal and one vertical line per tile
            p.fillRect(QRect(0, c, n, dot), colour)
            p.fillRect(QRect(c, 0, dot, n), colour)
        else:
            p.fillRect(QRect(c, c, dot, dot), colour)  # exact pixels
        p.end()
        tile.setDevicePixelRatio(dpr)
        return QBrush(tile)

    def paintEvent(self, e):
        dpr = self.devicePixelRatioF()
        if dpr != self._dpr:                          # first paint, or moved to another screen
            self._brush, self._dpr = self._tile(dpr), dpr
        p = QPainter(self)
        ph = self.photo
        if self.black:
            p.fillRect(self.rect(), QColor("#000000"))
        if ph is not None and ph.zoomed() and ph.isVisible():
            if not self.black:
                p.fillRect(self.rect(), QColor(COLORS["bg"]))
            pix = ph.hires or ph.source
            r = ph.view_rect().translated(QPointF(ph.mapTo(self, QPoint(0, 0))))
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawPixmap(r, pix, QRectF(pix.rect()))
        elif not self.black:
            p.fillRect(self.rect(), self._brush)

    # Gestures on empty parts of the window (and on panels' blank areas) move a zoomed photo too.
    def _active(self) -> bool:
        return self.photo is not None and self.photo.zoomed() and self.photo.isVisible()

    def _to_photo(self, pos) -> QPointF:
        return QPointF(self.photo.mapFrom(self, pos.toPoint()))

    def event(self, e):
        if (e.type() == QEvent.NativeGesture and e.gestureType() == Qt.ZoomNativeGesture and self._active()):
            self.photo.zoom_at(self.photo.zoom * (1.0 + e.value()), self._to_photo(e.position()))
            return True
        return super().event(e)

    def wheelEvent(self, e):
        if self._active():
            self.photo.handle_wheel(e, self._to_photo(e.position()))
        else:
            super().wheelEvent(e)

    def mousePressEvent(self, e):
        if self._active() and e.button() == Qt.LeftButton:
            self._press, self._panned = e.position(), False
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._press is not None and self._active():
            d = e.position() - self._press
            if self._panned or abs(d.x()) + abs(d.y()) > 4:
                self._panned = True
                self.photo.pan_by(d.x(), d.y())
                self._press = e.position()

    def mouseReleaseEvent(self, e):
        if self._press is not None:
            if not self._panned and self._active():
                self.photo.reset_zoom()               # a click on the background goes back to Fit
            self._press = None
            self.unsetCursor()


def _icon_pencil(p: QPainter, _arg=None):
    """A small pencil (drawn in a 20 x 20 box), pointing down to the left."""
    body = QPainterPath()
    body.moveTo(13.2, 4.8)
    body.lineTo(15.2, 6.8)
    body.lineTo(8.0, 14.0)
    body.lineTo(5.0, 15.0)
    body.lineTo(6.0, 12.0)
    body.closeSubpath()
    p.drawPath(body)
    p.drawLine(QPointF(11.6, 6.4), QPointF(13.6, 8.4))


def _icon_trash(p: QPainter, _arg=None):
    p.drawLine(QPointF(4.5, 6.0), QPointF(15.5, 6.0))                       # the lid
    p.drawPolyline([QPointF(8.0, 6.0), QPointF(8.5, 4.0), QPointF(11.5, 4.0), QPointF(12.0, 6.0)])
    p.drawPolyline([QPointF(6.0, 6.0), QPointF(7.0, 16.0), QPointF(13.0, 16.0), QPointF(14.0, 6.0)])
    p.drawLine(QPointF(8.8, 8.5), QPointF(9.1, 13.5))
    p.drawLine(QPointF(11.2, 8.5), QPointF(10.9, 13.5))


def _icon_plus(p: QPainter, _arg=None):
    p.drawLine(QPointF(10, 4.5), QPointF(10, 15.5))
    p.drawLine(QPointF(4.5, 10), QPointF(15.5, 10))


def _icon_save(p: QPainter, _arg=None):
    """Save the session: a small disk, its shutter at the top and its label below."""
    p.drawRoundedRect(QRectF(4.0, 4.0, 12.0, 12.0), 1.6, 1.6)
    p.drawRect(QRectF(7.0, 4.0, 5.5, 3.6))
    p.drawRect(QRectF(6.5, 10.5, 7.0, 5.5))


class SaveButton(IconButton):
    """The save icon: a small dot beside it while the session has changes it has not saved."""

    def __init__(self, slot):
        super().__init__(_icon_save, None, slot, f"Save this session: your rolls and adjustments, ready next time ({keys.MOD}+S)")
        self.dirty = False

    def set_dirty(self, dirty: bool):
        if dirty != self.dirty:
            self.dirty = dirty
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.dirty:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACCENT))
            p.drawEllipse(QPointF(self.width() - 5.0, 5.0), 2.4, 2.4)
            p.end()


def _icon_chevron_down(p: QPainter, _arg=None):
    p.drawPolyline([QPointF(6.0, 8.0), QPointF(10.0, 12.0), QPointF(14.0, 8.0)])


def paint_slider(p: QPainter, x0: float, x1: float, ty: float, xo: float, xv: float, ticks: int,
                 pointer: QColor, fill: QColor | None, major: int = 0, gradient: list | None = None,
                 lit: bool = False):
    """The Lightroom-like slider: ticks standing on a thin track, a fill from zero, a small pointer.
    `ticks` marks the track in that many even steps (ticks + 1 lines, the ends on the track's ends);
    with `major`, every major-th tick is a little longer. With `gradient` (colours, left to right) the
    track itself shows what the slider does, as Lightroom's colour sliders do, and has no fill."""
    p.setPen(QPen(QColor(COLORS["tick"]), 1))
    for i in range(ticks + 1):
        x = round(x0 + (x1 - x0) * i / ticks - 0.5) + 0.5      # on whole pixels, never past the ends
        x = min(max(x, x0 + 0.5), x1 - 0.5)
        long_ = major and i % major == 0
        p.drawLine(QPointF(x, ty - (5.5 if long_ else 4.0)), QPointF(x, ty - 1.0))   # standing on the track
    if gradient:
        g = QLinearGradient(x0, 0, x1, 0)
        for i, c in enumerate(gradient):
            c = QColor(c)
            c.setAlphaF(1.0 if lit else 0.7)
            g.setColorAt(i / max(len(gradient) - 1, 1), c)
        p.setPen(QPen(QBrush(g), 2, Qt.SolidLine, Qt.FlatCap))
        p.drawLine(QPointF(x0, ty), QPointF(x1, ty))
        fill = None
    else:
        p.setPen(QPen(QColor(COLORS["track"]), 2, Qt.SolidLine, Qt.FlatCap))
        p.drawLine(QPointF(x0, ty), QPointF(x1, ty))
    if fill is not None and abs(xv - xo) > 0.5:
        p.setPen(QPen(fill, 2, Qt.SolidLine, Qt.FlatCap))
        p.drawLine(QPointF(xo, ty), QPointF(xv, ty))
    p.setPen(Qt.NoPen)
    p.setBrush(pointer)
    p.drawPolygon(QPolygonF([QPointF(xv - 5.0, ty + 4.5), QPointF(xv - 5.0, ty - 0.5), QPointF(xv, ty - 5.5),
                             QPointF(xv + 5.0, ty - 0.5), QPointF(xv + 5.0, ty + 4.5)]))


class TickSlider(QSlider):
    """Straighten's slider, drawn like the Refine ones: a tick per degree, a longer one every five."""

    def __init__(self):
        super().__init__(Qt.Horizontal)
        self._hover = False
        self.setFixedHeight(20)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def _span(self) -> tuple[float, float]:
        return 6.0, self.width() - 6.0

    def _x(self, v: int) -> float:
        x0, x1 = self._span()
        return x0 + (v - self.minimum()) / max(self.maximum() - self.minimum(), 1) * (x1 - x0)

    def _value_at(self, x: float) -> int:
        x0, x1 = self._span()
        t = min(max((x - x0) / (x1 - x0), 0.0), 1.0)
        return round(self.minimum() + t * (self.maximum() - self.minimum()))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        x0, x1 = self._span()
        lit = self._hover or self.isSliderDown()
        moved = self.value() != 0
        degrees = round((self.maximum() - self.minimum()) / 10)        # the slider counts tenths
        paint_slider(p, x0, x1, self.height() / 2 + 3, self._x(0), self._x(self.value()), degrees,
                     QColor(ACCENT if lit else COLORS["pointer_on"] if moved else COLORS["pointer"]),
                     QColor(COLORS["fill"]), major=5)
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.setSliderDown(True)
            self.setValue(self._value_at(e.position().x()))

    def mouseMoveEvent(self, e):
        if self.isSliderDown():
            self.setValue(self._value_at(e.position().x()))

    def mouseReleaseEvent(self, e):
        if self.isSliderDown():
            self.setSliderDown(False)
            self.update()

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()


# The colour mixer's bands, in NegPy's order (engine Settings.hsl_*), with the colour each dot shows
HSL_BANDS = (("Red", "#e5484d"), ("Orange", "#ee8a3c"), ("Yellow", "#e6c548"), ("Green", "#4fae5f"),
             ("Aqua", "#3fb4c2"), ("Blue", "#4677d8"), ("Purple", "#8d5fd3"), ("Magenta", "#d4459c"))


class SlideTabs(QWidget):
    """A segmented strip: the names sit on one recessed row and a single raised highlight glides
    under the chosen one. Click or drag across it to choose."""

    chosen = Signal(int)

    def __init__(self, labels: list[str]):
        super().__init__()
        self.labels, self.current, self._hover = list(labels), 0, -1
        self._pos = 0.0                                   # where the highlight is, in tabs (glides)
        self._glide = QVariantAnimation(self, duration=260)
        self._glide.setEasingCurve(QEasingCurve.OutCubic)
        self._glide.valueChanged.connect(self._moved)
        self.setFixedHeight(28)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def _moved(self, v):
        self._pos = float(v)
        self.update()

    def _cell(self) -> float:
        return (self.width() - 6) / len(self.labels)

    def _at(self, x: float) -> int:
        return min(max(int((x - 3) // max(self._cell(), 1)), 0), len(self.labels) - 1)

    def select(self, i: int, animate: bool = True):
        if i == self.current and (self._glide.state() == QVariantAnimation.Running or self._pos == i):
            return
        self.current = i
        self._glide.stop()
        if animate and self.isVisible():
            self._glide.setStartValue(self._pos)
            self._glide.setEndValue(float(i))
            self._glide.start()
        else:
            self._pos = float(i)
            self.update()
        self.chosen.emit(i)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.select(self._at(e.position().x()))

    def mouseMoveEvent(self, e):
        hover = self._at(e.position().x())
        if e.buttons() & Qt.LeftButton:
            self.select(hover)
        if hover != self._hover:
            self._hover = hover
            self.update()

    def leaveEvent(self, e):
        self._hover = -1
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(COLORS["edge_lo"]), 1))                 # the recessed row
        p.setBrush(QColor(COLORS["field"]))
        p.drawRoundedRect(r, 6, 6)
        cell = self._cell()
        pill = QRectF(3 + self._pos * cell, 3, cell, self.height() - 6)
        g = QLinearGradient(0, pill.top(), 0, pill.bottom())          # the raised highlight
        g.setColorAt(0, QColor(COLORS["button_top"]))
        g.setColorAt(1, QColor(COLORS["button"]))
        p.setPen(QPen(QColor(COLORS["edge_lo"]), 1))
        p.setBrush(QBrush(g))
        p.drawRoundedRect(pill.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
        p.setPen(QPen(QColor(COLORS["edge_hi"]), 1))
        p.drawLine(QPointF(pill.left() + 4, pill.top() + 1.0), QPointF(pill.right() - 4, pill.top() + 1.0))
        f = p.font()
        f.setPixelSize(12)
        for i, name in enumerate(self.labels):
            near = max(0.0, 1.0 - abs(self._pos - i))                # the text lights as the highlight arrives
            f.setWeight(QFont.DemiBold if near > 0.5 else QFont.Normal)
            p.setFont(f)
            rest = COLORS["text"] if i == self._hover else COLORS["muted"]
            p.setPen(QColor(mix(rest, COLORS["text"], near)))
            p.drawText(QRectF(3 + i * cell, 0, cell, self.height()), Qt.AlignCenter, name)
        p.end()


class ClipBox(QWidget):
    """Holds one widget at its natural height and clips it: animating this box's height uncovers
    or hides the content instead of squeezing its rows together."""

    def __init__(self, content: QWidget):
        super().__init__()
        self.content = content
        content.setParent(self)
        content.installEventFilter(self)

    def eventFilter(self, obj, e):
        if obj is self.content and e.type() == QEvent.LayoutRequest:
            self.updateGeometry()
            self._place()
        return False

    def sizeHint(self):
        return self.content.sizeHint()

    def minimumSizeHint(self):
        return QSize(self.content.minimumSizeHint().width(), 0)

    def _place(self):
        self.content.setGeometry(0, 0, self.width(), max(self.content.sizeHint().height(), 1))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place()


class FitStack(QStackedWidget):
    """A stack as tall as the page it shows, not its tallest page (the Refine tabs)."""

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w else super().minimumSizeHint()


class BwJoke(QWidget):
    """The Color page of a black-and-white film: the mixer's dots gone grey, hopping once, and a joke."""

    JOKES = ("Colour? On black-and-white film? Bold move.",
             "This roll sees the world like a 1940s detective.",
             "Every colour here is grey. We checked twice.",
             "Plot twist: it was grey all along.",
             "The mixer is on a coffee break. Black, of course.",
             "Silver halide does not do hue. It does drama.")

    def __init__(self):
        super().__init__()
        self.setFixedHeight(120)
        self._joke = 0
        self._hop = QVariantAnimation(self, startValue=0.0, endValue=1.0, duration=900)
        self._hop.setEasingCurve(QEasingCurve.Linear)
        self._hop.valueChanged.connect(lambda _v: self.update())

    def joke(self) -> str:
        return self.JOKES[self._joke]

    def showEvent(self, e):
        super().showEvent(e)
        import random
        self._joke = random.choice([i for i in range(len(self.JOKES)) if i != self._joke])
        self._hop.stop()
        self._hop.start()

    def paintEvent(self, e):
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        t = float(self._hop.currentValue() or 0.0) if self._hop.state() == QVariantAnimation.Running else 1.0
        n, gap = len(HSL_BANDS), 22.0
        x0 = (self.width() - gap * (n - 1)) / 2
        for i, (_name, colour) in enumerate(HSL_BANDS):
            c = QColor(colour)
            g = round(0.30 * c.red() + 0.59 * c.green() + 0.11 * c.blue())
            phase = t * 1.6 - i * 0.08                   # a wave from left to right
            hop = 6.0 * math.sin(math.pi * phase / 0.5) if 0.0 < phase < 0.5 else 0.0
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(g, g, g))
            p.drawEllipse(QPointF(x0 + i * gap, 36.0 - hop), 7.0, 7.0)
        f = p.font()
        f.setPixelSize(12)
        p.setFont(f)
        p.setPen(QColor(COLORS["muted"]))
        p.drawText(QRectF(8, 58, self.width() - 16, 50), Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self.joke())
        p.end()


def hsl_gradients(band: int) -> dict:
    """What each mixer slider does to the chosen colour, as its track: Hue runs from the colour's
    neighbour to the other one (only half way: the hue turns 30°), Saturation from grey to the colour,
    Luminance from dark to light."""
    n = len(HSL_BANDS)
    c = QColor(HSL_BANDS[band][1])
    half = lambda a, b: QColor((a.red() + b.red()) // 2, (a.green() + b.green()) // 2, (a.blue() + b.blue()) // 2)  # noqa: E731
    grey = round(0.30 * c.red() + 0.59 * c.green() + 0.11 * c.blue())
    return {"hsl_hue": [half(c, QColor(HSL_BANDS[(band - 1) % n][1])), c, half(c, QColor(HSL_BANDS[(band + 1) % n][1]))],
            "hsl_saturation": [QColor(grey, grey, grey), c],
            "hsl_luminance": [c.darker(260), c, c.lighter(150)]}


class ColorDot(QPushButton):
    """One band of the colour mixer: a small round swatch; a ring when chosen, a dot when adjusted."""

    def __init__(self, name: str, colour: str):
        super().__init__()
        self.colour, self.edited = QColor(colour), False
        self.setCheckable(True)
        self.setFixedSize(26, 26)
        self.setToolTip(name)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("QPushButton { background: transparent; border: none; }")

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        if self.isChecked():
            p.setPen(QPen(QColor(COLORS["text"]), 1.4))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 10.0, 10.0)
        swatch = QColor(self.colour)
        if not self.isChecked() and not self.underMouse():
            swatch.setAlpha(170)
        p.setPen(Qt.NoPen)
        p.setBrush(swatch)
        p.drawEllipse(c, 7.0, 7.0)
        if self.edited:                            # adjusted: a small mark below
            p.setBrush(QColor(COLORS["text"]))
            p.drawEllipse(QPointF(c.x(), self.height() - 1.5), 1.4, 1.4)
        p.end()


class TrashButton(IconButton):
    """The roll card's trash: quiet grey at rest, red with a soft glow under the mouse."""

    def __init__(self, slot, tip: str):
        super().__init__(_icon_trash, None, slot, tip)
        self.setObjectName("rollEdit")            # flat, no plate
        self.setMinimumHeight(0)
        self.setFixedSize(22, 22)
        self.scale = 0.78

    def paintEvent(self, e):
        if self._t > 0.01 and self.isEnabled():
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            glow = QRadialGradient(QPointF(self.width() / 2, self.height() / 2), self.width() / 2)
            red = QColor(COLORS["danger"])
            red.setAlpha(int(90 * self._t))
            glow.setColorAt(0.0, red)
            red.setAlpha(0)
            glow.setColorAt(1.0, red)
            p.setPen(Qt.NoPen)
            p.setBrush(glow)
            p.drawEllipse(self.rect())
            p.end()
        rest = self.rest_color or COLORS["muted"]
        self.rest_color = mix(rest, COLORS["danger"], self._t) if self._t > 0.01 else rest
        accent = self._t
        self._t = 0.0                               # the icon in its own red, not the accent of other icons
        super().paintEvent(e)
        self._t, self.rest_color = accent, rest


class ElidedLabel(QLabel):
    """One line that ends in … when it does not fit, instead of widening its parent; the whole
    text is then the tooltip of `tip_on`."""

    def __init__(self, tip_on: QWidget | None = None):
        super().__init__()
        self._full = ""
        self._tip_on = tip_on
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)   # as wide as its text, or less

    def setText(self, text: str):
        self._full = text
        self._elide()
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 2, super().sizeHint().height())

    def text(self) -> str:
        return self._full

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _elide(self):
        shown = self.fontMetrics().elidedText(self._full, Qt.ElideRight, max(self.width(), 0))
        if shown != super().text():
            super().setText(shown)
        (self._tip_on or self).setToolTip(self._full if shown != self._full else "")


class RollCard(QFrame):
    """One roll (or the single frames) in the left column: its name, then film stock and frame count.
    A roll's chevron expands the card to show its details (the details widget is moved in)."""

    clicked = Signal()
    toggled = Signal(bool)
    menu = Signal(object)                 # right-click: the global position for a menu
    renamed = Signal(str)                 # the pencil: a new name typed for the roll
    remove = Signal()                     # the trash

    def contextMenuEvent(self, e):
        self.menu.emit(e.globalPos())

    def __init__(self, is_roll: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("rollCard")
        self.setCursor(Qt.PointingHandCursor)
        self._selected = False
        self._open = Fade(self, self._paint_state)      # details shown: the name plate turns white
        self._hover = Fade(self, self._paint_state)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.head = QFrame()                          # the name plate: white with black text when selected
        self.head.setObjectName("rollHead")
        head = QHBoxLayout(self.head)
        head.setContentsMargins(14, 10, 10, 10)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title = ElidedLabel(tip_on=self.head)    # a long folder name ends in …, the card stays put
        self.title.setObjectName("rollTitle")
        self.sub = ElidedLabel()
        self.sub.setObjectName("rollSub")
        for w in (self.title, self.sub):
            w.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.name_edit = QLineEdit()                  # replaces the title while renaming
        self.name_edit.setObjectName("rollRename")
        self.name_edit.hide()
        self.name_edit.returnPressed.connect(self._finish_rename)
        self.name_edit.installEventFilter(self)
        self.pencil = IconButton(_icon_pencil, None, self.start_rename, "Rename the roll")
        self.pencil.setObjectName("rollEdit")
        self.pencil.setMinimumHeight(0)
        self.pencil.setFixedSize(16, 16)
        self.pencil.scale = 0.72                  # a small mark, not a button
        title_row = QHBoxLayout()
        title_row.setSpacing(5)
        title_row.addWidget(self.pencil)          # left of the name: always in the same place
        title_row.addWidget(self.title, 1)
        title_row.addWidget(self.name_edit, 1)
        text.addLayout(title_row)
        text.addWidget(self.sub)
        head.addLayout(text, 1)
        self.trash = TrashButton(self.remove.emit, "Remove from the list (the scans on disk are not touched)")
        head.addWidget(self.trash, 0, Qt.AlignVCenter)
        head.addSpacing(2)
        self.chevron = Chevron()
        self.chevron.setToolTip("Roll details")
        self.chevron.toggled.connect(self._on_toggle)
        head.addWidget(self.chevron, 0, Qt.AlignVCenter)
        lay.addWidget(self.head)
        self.body = QVBoxLayout()                     # the details go here while expanded
        self.body.setContentsMargins(12, 0, 12, 0)
        lay.addLayout(self.body)
        # only now that it has a parent: a visible widget without one opens as a window of its own
        self.chevron.setVisible(is_roll)
        self.pencil.setVisible(is_roll)

    def _on_toggle(self, on: bool):
        self.toggled.emit(on)

    def start_rename(self):
        self.name_edit.setText(self.title.text())
        self.title.hide()
        self.pencil.hide()
        self.name_edit.show()
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _finish_rename(self, keep: bool = True):
        if not self.name_edit.isVisible():
            return
        name = self.name_edit.text().strip()
        self.name_edit.hide()
        self.title.show()
        self.pencil.show()
        if keep and name and name != self.title.text():
            self.renamed.emit(name)

    def eventFilter(self, obj, e):
        if obj is self.name_edit:
            # Esc cancels; caught before the window's own Esc shortcut takes the focus (which keeps the text)
            if e.type() in (QEvent.ShortcutOverride, QEvent.KeyPress) and e.key() == Qt.Key_Escape:
                self._finish_rename(keep=False)
                e.accept()
                return True
            if e.type() == QEvent.FocusOut:           # clicking elsewhere keeps what was typed
                self._finish_rename()
        return super().eventFilter(obj, e)

    def set_text(self, title: str, sub: str):
        self.title.setText(title)
        need = FILM_NEEDED in sub                     # the one thing to do first: the line in the accent
        self.sub.setText(sub)
        if need != getattr(self, "_needs_film", False):
            self._needs_film = need
            self._paint_state()

    def set_selected(self, on: bool):
        """The roll on screen: its name in neon (and, while its details are open, a white plate)."""
        self._selected = on
        self._paint_state()

    def set_open(self, on: bool):
        self._open.to(1.0 if on else 0.0)

    def enterEvent(self, e):
        self._hover.to(1.0)

    def leaveEvent(self, e):
        self._hover.to(0.0)

    def _paint_state(self, _t: float = 0.0):
        o, h = self._open.t, self._hover.t
        bg = mix(mix(COLORS["panel"], COLORS["raise"], h), COLORS["button"], o)    # open: a lighter grey plate
        title = mix(ACCENT if self._selected else COLORS["text"], COLORS["text"], o)
        sub = COLORS["muted"]
        line = ACCENT2 if getattr(self, "_needs_film", False) else sub
        self.head.setStyleSheet(f"QFrame#rollHead {{ background: {bg}; border-radius: 9px; }} "
                                f"QLabel#rollTitle {{ color: {title}; }} "
                                f"QLabel#rollSub {{ color: {line}; }}")
        self.pencil.rest_color = sub
        self.pencil.update()
        self.trash.rest_color = sub
        self.trash.update()
        self.chevron.on_light = False
        self.chevron.update()

    def mousePressEvent(self, e):
        self.clicked.emit()


class ChoiceCard(QFrame):
    """A clickable card: a title and a smaller description under it (the welcome screen)."""

    clicked = Signal()

    def __init__(self, title: str, text: str):
        super().__init__()
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(26, 22, 26, 22)
        lay.setSpacing(8)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        d = QLabel(text)
        d.setObjectName("cardText")
        d.setWordWrap(True)
        for w in (t, d):
            w.setAttribute(Qt.WA_TransparentForMouseEvents)
            w.setAlignment(Qt.AlignCenter)
        lay.addSpacing(18)                       # titles line up whatever the description length
        t.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)   # as wide as its text: so is the underline
        lay.addWidget(t, 0, Qt.AlignHCenter)
        lay.addSpacing(4)
        lay.addWidget(d)
        lay.addStretch(1)
        self._fade = Fade(self, self._tint)

    def _tint(self, t: float):
        """Under the mouse: black turns to white and the text reverses; the neon underline stays."""
        if t <= 0.001:
            self.setStyleSheet("")
            return
        self.setStyleSheet(
            f"QFrame#card {{ background: {mix(COLORS['panel'], COLORS['select'], t)}; }} "
            f"QLabel#cardTitle {{ color: {mix(COLORS['text'], COLORS['on_select'], t)}; }} "
            f"QLabel#cardText {{ color: {mix(COLORS['muted'], COLORS['on_select_muted'], t)}; }}")

    def enterEvent(self, e):
        self._fade.to(1.0)

    def leaveEvent(self, e):
        self._fade.to(0.0)

    def mousePressEvent(self, e):
        self.clicked.emit()


class StripGuard(QObject):
    """The filmstrip under the mouse. A clicked frame is centred only after the release: centred on the
    press, the strip slid under a still-held button, and the slightest move (or the release itself)
    selected the frames it passed. Dragging without Cmd or Shift selects nothing for the same reason."""

    def __init__(self, strip: QWidget, center):
        super().__init__(strip)
        self._center = center
        self.pressed = False
        self.center_on_release = False

    def eventFilter(self, obj, e):
        t = e.type()
        if t == QEvent.MouseButtonPress and e.button() == Qt.LeftButton:
            self.pressed = True
        elif t == QEvent.MouseMove and self.pressed and not (e.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier)):
            return True                               # no drag-selection: a click is a click
        elif t == QEvent.MouseButtonRelease and e.button() == Qt.LeftButton:
            self.pressed = False
            if self.center_on_release:
                self.center_on_release = False
                QTimer.singleShot(0, self._center)    # after the view has finished the click
        return False


class SelectOnFocus(QObject):
    """Clicking into a field selects its text, so typing replaces it (the film stock search)."""

    def eventFilter(self, obj, e):
        if e.type() == QEvent.FocusIn:
            # an editable dropdown keeps the focus itself and hands the keys to its line edit
            line = obj.lineEdit() if isinstance(obj, QComboBox) else obj
            if isinstance(line, QLineEdit):
                QTimer.singleShot(0, line.selectAll)  # after the click has placed the cursor
        return False


class WheelGuard(QObject):
    """Scrolling a panel scrolls the panel. A dropdown, number field or slider under the pointer
    would take the wheel and change its value (a roll's film stock, its development) on the way
    past; inside a scroll area the wheel goes to the area instead. A focused field or slider
    still takes it: it was clicked first. The app's own faders keep their wheel."""

    def eventFilter(self, obj, e):
        if e.type() != QEvent.Wheel or not isinstance(obj, (QComboBox, QAbstractSpinBox, QSlider)):
            return False
        if not isinstance(obj, QComboBox) and obj.hasFocus():
            return False
        w = obj.parentWidget()
        while w is not None and not isinstance(w, QScrollArea):
            w = w.parentWidget()
        if w is None:
            return False
        QApplication.sendEvent(w.verticalScrollBar(), e)
        return True


class DecimalSpinBox(QDoubleSpinBox):
    """A number field that takes "." and "," alike as the decimal point.

    Qt only accepts the system's own separator: with a comma region (Italy, most of Europe) a "."
    typed from the numeric keypad was dropped without a word, and "-2.3" became -2."""

    def _local(self, text: str) -> str:
        point = self.locale().decimalPoint()
        other = "," if point == "." else "."
        return text.replace(other, point)

    def keyPressEvent(self, e):
        if e.text() in (".", ",") and e.text() != self.locale().decimalPoint():
            self.lineEdit().insert(self.locale().decimalPoint())
            return
        super().keyPressEvent(e)

    def validate(self, text: str, pos: int):
        return super().validate(self._local(text), pos)        # pasted text too

    def valueFromText(self, text: str) -> float:
        return super().valueFromText(self._local(text))


class LabeledSlider(QWidget):
    """A slider with a name and editable angle (used for Straighten)."""

    changed = Signal()

    def __init__(self, name: str, lo: float, hi: float, default: float):
        super().__init__()
        self.default = default
        self.name = QLabel(name)
        self.name.setToolTip("Double-click to reset")
        self.value_input = DecimalSpinBox()
        self.value_input.setObjectName("straightenAngle")
        self.value_input.setRange(lo, hi)
        self.value_input.setDecimals(1)
        self.value_input.setSingleStep(0.1)
        self.value_input.setSuffix("°")
        self.value_input.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_input.setAccessibleName(name)
        self.value_input.setKeyboardTracking(False)
        self.value_input.valueChanged.connect(lambda v: self.set_value(v, emit=True))
        self.slider = TickSlider()
        self.slider.setRange(round(lo * 10), round(hi * 10))
        self.slider.valueChanged.connect(self._on_slider)
        self.pressed, self.released = self.slider.sliderPressed, self.slider.sliderReleased
        self.name.setObjectName("rowLabel")
        self.top = top = QHBoxLayout()            # a button can be added at its right end
        top.addWidget(self.name)
        top.addStretch(1)
        top.setAlignment(Qt.AlignVCenter)
        top.addWidget(self.value_input)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)
        lay.addLayout(top)
        lay.addWidget(self.slider)
        self.name.mouseDoubleClickEvent = lambda e: self.set_value(self.default, emit=True)
        self.set_value(default)

    def value(self) -> float:
        return self.slider.value() / 10

    def set_value(self, v: float, emit: bool = False):
        self.slider.blockSignals(not emit)
        self.slider.setValue(round(v * 10))
        self.slider.blockSignals(False)
        self._show_value()

    def _show_value(self):
        self.value_input.blockSignals(True)
        self.value_input.setValue(self.value())
        self.value_input.blockSignals(False)

    def _on_slider(self):
        self._show_value()
        self.changed.emit()


class PreviewLabel(QLabel):
    """Shows the image scaled to fit. In crop mode a frame lies on it: pull a corner to resize,
    drag inside to move (a chosen film format keeps its shape); in pick mode a click reports
    where it landed (white-balance picker)."""

    picked = Signal(QPointF)
    heal_clicked = Signal(QPointF)
    area_drawn = Signal(object)          # a closed shape: [QPointF, ...] in 0..1 of the picture as shown
    area_removed = Signal(QPointF)       # right-click: take away the area under this point
    brush_wheel = Signal(int)
    zoom_changed = Signal(float)          # 1.0 = fit (the furthest out)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.source: QPixmap | None = None
        self._aspect = 0.0                   # shape of the last full render (drafts are drawn in it)
        self._canvas_zoomed = False          # the window background last drew the photo
        self.angle = 0.0          # extra rotation shown on top of the last render (live straighten)
        self.turn_fills = True    # zoom the live turn to hide the corners (no crop: NegPy trims them)
        self.grid = False         # levelling grid while straightening
        self.crop_mode = False
        self.picking = False
        self.heal_mode = False
        self.brush_px = 10.0                 # brush radius on screen
        self.heal_marks: list[tuple[QPointF, float]] = []   # healed spots: (0..1 position, radius px)
        self.area_mode = False               # drawing areas to clean (dust and scratches found for you)
        self.area_marks: list[list[QPointF]] = []           # the areas of this frame, 0..1 of the picture
        self._lasso: list[QPointF] | None = None            # the shape being drawn, in widget pixels
        self._mouse: QPointF | None = None
        self.setMouseTracking(True)
        self.crop: QRectF | None = None      # normalized 0..1 in the image
        self.ratio: float | None = None      # width / height the frame is held to (None = free)
        # where a film format's frame may go: the whole picture, or what Auto crop kept (not the border)
        self.format_area = QRectF(0, 0, 1, 1)
        self._grab: str | None = None        # what is being dragged: "tl" "tr" "bl" "br" or "move"
        self._grab_from: QPointF | None = None
        self._grab_rect: QRectF | None = None
        # Zoom, like Lightroom: 1.0 = fit (never smaller); `center` = the picture point (0..1) in the
        # middle of the view; `hires` = a full-resolution render, used once the preview runs out of pixels
        self.zoom = 1.0
        self.border = 0.0                 # the white border, as a fraction of the picture's long side
        # Moving between views glides and cross-fades (frame switch, turn, crop on/off); edits never do.
        self._glide = QVariantAnimation(self)
        self._glide.setDuration(240)
        self._glide.setEasingCurve(QEasingCurve.OutCubic)
        self._glide.setStartValue(0.0)
        self._glide.setEndValue(1.0)
        self._glide.valueChanged.connect(lambda _v: self.refresh())
        self._glide_from: QRectF | None = None
        self._fade = QVariantAnimation(self)
        self._fade.setDuration(200)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.valueChanged.connect(lambda _v: self.refresh())
        self._fade.finished.connect(self._end_fade)
        self._old: QPixmap | None = None
        self._old_rect: QRectF | None = None
        self.center = QPointF(0.5, 0.5)
        self.zoom_100 = 0.0                  # the zoom at which one image pixel = one screen pixel (0 = unknown)
        self.hires: QPixmap | None = None
        self.fit_hires: QPixmap | None = None    # full view: the full-resolution render, drawn at fit size
        self.loading = False
        self.full_view = False                    # the window shows only the picture (F)
        self._spin = 0.0                          # the centre loader's turn, degrees
        self._spin_timer = QTimer(self, interval=16)
        self._spin_timer.timeout.connect(self._tick_spin)
        self._press: QPointF | None = None
        self._panned = False
        self._brush_delta = 0                # Alt + scroll not yet turned into brush steps (1/8 degrees)
        self._hint_timer = QTimer(singleShot=True, interval=1200)
        self._hint_timer.timeout.connect(self.update)

    def glide_from(self, rect: QRectF):
        """Move the picture smoothly from `rect` (in this widget's coordinates now) to where it belongs."""
        if self.zoom > 1.0 or rect.isEmpty() or not self.source:
            return
        self._glide_from = QRectF(rect)
        self._glide.stop()
        self._glide.start()

    def _end_fade(self):
        self._old = self._old_rect = None
        self.refresh()

    def set_image(self, pix: QPixmap | None, draft: bool = False, fade: bool = False):
        """A draft (a quicker, smaller render while a slider moves) keeps the frame of the last full
        render: their shapes differ by a hair, which would nudge the picture up and down.
        `fade`: a new view of the picture (another frame, a turn, the crop tool): the last one fades
        out over it and the picture glides to its new place, instead of jumping."""
        if fade and pix is not None and self.source is not None and self.zoom <= 1.0:
            self._old, self._old_rect = self.source, self.view_rect()
        else:
            fade = False
        self.source = pix
        if pix is not None and pix.height() > 0:
            aspect = pix.width() / pix.height()
            if not draft or not self._aspect or abs(aspect - self._aspect) / self._aspect > 0.01:
                old, self._aspect = self._aspect, aspect
                if self.crop_mode and self.ratio and self.crop and old and abs(aspect - old) / old > 0.001:
                    self._refit_crop()       # a format picked before the whole picture arrived
        if fade:
            if self.image_rect() != self._old_rect:
                self.glide_from(self._old_rect)
            self._fade.stop()
            self._fade.start()
        self.refresh()

    def _refit_crop(self):
        """Give the crop frame the held format again on this picture (same centre, same width
        where it fits): its 0..1 units changed shape with the picture."""
        k, c = self._k(), self.crop
        w = c.width()
        h = w / k
        if h > 1.0:
            h, w = 1.0, k
        x = min(max(c.center().x() - w / 2, 0.0), 1.0 - w)
        y = min(max(c.center().y() - h / 2, 0.0), 1.0 - h)
        self.crop = QRectF(x, y, w, h)

    # ----- zoom ----------------------------------------------------------------
    MAX_OVER_100 = 2.0                       # zoom in up to 2:1 (200%)

    def zoomed(self) -> bool:
        return self.zoom > 1.0 and self.source is not None

    def refresh(self):
        """Repaint. Zoomed in, the window background draws the photo, so that repaints too."""
        canvas = self.window().centralWidget() if hasattr(self.window(), "centralWidget") else None
        if isinstance(canvas, Backdrop) and (self.zoomed() or self._canvas_zoomed):
            self._canvas_zoomed = self.zoomed()
            canvas.update()
        self.update()

    def bounds(self) -> QRectF:
        """The area a zoomed photo may fill: the whole window (in this widget's coordinates)."""
        canvas = self.window().centralWidget() if hasattr(self.window(), "centralWidget") else None
        if isinstance(canvas, Backdrop):
            tl = self.mapFrom(canvas, QPoint(0, 0))
            return QRectF(tl.x(), tl.y(), canvas.width(), canvas.height())
        return QRectF(self.rect())

    def _ref(self) -> QPointF:
        """The point zooming is measured from: the middle of the picture at fit."""
        r = self.image_rect()
        return r.center() if not r.isEmpty() else QPointF(self.width() / 2, self.height() / 2)

    def view_rect(self) -> QRectF:
        """Where the whole picture is drawn now: the fit rectangle, zoomed around `center`."""
        r = self.image_rect()
        if self.zoom <= 1.0 and self._glide_from is not None and self._glide.state() == QVariantAnimation.Running \
                and not r.isEmpty():
            t, a = float(self._glide.currentValue()), self._glide_from
            return QRectF(a.x() + (r.x() - a.x()) * t, a.y() + (r.y() - a.y()) * t,
                          a.width() + (r.width() - a.width()) * t, a.height() + (r.height() - a.height()) * t)
        if self.zoom <= 1.0 or r.isEmpty():
            return r
        w, h = r.width() * self.zoom, r.height() * self.zoom
        ref, b = self._ref(), self.bounds()
        x = ref.x() - self.center.x() * w
        y = ref.y() - self.center.y() * h
        # bigger than the window: no empty space at its sides; smaller: kept wholly in view
        x = min(b.left(), max(b.right() - w, x)) if w > b.width() else min(max(x, b.left()), b.right() - w)
        y = min(b.top(), max(b.bottom() - h, y)) if h > b.height() else min(max(y, b.top()), b.bottom() - h)
        return QRectF(x, y, w, h)

    def _sync_center(self):
        v = self.view_rect()                 # store the clamped centre, so panning has no dead zone
        if v.width() > 0 and v.height() > 0:
            ref = self._ref()
            self.center = QPointF((ref.x() - v.x()) / v.width(), (ref.y() - v.y()) / v.height())

    def max_zoom(self) -> float:
        return max(1.0, (self.zoom_100 or 4.0) * self.MAX_OVER_100)

    def zoom_at(self, z: float, pos: QPointF | None = None):
        """Zoom to `z`, keeping the picture point under `pos` (default: the middle) where it is."""
        z = min(max(z, 1.0), self.max_zoom())
        ref = self._ref()
        pos = pos if pos is not None else ref
        v = self.view_rect()
        if v.width() <= 0:
            return
        px, py = (pos.x() - v.x()) / v.width(), (pos.y() - v.y()) / v.height()
        r = self.image_rect()
        w, h = r.width() * z, r.height() * z
        self.zoom = z
        self.center = QPointF((ref.x() - (pos.x() - px * w)) / w, (ref.y() - (pos.y() - py * h)) / h)
        self._sync_center()
        self._hint_timer.start()
        self._update_cursor()
        self.refresh()
        self.zoom_changed.emit(self.zoom)

    def toggle_zoom(self, pos: QPointF | None = None):
        """Fit <-> 100% (one image pixel per screen pixel), like a click in Lightroom."""
        if self.zoom > 1.001:
            self.reset_zoom()
        else:
            self.zoom_at(max(self.zoom_100 or 2.0, 1.5), pos)

    def reset_zoom(self):
        if self.zoom != 1.0:
            self.zoom, self.center = 1.0, QPointF(0.5, 0.5)
            self._hint_timer.start()
            self._update_cursor()
            self.refresh()
            self.zoom_changed.emit(1.0)

    def pan_by(self, dx: float, dy: float):
        v = self.view_rect()
        if self.zoom > 1.0 and v.width() > 0:
            self.center = QPointF(self.center.x() - dx / v.width(), self.center.y() - dy / v.height())
            self._sync_center()
            self.refresh()

    def _plain(self) -> bool:
        return not (self.crop_mode or self.picking or self.heal_mode or self.area_mode)

    def _update_cursor(self):
        if self._plain():
            self.setCursor(Qt.OpenHandCursor if self.zoom > 1.0 else Qt.ArrowCursor)

    def event(self, e):
        if e.type() == QEvent.NativeGesture and e.gestureType() == Qt.ZoomNativeGesture and self.source:
            self.zoom_at(self.zoom * (1.0 + e.value()), e.position())      # trackpad pinch
            return True
        return super().event(e)

    def _framed(self) -> float:
        """The white border's width as a fraction of the picture's long side; 0 while it is not shown."""
        return self.border if (self.border > 0 and not self.crop_mode) else 0.0

    def image_rect(self) -> QRectF:
        """Where the picture sits inside the label at fit: on whole pixels, from the last full render's shape.
        With the white border on, picture and border fit together; this is still the picture's own rectangle."""
        if not self.source:
            return QRectF()
        aspect = self._aspect or self.source.width() / max(self.source.height(), 1)
        f = self._framed()
        pw, ph = (1.0, 1.0 / aspect) if aspect >= 1 else (aspect, 1.0)    # the picture, long side 1
        ow, oh = pw + 2 * f, ph + 2 * f                                     # with its border
        w, h = self.width(), self.height()
        if w / max(h, 1) > ow / oh:
            s = h / oh
        else:
            s = w / ow
        iw, ih = int(round(pw * s)), int(round(ph * s))
        b = int(round(f * s))
        x0 = (self.width() - (iw + 2 * b)) // 2 + b
        y0 = (self.height() - (ih + 2 * b)) // 2 + b
        return QRectF(x0, y0, iw, ih)

    def paintEvent(self, e):
        if not self.source:
            return super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = self.view_rect()
        f = self._framed()
        if f and self.zoom <= 1.0:                    # the white border, like a print's
            b = round(f * max(r.width(), r.height()))
            p.fillRect(r.adjusted(-b, -b, b, b), QColor("#ffffff"))
        if self.zoom > 1.0:
            pass                             # the window background draws it, across the whole window
        elif abs(self.angle) > 0.001:
            # Live preview: turn the last render. With no crop it is zoomed so no empty corners show,
            # as the real render trims them; a crop (auto or yours) stays put, the picture turns in it.
            import math
            a = math.radians(abs(self.angle))
            ratio = max(r.width() / max(r.height(), 1), r.height() / max(r.width(), 1))
            zoom = math.cos(a) + math.sin(a) * ratio if self.turn_fills else 1.0
            p.save()
            p.setClipRect(r)
            p.translate(r.center())
            p.rotate(-self.angle)            # positive angle = counter-clockwise, like NegPy
            p.scale(zoom, zoom)
            p.translate(-r.center())
            p.drawPixmap(r.toRect(), self.source)
            p.restore()
        else:
            p.drawPixmap(r.toRect(), self.fit_hires or self.source)
        if self._old is not None and self._fade.state() == QVariantAnimation.Running and self.zoom <= 1.0:
            p.setOpacity(1.0 - float(self._fade.currentValue()))   # the last view, fading out over the new
            p.drawPixmap(self._old_rect.toRect(), self._old)
            p.setOpacity(1.0)
        if self.grid:
            p.setPen(QPen(QColor(255, 255, 255, 70), 1))
            for i in range(1, 8):
                x = r.left() + r.width() * i / 8
                y = r.top() + r.height() * i / 8
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        if self.crop_mode and self.crop:
            c = self._crop_px()
            shade = QColor(0, 0, 0, 165)          # darken everything outside the crop
            p.fillRect(QRectF(r.left(), r.top(), r.width(), c.top() - r.top()), shade)
            p.fillRect(QRectF(r.left(), c.bottom(), r.width(), r.bottom() - c.bottom()), shade)
            p.fillRect(QRectF(r.left(), c.top(), c.left() - r.left(), c.height()), shade)
            p.fillRect(QRectF(c.right(), c.top(), r.right() - c.right(), c.height()), shade)
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setPen(QPen(QColor(255, 255, 255, 60), 1))      # thirds, to compose by
            for i in (1, 2):
                x, y = c.left() + c.width() * i / 3, c.top() + c.height() * i / 3
                p.drawLine(QPointF(x, c.top()), QPointF(x, c.bottom()))
                p.drawLine(QPointF(c.left(), y), QPointF(c.right(), y))
            p.setPen(QPen(QColor(255, 255, 255, 210), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(c)
            arm = min(22.0, c.width() / 3, c.height() / 3)    # the corner grips
            p.setPen(QPen(QColor("#ffffff"), 3, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin))
            for cx, cy, dx, dy in ((c.left(), c.top(), 1, 1), (c.right(), c.top(), -1, 1),
                                   (c.left(), c.bottom(), 1, -1), (c.right(), c.bottom(), -1, -1)):
                p.drawPolyline([QPointF(cx + dx * arm, cy), QPointF(cx, cy), QPointF(cx, cy + dy * arm)])
            p.setRenderHint(QPainter.Antialiasing, True)
        if self.heal_mode:
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(Qt.NoBrush)
            for pos, rad in self.heal_marks:          # spots already healed
                c = QPointF(r.x() + pos.x() * r.width(), r.y() + pos.y() * r.height())
                p.setPen(QPen(QColor(0, 0, 0, 120), 2.5))
                p.drawEllipse(c, rad, rad)
                p.setPen(QPen(QColor(255, 255, 255, 170), 1))
                p.drawEllipse(c, rad, rad)
            if self._mouse is not None and r.contains(self._mouse):   # the brush
                p.setPen(QPen(QColor(0, 0, 0, 160), 3))
                p.drawEllipse(self._mouse, self.brush_px, self.brush_px)
                p.setPen(QPen(QColor(ACCENT), 1.5))
                p.drawEllipse(self._mouse, self.brush_px, self.brush_px)
        if self.area_mode:
            p.setRenderHint(QPainter.Antialiasing)
            wash = QColor(ACCENT2)
            wash.setAlpha(60)
            for shape in self.area_marks:             # the areas being cleaned: seen on any photo
                poly = QPolygonF([QPointF(r.x() + q.x() * r.width(), r.y() + q.y() * r.height()) for q in shape])
                p.setPen(Qt.NoPen)
                p.setBrush(wash)
                p.drawPolygon(poly)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(0, 0, 0, 170), 3.2))
                p.drawPolygon(poly)
                p.setPen(QPen(QColor(255, 255, 255, 235), 1.4, Qt.DashLine))
                p.drawPolygon(poly)
            if self._lasso and len(self._lasso) > 1:  # the one being drawn
                line = QPolygonF(self._lasso)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(0, 0, 0, 150), 3))
                p.drawPolyline(line)
                p.setPen(QPen(QColor("#ffffff"), 1.4))
                p.drawPolyline(line)
                p.setPen(QPen(QColor(255, 255, 255, 110), 1, Qt.DotLine))    # how it will close
                p.drawLine(self._lasso[-1], self._lasso[0])
        if self.zoom > 1.0 and self.loading and self.hires is None:
            self._badge(p, "Loading detail…", Qt.AlignTop | Qt.AlignRight)
        if self.centre_loading():
            self._paint_centre_loader(p)
            if not self._spin_timer.isActive():
                self._spin_timer.start()
        if self._hint_timer.isActive():
            pct = f"{round(100 * self.zoom / self.zoom_100)}%" if self.zoom_100 and self.zoom > 1.0 else "Fit"
            self._badge(p, pct, Qt.AlignBottom | Qt.AlignLeft)

    # ----- the crop frame ---------------------------------------------------------
    MIN_CROP = 0.06                          # smallest frame, as a share of the picture

    def _crop_px(self) -> QRectF:
        r = self.view_rect()
        return QRectF(r.x() + self.crop.x() * r.width(), r.y() + self.crop.y() * r.height(),
                      self.crop.width() * r.width(), self.crop.height() * r.height())

    def _k(self) -> float | None:
        """The held shape in the frame's own 0..1 units (the picture itself is not square)."""
        return self.ratio / self._aspect if self.ratio and self._aspect else None

    def crop_is_portrait(self) -> bool:
        c = self.crop or QRectF(0, 0, 1, 1)
        a = self._aspect or 1.0
        shape = c.width() * a / max(c.height(), 1e-6)
        if abs(shape - 1.0) < 0.03:              # a square frame has no direction: follow the picture's
            return a < 1.0
        return shape < 1.0

    def set_ratio(self, ratio: float | None):
        """Hold the frame to width / height `ratio` (None = free): the largest such frame the
        whole picture allows, kept as near the old frame's centre as it fits. (Fitting it inside
        the old frame instead would make every format after a narrow one, like 1:1, narrow too.)"""
        self.ratio = ratio
        k = self._k()
        if k and self.crop:
            self.crop = self._largest(k, self.crop.center(), self.format_area)
        self.refresh()

    @staticmethod
    def _largest(k: float, centre: QPointF, area: QRectF | None = None) -> QRectF:
        """The biggest frame of shape `k` (width / height in 0..1 units) inside `area` of the picture."""
        area = area or QRectF(0, 0, 1, 1)
        w = min(area.width(), area.height() * k)
        h = w / k
        x = min(max(centre.x() - w / 2, area.left()), area.right() - w)
        y = min(max(centre.y() - h / 2, area.top()), area.bottom() - h)
        return QRectF(x, y, w, h)

    def _grip_at(self, pos: QPointF) -> str | None:
        if not (self.crop_mode and self.crop and self.source):
            return None
        c, reach = self._crop_px(), 16.0
        for name, corner in (("tl", c.topLeft()), ("tr", c.topRight()), ("bl", c.bottomLeft()), ("br", c.bottomRight())):
            if abs(pos.x() - corner.x()) <= reach and abs(pos.y() - corner.y()) <= reach:
                return name
        return "move" if c.contains(pos) else None

    def _drag_crop(self, pos: QPointF):
        r, start, g = self.view_rect(), self._grab_rect, self._grab
        if r.width() <= 0 or r.height() <= 0:
            return
        if g == "move":
            dx = (pos.x() - self._grab_from.x()) / r.width()
            dy = (pos.y() - self._grab_from.y()) / r.height()
            x = min(max(start.x() + dx, 0.0), 1.0 - start.width())
            y = min(max(start.y() + dy, 0.0), 1.0 - start.height())
            self.crop = QRectF(x, y, start.width(), start.height())
        else:
            m = self._norm(pos)
            # the corner opposite the one being pulled stays where it is
            ax = start.right() if "l" in g else start.left()
            ay = start.bottom() if "t" in g else start.top()
            sx, sy = (-1 if "l" in g else 1), (-1 if "t" in g else 1)
            room_x = ax if sx < 0 else 1.0 - ax              # how far the frame may reach
            room_y = ay if sy < 0 else 1.0 - ay
            w = min(max((m.x() - ax) * sx, self.MIN_CROP), room_x)
            h = min(max((m.y() - ay) * sy, self.MIN_CROP), room_y)
            k = self._k()
            if k:                                            # keep the shape: follow the further pull
                w = min(max(w, h * k), room_x, room_y * k)
                h = w / k
            x0, y0 = (ax - w if sx < 0 else ax), (ay - h if sy < 0 else ay)
            self.crop = QRectF(x0, y0, w, h)
        self.refresh()

    def centre_loading(self) -> bool:
        """Full view, fitted, while the full-resolution picture is still on its way."""
        return (self.full_view and self.loading and self.zoom <= 1.0 and self.fit_hires is None
                and self.source is not None)

    def _tick_spin(self):
        if not self.centre_loading():
            self._spin_timer.stop()
            self.update()
            return
        self._spin = (self._spin + 6.0) % 360.0
        self.update()

    def _paint_centre_loader(self, p: QPainter):
        """A dark pill in the middle of the picture: a turning arc in the accent colour and what loads."""
        f = QFont()
        f.setPixelSize(12)
        p.setFont(f)
        text = "Loading full resolution…"
        w = p.fontMetrics().horizontalAdvance(text) + 58
        box = QRectF((self.width() - w) / 2, (self.height() - 36) / 2, w, 36)
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 175))
        p.drawRoundedRect(box, 18, 18)
        ring = QRectF(box.left() + 16, box.center().y() - 8, 16, 16)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 40), 2))
        p.drawEllipse(ring)
        pen = QPen(QColor(ACCENT2), 2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(ring, int(-self._spin * 16), 100 * 16)
        p.setPen(QColor(COLORS["soft"]))
        p.drawText(box.adjusted(42, 0, -16, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
        p.restore()

    def _badge(self, p: QPainter, text: str, where):
        f = QFont()
        f.setPixelSize(11)
        p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(text) + 18
        x = 10 if where & Qt.AlignLeft else self.width() - w - 10
        y = 10 if where & Qt.AlignTop else self.height() - 32
        box = QRectF(x, y, w, 22)
        p.fillRect(box, QColor(0, 0, 0, 170))
        p.setPen(QColor(COLORS["text"]))
        p.drawText(box, Qt.AlignCenter, text)

    def leaveEvent(self, e):
        self._mouse = None
        self.refresh()

    def wheelEvent(self, e):
        """Cmd + scroll zooms; scroll pans a zoomed picture; Alt + scroll sizes the heal brush."""
        if not self.source:
            return super().wheelEvent(e)
        self.handle_wheel(e, e.position())

    def handle_wheel(self, e, pos: QPointF):
        dy = e.angleDelta().y()
        if e.modifiers() & Qt.ControlModifier:           # Cmd on a Mac
            self.zoom_at(self.zoom * (1.0015 ** dy), pos)
        elif self.heal_mode and e.modifiers() & Qt.AltModifier:
            # One step per wheel notch (120); a trackpad's many small deltas add up, and its
            # zero-delta begin / end events do nothing (they used to shrink the brush)
            self._brush_delta += dy or e.angleDelta().x()
            while abs(self._brush_delta) >= 120:
                step = 1 if self._brush_delta > 0 else -1
                self._brush_delta -= 120 * step
                self.brush_wheel.emit(step)
        elif self.zoom > 1.0:
            d = e.pixelDelta() if not e.pixelDelta().isNull() else e.angleDelta() / 3
            self.pan_by(d.x(), d.y())
        e.accept()

    def _norm(self, pos) -> QPointF:
        r = self.view_rect()
        x = min(max((pos.x() - r.x()) / max(r.width(), 1), 0.0), 1.0)
        y = min(max((pos.y() - r.y()) / max(r.height(), 1), 0.0), 1.0)
        return QPointF(x, y)

    def mousePressEvent(self, e):
        if self.heal_mode and self.source and self.view_rect().contains(e.position()):
            self.heal_clicked.emit(self._norm(e.position()))
            return
        if self.area_mode and self.source:
            if e.button() == Qt.RightButton:
                self.area_removed.emit(self._norm(e.position()))
            elif e.button() == Qt.LeftButton and self.view_rect().contains(e.position()):
                self._lasso = [e.position()]
            return
        if self._plain() and self.source and e.button() == Qt.LeftButton:   # click: zoom; drag: pan
            self._press, self._panned = e.position(), False
            return
        if self.picking and self.source and self.view_rect().contains(e.position()):
            self.picked.emit(self._norm(e.position()))
            return
        if self.crop_mode and self.source and e.button() == Qt.LeftButton:
            self._grab = self._grip_at(e.position())
            self._grab_from, self._grab_rect = e.position(), QRectF(self.crop) if self.crop else None
            if self._grab == "move":
                self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._press is not None and e.buttons() & Qt.LeftButton:
            d = e.position() - self._press
            if self._panned or abs(d.x()) + abs(d.y()) > 4:
                self._panned = True
                self.setCursor(Qt.ClosedHandCursor)
                self.pan_by(d.x(), d.y())
                self._press = e.position()
            return
        if self.heal_mode:
            self._mouse = e.position()
            self.refresh()
        if self.area_mode and self._lasso is not None and e.buttons() & Qt.LeftButton:
            last, pos = self._lasso[-1], e.position()
            if abs(pos.x() - last.x()) + abs(pos.y() - last.y()) >= 3:      # a point every few pixels is plenty
                self._lasso.append(pos)
                self.refresh()
            return
        if self.crop_mode:
            if self._grab and e.buttons() & Qt.LeftButton:
                self._drag_crop(e.position())
            else:                                 # show what a press here would do
                g = self._grip_at(e.position())
                self.setCursor(Qt.SizeFDiagCursor if g in ("tl", "br") else Qt.SizeBDiagCursor if g in ("tr", "bl")
                               else Qt.OpenHandCursor if g == "move" else Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        if self._lasso is not None:
            shape, self._lasso = self._lasso, None
            box = QPolygonF(shape).boundingRect()
            if len(shape) >= 3 and box.width() > 8 and box.height() > 8:     # a click or a scribble is not an area
                self.area_drawn.emit([self._norm(q) for q in shape])
            self.refresh()
            return
        if self._press is not None:
            if not self._panned and self.view_rect().contains(e.position()):
                self.toggle_zoom(e.position())
            self._press = None
            self._update_cursor()
            return
        if self._grab == "move":
            self.setCursor(Qt.OpenHandCursor)
        self._grab = None


class DropOverlay(QWidget):
    """Shown while dragging files over the window: drop left = roll, right = single frames."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.side = "roll"
        self.hide()

    def side_at(self, x: float) -> str:
        return "roll" if x < self.width() / 2 else "frames"

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(15, 16, 18, 215))
        w, h, m = self.width(), self.height(), 40
        halves = [("roll", QRectF(m, m, w / 2 - 1.5 * m, h - 2 * m),
                   "Drop as a roll", "Frames of one film. Analysed together,\nso colours match across the roll."),
                  ("frames", QRectF(w / 2 + m / 2, m, w / 2 - 1.5 * m, h - 2 * m),
                   "Drop as single frames", "Unrelated pictures.\nEach one is analysed on its own.")]
        for key, r, title, text in halves:
            on = key == self.side
            p.setPen(QPen(QColor(ACCENT if on else "#55575d"), 3 if on else 2, Qt.DashLine))
            glow = QColor(ACCENT)
            glow.setAlpha(40)
            p.setBrush(glow if on else Qt.NoBrush)
            p.drawRect(r)
            f = QFont()
            f.setPixelSize(22)
            f.setWeight(QFont.Light)
            p.setFont(f)
            p.setPen(QColor(ACCENT if on else "#9a9ca1"))
            p.drawText(r.adjusted(0, 0, 0, -56), Qt.AlignCenter, title)
            f.setPixelSize(13)
            f.setWeight(QFont.Normal)
            p.setFont(f)
            p.drawText(r.adjusted(0, 70, 0, 0), Qt.AlignCenter, text)


class ExportDialog(QDialog):
    """Every export option in one place."""

    def __init__(self, parent, opts: engine.ExportOptions, extra: dict, selected: int, has_roll: bool):
        """`selected`: how many frames are selected; offered as "Selected (n)" when there are any.
        extra["scope"] picks the choice shown first."""
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(560)
        form = QFormLayout()
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)   # macOS keeps text fields narrow otherwise

        self.scope = exif_panel.ChevronCombo()
        self.scope.addItem("All groups", "all")
        self.scope.addItem("Current group", "group")
        if selected:
            self.scope.addItem(f"Selected ({selected})", "selection")
        self.scope.setCurrentIndex(max(self.scope.findData(extra.get("scope", "all")), 0))
        form.addRow("Export", self.scope)

        self.fmt = exif_panel.ChevronCombo()
        for key, (label, _ext) in engine.EXPORT_FORMATS.items():
            self.fmt.addItem(label, key)
        self.fmt.setCurrentIndex(max(self.fmt.findData(opts.fmt), 0))
        form.addRow("Format", self.fmt)

        size_row = QHBoxLayout()
        self.size = exif_panel.ChevronCombo()
        self.size.addItem("Full resolution", "original")
        self.size.addItem("Long edge in pixels", "long_edge")
        self.size.addItem("Print size", "print")
        self.size.setCurrentIndex(max(self.size.findData(opts.size), 0))
        self.long_edge = QSpinBox()
        self.long_edge.setRange(256, 20000)
        self.long_edge.setSingleStep(256)
        self.long_edge.setSuffix(" px")
        self.long_edge.setValue(opts.long_edge)
        self.print_cm = DecimalSpinBox()
        self.print_cm.setRange(5, 300)
        self.print_cm.setSuffix(" cm long side")
        self.print_cm.setValue(opts.print_cm)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setSuffix(" dpi")
        self.dpi.setValue(opts.dpi)
        for w in (self.size, self.long_edge, self.print_cm, self.dpi):
            size_row.addWidget(w)
        form.addRow("Size", size_row)

        self.color = exif_panel.ChevronCombo()
        for key, label in engine.COLOR_SPACES.items():
            self.color.addItem(label, key)
        self.color.setCurrentIndex(max(self.color.findData(opts.color_space), 0))
        form.addRow("Colour space", self.color)

        self.pattern = QLineEdit(opts.pattern)
        self.example = QLabel()
        self.example.setObjectName("hint")
        tokens = QLabel(engine.NAME_TOKENS)
        tokens.setObjectName("hint")
        tokens.setWordWrap(True)
        form.addRow("File name", self.pattern)
        form.addRow("", tokens)
        form.addRow("", self.example)

        self.overwrite = exif_panel.ChevronCombo()
        self.overwrite.addItem("Keep both (add a number)", False)
        self.overwrite.addItem("Overwrite", True)
        self.overwrite.setCurrentIndex(1 if opts.overwrite else 0)
        form.addRow("If the file exists", self.overwrite)

        self.per_roll = QCheckBox("One subfolder per roll")
        self.per_roll.setChecked(extra.get("per_roll", True))
        self.per_roll.setEnabled(has_roll)
        form.addRow("", self.per_roll)

        dest_row = QHBoxLayout()
        self.dest = QLineEdit(extra.get("dest", ""))
        choose = button("Choose…", self._choose_dest)
        dest_row.addWidget(self.dest, 1)
        dest_row.addWidget(choose)
        form.addRow("Save to", dest_row)

        self.reveal = QCheckBox(f"Show in {FILE_MANAGER} when done")
        self.reveal.setChecked(extra.get("reveal", True))
        form.addRow("", self.reveal)

        self.border = QCheckBox(f"Export with a white border ({pref('border_px')} px)")
        self.border.setChecked(pref("border"))
        self.border.setToolTip("The width is set in Settings. DNG files are exported without a border")
        form.addRow("", self.border)

        # Never remembered: deleting files must be chosen each time
        self.trash = QCheckBox(f"Move the original scans to the {TRASH_NAME} after export")
        self.trash.setChecked(False)
        form.addRow("", self.trash)
        self.trash_hint = QLabel(f"Only frames that export without an error. You can put them back from the "
                                 f"{TRASH_NAME}. The frames leave the list.")
        self.trash_hint.setObjectName("hint")
        self.trash_hint.setWordWrap(True)
        self.trash_hint.setVisible(False)
        self.trash.toggled.connect(self.trash_hint.setVisible)
        self.trash.toggled.connect(lambda _on: QTimer.singleShot(0, self.adjustSize))   # room for the hint
        form.addRow("", self.trash_hint)

        self.note = QLabel("")
        self.note.setObjectName("hint")
        self.note.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = button("Cancel", self.reject)
        go = button("Export", self.accept)
        go.setDefault(True)
        go.setObjectName("primary")
        buttons.addWidget(cancel)
        buttons.addWidget(go)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.note)
        lay.addLayout(buttons)

        for sig in (self.size.currentIndexChanged, self.fmt.currentIndexChanged, self.color.currentIndexChanged,
                    self.pattern.textChanged):
            sig.connect(self._update)
        self._sample = extra.get("sample", {"name": "P1010264", "roll": "Roll 1", "film": "Kodak Gold 200",
                                            "frame": "01", "camera": "Nikon FM2"})
        self._update()

    def _choose_dest(self):
        folder = QFileDialog.getExistingDirectory(self, "Export to folder", self.dest.text())
        if folder:
            self.dest.setText(folder)

    def _update(self):
        mode = self.size.currentData()
        self.long_edge.setVisible(mode == "long_edge")
        self.print_cm.setVisible(mode == "print")
        self.dpi.setVisible(mode == "print")
        fmt = self.fmt.currentData()
        ext = engine.EXPORT_FORMATS[fmt][1]
        name = engine.file_name(self.pattern.text(), self._sample) + ("_lossless" if fmt == "dng16" else "")
        self.example.setText(f"Example: {name}.{ext}")
        dng = fmt in ("dng", "dng16")
        self.color.setEnabled(not dng)
        jxl_adobe = fmt == "jxl" and self.color.currentData() == "Adobe RGB"
        self.note.setText("The best DNG for most uses: about the size of the original raw, and your raw editor "
                          "treats it like one. Always stored in linear Adobe RGB."
                          if fmt == "dng" else
                          "Exact 16-bit data, but about three times larger: roughly 100 MB for a 24 MP scan. "
                          "The compact DNG is the better choice unless you need every bit. Always stored in linear Adobe RGB."
                          if fmt == "dng16" else
                          "JPEG XL cannot hold Adobe RGB: these files are saved in Display P3 instead." if jxl_adobe else "")

    def options(self) -> tuple[engine.ExportOptions, dict]:
        opts = engine.ExportOptions(fmt=self.fmt.currentData(), size=self.size.currentData(),
                                    long_edge=self.long_edge.value(), print_cm=self.print_cm.value(),
                                    dpi=self.dpi.value(), color_space=self.color.currentData(),
                                    pattern=self.pattern.text(), overwrite=self.overwrite.currentData(),
                                    border_px=pref("border_px") if self.border.isChecked() else 0)
        extra = {"scope": self.scope.currentData(), "per_roll": self.per_roll.isChecked(),
                 "dest": self.dest.text().strip(), "reveal": self.reveal.isChecked(),
                 "trash": self.trash.isChecked()}
        return opts, extra


class ContactSheetDialog(QDialog):
    """A roll's contact sheet: every frame on one sheet of paper, as it would export."""

    def __init__(self, parent, roll: str, frames: int, opts: engine.SheetOptions, dest: str, reveal: bool):
        super().__init__(parent)
        self.setWindowTitle("Contact Sheet")
        self.setMinimumWidth(560)
        form = QFormLayout()
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)   # macOS keeps text fields narrow otherwise

        what = QLabel(f"{roll}  ·  {frames} frame{'s' if frames != 1 else ''}")
        what.setObjectName("hint")
        form.addRow("Roll", what)

        def combo(items: dict, current):
            box = exif_panel.ChevronCombo()
            for key, label in items.items():
                box.addItem(label, key)
            box.setCurrentIndex(max(box.findData(current), 0))
            return box

        self.paper = combo({k: v[0] for k, v in engine.SHEET_PAPERS.items()}, opts.paper)
        form.addRow("Paper", self.paper)
        self.film = combo(engine.SHEET_FILMS, opts.film)
        form.addRow("Film", self.film)
        self.style = combo({True: "Strips of film, with perforations and edge print",
                            False: "Upright frames in rows"}, opts.film_edges)
        form.addRow("Look", self.style)
        self.white = combo({False: "Black", True: "White (no ink around the film)"}, opts.white_paper)
        form.addRow("Around the film", self.white)
        self.dpi = combo(engine.SHEET_DPI, opts.dpi)
        form.addRow("Quality", self.dpi)
        self.label = QCheckBox("Print the roll's name, film, camera and date above the frames")
        self.label.setChecked(opts.label)
        form.addRow("", self.label)

        dest_row = QHBoxLayout()
        self.dest = QLineEdit(dest)
        dest_row.addWidget(self.dest, 1)
        dest_row.addWidget(button("Choose…", self._choose_dest))
        form.addRow("Save to", dest_row)
        self.reveal = QCheckBox(f"Show in {FILE_MANAGER} when done")
        self.reveal.setChecked(reveal)
        form.addRow("", self.reveal)

        note = QLabel("Saved as JPEG (sRGB). The frames are printed with their adjustments, in the roll's order; "
                      "a roll that does not fit on one sheet continues on the next.")
        note.setObjectName("hint")
        note.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        go = button("Make Contact Sheet", self.accept)
        go.setDefault(True)
        go.setObjectName("primary")
        buttons.addWidget(button("Cancel", self.reject))
        buttons.addWidget(go)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(note)
        lay.addLayout(buttons)

    def _choose_dest(self):
        folder = QFileDialog.getExistingDirectory(self, "Save the contact sheet to", self.dest.text())
        if folder:
            self.dest.setText(folder)

    def options(self) -> tuple[engine.SheetOptions, str, bool]:
        opts = engine.SheetOptions(paper=self.paper.currentData(), film=self.film.currentData(),
                                   film_edges=bool(self.style.currentData()), white_paper=bool(self.white.currentData()),
                                   label=self.label.isChecked(), dpi=int(self.dpi.currentData()))
        return opts, self.dest.text().strip(), self.reveal.isChecked()


class MainWindow(QMainWindow):
    TOOLTIPS = {
        "highlight_tone": "The bright parts: left brings back detail in skies and skin, right brightens them",
        "shadow_tone": "The dark parts: right opens up detail in the shadows, left deepens them",
        "temp": "White balance like a raw file's: cooler (blue) ← → warmer (amber)",
        "tint": "White balance like a raw file's: green ← → magenta",
        "glow": "A soft glow around bright areas, like the soft focus of a minilab scan",
        "sharpen": "How crisp fine detail is. Double-click to go back to the default",
        "color_noise": "Smooths coloured speckle in the grain. More can wash out fine colour detail",
        "exposure": "Exposure in stops, like a raw converter's: right brightens shadows and midtones while "
                    "highlights roll off to white; left darkens everything",
        "white_point": "The lightest tones: left dims them, right brightens them toward white",
        "black_point": "The darkest tones: left deepens them toward black, right lifts them",
    }
    STEPPERS = ["contrast", "density", "saturation", "cyan", "magenta", "yellow"]

    def __init__(self):
        super().__init__()
        engine.USE_GPU = pref("gpu")
        self._wheel_guard = WheelGuard(self)
        QApplication.instance().installEventFilter(self._wheel_guard)
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 950)
        self.setAcceptDrops(True)

        self.groups: list[Group] = []
        self.group: Group | None = None          # the group shown in the thumbnail grid
        self.photos: list[Photo] = []            # = self.group.photos (the grid's rows)
        self.current: Photo | None = None
        # Vignetting fix: one setting for every image (it comes from the scanning setup)
        self.flat_field: str | None = None

        # Thumbnails + exports: several threads. Preview: exactly one thread, so
        # renders never pile up; while one runs, only the latest request waits.
        self.pool = QThreadPool.globalInstance()
        self.preview_pool = QThreadPool()
        self.preview_pool.setMaxThreadCount(1)
        # Threads are kept for the app's lifetime: each owns a NegPy engine (engine._processor),
        # and a thread that expired and came back would leave its old engine behind.
        self.pool.setExpiryTimeout(-1)
        self.preview_pool.setExpiryTimeout(-1)
        # Exports run on their own few threads, never on the ones the thumbnails use. How many
        # really render at once is decided by memory (engine._export_memory): big scans take turns.
        import os
        self.export_pool = QThreadPool()
        self.export_pool.setMaxThreadCount(max(1, min(4, (os.cpu_count() or 4) // 2)))
        self.export_pool.setExpiryTimeout(-1)
        self.preview_busy = False
        self.preview_pending = False
        self.preview_request = 0

        self.signals = WorkerSignals()
        self.signals.thumb.connect(self.on_thumb)
        self.signals.uv.connect(self.on_uv)
        self.signals.hires.connect(self.on_hires)
        self.zoom_pool = QThreadPool()                # full-resolution renders for zooming, one at a time
        self.zoom_pool.setMaxThreadCount(1)
        self.zoom_pool.setExpiryTimeout(-1)
        self._hires_key = None                        # what the full-resolution render on screen shows
        self._hires_busy = False
        self._src_long: dict = {}                     # scan size per file (to know where 100% is)
        self.heal_size = 24.0                     # brush diameter, px at NegPy's 1600 px reference
        self._uv = None                           # (photo, geometry key, grid) for mapping clicks
        self._uv_busy = False
        self._heal_queue: list = []               # clicks made while the grid was being measured
        self._area_queue: list = []               # shapes drawn meanwhile
        self._locked: dict = {}                   # what was enabled before a tool took the window over
        self._key_tips: list = []                 # (widget, tooltip, action): tooltips that name a key
        self.signals.preview.connect(self.on_preview)
        self.signals.wb.connect(self.on_wb)
        self.signals.exported.connect(self.on_exported)
        self.signals.sheet.connect(self.on_sheet)
        self.signals.sheet_progress.connect(self.on_sheet_progress)
        self.signals.roll.connect(self.on_roll_analyzed)
        self.signals.roll_progress.connect(self.on_roll_progress)
        self.roll_progress = RollProgress()
        self.statusBar().addWidget(self.roll_progress)

        # While a control is moving: quick draft renders, back to back. Once it rests for a
        # moment: one full-quality render (and the thumbnails of other selected frames).
        self.render_timer = QTimer(singleShot=True, interval=30)
        self.render_timer.timeout.connect(lambda: self.request_preview(draft=True))
        # sharp only once you really stop: switching on every hesitation of a drag looked like a strobe
        self._settle_timer = QTimer(singleShot=True, interval=450)
        self._settle_timer.timeout.connect(self._settled)
        self._thumb_timer = QTimer(singleShot=True, interval=120)
        self._thumb_timer.timeout.connect(self._flush_thumbs)
        self._thumb_busy: set = set()             # frames whose thumbnail is rendering
        self._thumb_again: set = set()            # ...and changed meanwhile
        self._pending_full = False
        self._dirty_thumbs: set = set()
        self._thumb_timer = QTimer(singleShot=True, interval=600)
        self._thumb_timer.timeout.connect(self._refresh_all_thumbs)
        self._reanalyze_timer = QTimer(singleShot=True, interval=900)
        self._reanalyze_timer.timeout.connect(self._reanalyze_rolls)
        # an edit becomes one undo step once the clicking/dragging pauses
        self._pick_target = "wb"                  # what the next click on the photo picks (toggle_pick)
        self._history_timer = QTimer(singleShot=True, interval=500)
        self._history_timer.timeout.connect(self._commit_history)
        # Grouped undo: a change that reached several frames is one step on all of them
        self._pending_batch: set = set()          # frames changed with the current one, committed with it
        self._batches: list[dict] = []            # {"frames": {photo: its step}, "seq": ...}, newest last
        self._redo_batches: list[dict] = []
        self._history_seq = 0                     # counts every new undo step, on any frame
        self._seq_now = 0                         # where undo stands in that count (undo of a group moves it back)
        self._sync_base: engine.Settings | None = None    # for applying edits to selected frames
        self._show_before = False
        self._show_original = False               # the scan before conversion (Original / its key)
        self._adjusting: str | None = None        # Settings field changed by ↑ / ↓ while its key is held
        self._rendered_angle = 0.0                # fine rotation of the picture on screen
        self._request_geom: dict = {}             # render id -> (turns, mirror) it was asked with
        self._prefetched: set = set()             # frames whose viewer decode is prepared for a quick switch
        self._fade_next = False                   # the next render is a new view (crop on/off, Original): fade it in
        self._request_key: dict = {}              # render id -> what it renders (see _render_key)
        self._shown_key = None                    # what the picture on screen is a render of
        self._request_angle: dict[int, float] = {}
        self._request_draft: dict[int, bool] = {}
        self._straighten_timer = QTimer(singleShot=True, interval=250)   # real render once dragging pauses
        self._straighten_timer.timeout.connect(self.request_preview)
        self._clipboard: engine.Settings | None = None
        self._fullscreen = False
        self._was_maximized = False
        self._recovery_on = True                      # off while another running window owns the file
        self._recovery_text = ""
        self._saved_text = ""                         # the session as last saved by hand ("" = never)
        self._recovery_timer = QTimer(interval=4000)
        self._recovery_timer.timeout.connect(self._write_recovery)
        self._stale_thumb: Photo | None = None    # frame whose filmstrip picture lags behind an edit
        self.export_opts = engine.ExportOptions()
        self.export_extra: dict = {}
        try:                                          # the options of your last export
            import json
            saved = json.loads(pref("export") or "{}")
            fields_ = engine.ExportOptions.__dataclass_fields__
            self.export_opts = engine.ExportOptions(**{k: v for k, v in saved.get("opts", {}).items() if k in fields_})
            self.export_extra = {k: saved[k] for k in ("per_roll", "reveal") if k in saved}
        except (ValueError, TypeError):
            pass
        self._export_total = self._export_left = self._export_errors = 0
        self._export_failed: list[str] = []
        self._to_trash: list = []                    # exported frames whose scans go to the Trash at the end
        self._export_dir = ""
        self._sheets: dict = {}                       # roll -> (reveal in Finder?) while its contact sheet is made
        self.signals.straighten.connect(self.on_straightened)
        self.signals.level.connect(self.on_levelled)
        self.signals.autocrop.connect(self.on_auto_crop_found)
        self._autocrop_asked: dict = {}           # photo -> the framing key a search is running for
        self._levelling: set = set()              # frames whose Auto crop levelling is still being measured
        self._crop_seed: QRectF | None = None     # the auto crop the crop tool opened on (unchanged = still Auto)
        self._wb_running = 0                      # white balances still being measured
        self.signals.note.connect(self.on_note)

        self._build_ui()
        self.update_nav()
        self._build_keys()

    # ----- Keyboard workflow -------------------------------------------------
    def _build_keys(self):
        """Keyboard shortcuts. Single keys come from the user's bindings (keys.py, "Keys" tab);
        the Cmd shortcuts and Esc are fixed.

        QShortcut works anywhere in the window. Text fields take letters first,
        so typing a camera name never triggers a correction.
        """
        self._fixed_shortcuts = []
        for seq, fn in [(QKeySequence.Copy, self.copy_settings), (QKeySequence.Paste, self.paste_settings),
                        (QKeySequence.Undo, self.undo), (QKeySequence.Redo, self.redo),
                        ("Ctrl+Shift+Z", self.redo), ("Ctrl+E", self.open_export), (QKeySequence.Save, self.save_session),
                        (QKeySequence.SelectAll, self.grid.selectAll), (Qt.Key_Escape, self.escape),
                        ("Ctrl+Backspace", self.remove_selected_frames)]:
            sc = QShortcut(QKeySequence(seq), self, activated=fn)
            sc.setProperty("aid", {self.escape: "escape", self.undo: "undo", self.redo: "redo"}.get(fn, ""))
            self._fixed_shortcuts.append(sc)
        self._shortcuts: list[QShortcut] = []
        self.bindings = keys.load()
        self._apply_bindings()
        self._refresh_tips()                       # the window was built before the keys were known
        # "Before" is held, not pressed, so it is watched as raw key events (see eventFilter)
        QApplication.instance().installEventFilter(self)

    def _action_functions(self) -> dict:
        """Action id (keys.ACTIONS) -> (function, function for Shift + the key, or None)."""
        fns = {
            "accept": (self.accept_frame, None),
            "next": (lambda: self.step_image(+1), None),
            "prev": (lambda: self.step_image(-1), None),
            "next_roll": (lambda: self.step_group(+1), None),
            "prev_roll": (lambda: self.step_group(-1), None),
            "first": (lambda: self.go_to_frame(0), None),
            "last": (lambda: self.go_to_frame(len(self.photos) - 1), None),
            "auto_wb": (self.auto_wb, None),
            "pick_wb": (self.toggle_pick_wb, None),
            "heal": (self.toggle_heal, None),
            "heal_area": (self.toggle_heal_area, None),
            "zoom": (self.toggle_zoom_key, None),
            "fullscreen": (self.toggle_fullscreen, None),
            "reset": (self.reset_current, None),
            "rotate_left": (lambda: self.rotate(+1), None),
            "rotate_right": (lambda: self.rotate(-1), None),
            "mirror": (self.mirror, None),
            "tilt_left": (lambda: self.nudge_straighten(+0.1), lambda: self.nudge_straighten(+1.0)),
            "tilt_right": (lambda: self.nudge_straighten(-0.1), lambda: self.nudge_straighten(-1.0)),
            "auto_straighten": (self.auto_straighten, None),
            "crop": (self.toggle_crop, None),
            "auto_crop": (lambda: self.current and self.autocrop_cb.toggle(), None),
            "auto_crop_refresh": (self.refresh_auto_crop, None),
            "sync_roll": (self.sync_roll, None),
            "border": (self.toggle_border, None),
            "original": (self.toggle_original, None),
            "copy": (self.copy_settings, None),
            "paste": (self.paste_settings, None),
            "undo": (self.undo, None),
            "redo": (self.redo, None),
            "export": (self.open_export, None),
            "contact_sheet": (self.open_contact_sheet, None),
            "keys": (lambda: self.set_tab(2), None),
        }
        for field_ in self.STEPPERS:               # density_up / density_down, ...
            for word, d in (("up", +1), ("down", -1)):
                fns[f"{field_}_{word}"] = (lambda f=field_, d=d: self.nudge(f, d),
                                           lambda f=field_, d=d: self.nudge(f, 4 * d))
        return fns

    def _apply_bindings(self):
        """(Re)create one QShortcut per bound key; called at start and after every change in the tab."""
        for sc in self._shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._shortcuts = []
        fns = self._action_functions()

        def guarded(fn, aid):
            # in a text field, keys type; only "next" handles Return there (it finishes the typing)
            return lambda: fn() if aid == "accept" or not self._typing() else None

        for name, aid in self.bindings.items():
            action = keys.ACTION_BY_ID[aid]
            if action.hold:
                continue
            fn, shift_fn = fns[aid]
            seqs = [(name, fn)] + ([("Shift+" + name, shift_fn)] if shift_fn else [])
            if name == "Return":                   # the keypad's Enter does the same as Return
                seqs.append(("Enter", fn))
            for seq, f in seqs:
                sc = QShortcut(QKeySequence(seq), self, activated=guarded(f, aid))
                sc.setAutoRepeat(action.repeat)    # holding a correction key keeps nudging
                sc.setProperty("aid", aid)
                self._shortcuts.append(sc)

    def key_for(self, aid: str) -> str:
        """The key an action is on right now ("" if none), as it is written on the Keys page."""
        return next((keys.pretty(k) for k, a in self.bindings.items() if a == aid), "")

    def _tip(self, widget: QWidget, text: str, aid: str):
        """A tooltip that ends with its action's key, kept right when keys are rebound."""
        self._key_tips.append((widget, text, aid))
        key = self.key_for(aid) if hasattr(self, "bindings") else ""
        widget.setToolTip(f"{text} ({key})" if key else text)

    def _refresh_tips(self):
        for widget, text, aid in self._key_tips:
            key = self.key_for(aid)
            widget.setToolTip(f"{text} ({key})" if key else text)

    def on_bindings_changed(self, bindings: dict):
        self.bindings = bindings
        keys.save(bindings)
        self._apply_bindings()
        self._refresh_tips()
        self.set_tab(2)                            # keep shortcuts off while still on the bindings tab

    def set_tab(self, index: int):
        """0 = Convert, 1 = Exif (same page, other right column), 2 = Keys, 3 = Settings.
        Shortcuts are off on the Keys and Settings tabs, so keys can be picked and text typed there."""
        working = index in (0, 1)
        self.main_stack.setCurrentIndex({2: 1, 3: 2}.get(index, 0))
        self.right_stack.setCurrentIndex(1 if index == 1 else 0)
        self._update_toolbars()
        self.tab_buttons[index].setChecked(True)
        self.export_wrap.setVisible(bool(self.groups) and working)
        keep = self.title_box.sizePolicy()
        keep.setRetainSizeWhenHidden(True)            # the tab row keeps its layout
        self.title_box.setSizePolicy(keep)
        self.title_box.setVisible(working)            # the photo's name only over the photo
        for sc in self._shortcuts + self._fixed_shortcuts:
            sc.setEnabled(working)
        (self.bindings_page if index == 2 else self.settings_page if index == 3 else self.preview).setFocus()
        if index == 1:
            self.refresh_exif()

    def _update_toolbars(self):
        """Under the photo: the Convert actions, or the EXIF copy / paste actions on the Exif tab."""
        loaded = bool(self.groups) and not self._fullscreen
        exif = self.right_stack.currentIndex() == 1
        cropping = self.preview.crop_mode
        self.toolbar.setVisible(loaded and not exif and not cropping)
        self.toolbar_crop.setVisible(loaded and not exif and cropping)
        self.toolbar_exif.setVisible(loaded and exif)

    def _typing(self) -> bool:
        """True while a text field or list (e.g. film stock search) has the keyboard."""
        w = QApplication.focusWidget()
        # an editable combo box (film stock) keeps the focus itself and hands the keys to its line edit
        editable_combo = isinstance(w, QComboBox) and w.isEditable()
        return isinstance(w, (QLineEdit, QAbstractSpinBox)) or editable_combo or (w is not None and w.window() is not self)

    # what ↑ / ↓ mean while each adjust key is held
    ADJUST_HINTS = {"density": ("darker", "lighter"), "cyan": ("more cyan", "more red"),
                    "magenta": ("more magenta", "more green"), "yellow": ("more yellow", "more blue"),
                    "contrast": ("more contrast", "less contrast"), "saturation": ("more saturation", "less saturation")}

    def eventFilter(self, obj, e):
        """Keys that act while held: "Before", and the adjust keys (hold C, then ↑ / ↓).

        Qt first asks the focused widget whether it wants a key itself (ShortcutOverride);
        accepting that keeps ↑ / ↓ from also triggering their own shortcut (next roll...).
        """
        t = e.type()
        if t == QEvent.ApplicationDeactivate:
            self._set_adjusting(None)
        if t not in (QEvent.KeyPress, QEvent.KeyRelease, QEvent.ShortcutOverride):
            return super().eventFilter(obj, e)
        if (self.main_stack.currentIndex() != 0 or QApplication.activeWindow() is not self or self._typing()
                or self.preview.crop_mode or self.preview.heal_mode or self.preview.area_mode):
            return super().eventFilter(obj, e)

        if self._adjusting and e.key() in (Qt.Key_Up, Qt.Key_Down):
            if t == QEvent.ShortcutOverride:
                e.accept()
            elif t == QEvent.KeyPress:
                steps = 4 if e.modifiers() & Qt.ShiftModifier else 1
                self.nudge(self._adjusting, steps if e.key() == Qt.Key_Up else -steps)
            return True

        action = keys.ACTION_BY_ID.get(self.bindings.get(keys.key_name(e.key()), ""))
        if not action or not action.hold or e.modifiers() & ~Qt.ShiftModifier:
            return super().eventFilter(obj, e)
        if t == QEvent.ShortcutOverride:
            e.accept()
            return True
        if e.isAutoRepeat():
            return True
        down = t == QEvent.KeyPress
        if action.adjust:
            self._set_adjusting(action.adjust if down else None)
        elif action.id == "before" and self.current and down != self._show_before:
            self._set_before(down)
            self.statusBar().showMessage("Before: the automatic conversion" if down else "", 0 if down else 1)
        return True

    def _set_adjusting(self, field_: str | None):
        if field_ == self._adjusting:
            return
        if self._adjusting:
            self.steppers[self._adjusting].set_active(False)
        self._adjusting = field_
        if field_:
            self.steppers[field_].set_active(True)
            up, down = self.ADJUST_HINTS[field_]
            self.statusBar().showMessage(f"{field_.capitalize()}:  ↑ {up}   ↓ {down}   (Shift: bigger steps)")
        else:
            self.statusBar().clearMessage()

    def nudge(self, field_: str, steps: int):
        if not self.current or self._typing():
            return
        s = self.steppers[field_]
        s._set(s.value() + steps * s.step, emit=True)   # same path as clicking + / -

    def nudge_straighten(self, degrees: float):
        if not self.current or self._typing():
            return
        self.straighten.set_value(max(-15.0, min(15.0, self.current.settings.fine_rotation + degrees)), emit=True)
        self.statusBar().showMessage(f"Straighten {self.current.settings.fine_rotation:+.1f}°", 1500)

    def accept_frame(self):
        """Space / Return: done with this frame, on to the next (and into the next roll)."""
        if self._typing():                      # Return in a text field: finish typing instead
            self.escape()
            return
        if self.preview.crop_mode:              # Return while cropping applies the crop
            self.toggle_crop()
            return
        row = self.grid.currentRow()
        if row < len(self.photos) - 1:
            self.step_image(+1)
        elif self.group in self.groups and self.groups.index(self.group) < len(self.groups) - 1:
            self.step_group(+1)
            self.statusBar().showMessage(f"Next roll: {self.group.name}", 2500)

    def step_group(self, direction: int):
        if not self.group or self._typing():
            return
        i = self.groups.index(self.group) + direction
        if 0 <= i < len(self.groups):
            self.show_group(self.groups[i])

    def go_to_frame(self, row: int):
        if 0 <= row < len(self.photos) and not self._typing():
            self.grid.setCurrentRow(row, QItemSelectionModel.ClearAndSelect)   # just this frame

    def toggle_fullscreen(self):
        """F: only the photo, on black, across the whole screen. F or Esc brings the window back.
        Every key still works there (next frame, corrections, zoom...)."""
        if not self.current and not self._fullscreen:
            return
        on = not self._fullscreen
        self._fullscreen = on
        if on:
            self._was_maximized = self.isMaximized()
            self.set_tab(0)
        self.backdrop.black = on
        self.tab_row.setVisible(not on)
        self.title.setVisible(not on)
        self.statusBar().setVisible(not on)
        self.right_stack.setVisible(not on and bool(self.groups))
        self.grid.setVisible(not on and bool(self.groups))
        self._root.setContentsMargins(*((0, 0, 0, 0) if on else self._root_margins))
        if on:
            self.left.setVisible(False)
            self.toolbar.setVisible(False)
            self.toolbar_exif.setVisible(False)
            self.showFullScreen()
        else:
            self._update_toolbars()
            self.showMaximized() if self._was_maximized else self.showNormal()
            self._fit_left_column()
        self.preview.full_view = on
        self.preview.reset_zoom()
        self.preview.setFocus()
        self.backdrop.update()
        if on:
            QTimer.singleShot(150, self._check_sharp_fit)   # once the window has its full-screen size
        else:
            self.preview.fit_hires = None
            if self.preview.hires is not None or self._hires_key is not None:
                self.preview.hires, self._hires_key = None, None
                self.zoom_pool.start(Job(engine.release_cache, lambda r: None))   # free the full decode

    def _sharp_fit_wanted(self) -> bool:
        """Full view shows the whole picture larger than the preview has pixels for: render it in full."""
        pv, p = self.preview, self.current
        if not (self._fullscreen and p and pv.source and pv.zoom <= 1.0) or pv.crop_mode \
                or self._show_original or self._show_before:
            return False
        r = pv.view_rect()
        return max(r.width(), r.height()) * pv.devicePixelRatioF() > max(pv.source.width(), pv.source.height()) * 1.02

    def _check_sharp_fit(self):
        if self._sharp_fit_wanted() and self._hires_key != self._settings_key(self.current):
            self._request_hires()

    def escape(self):
        if self._fullscreen and not (self.preview.heal_mode or self.preview.picking or self.preview.crop_mode
                                     or self.preview.area_mode):
            self.toggle_fullscreen()
            return
        if self.preview.heal_mode:
            self.toggle_heal()
        if self.preview.area_mode:
            self.toggle_heal_area()
        if self.preview.picking:
            self.stop_pick()
        if self.preview.crop_mode:
            self.toggle_crop(apply=False)
        self.preview.setFocus()                 # leaves a text field; keys drive the scanner again

    # ----- UI construction -------------------------------------------------
    def _build_ui(self):
        # ---- Bottom: the filmstrip, one row of frames (self.grid)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setFlow(QListWidget.LeftToRight)
        self.grid.setWrapping(False)                  # a single row that scrolls sideways
        self.grid.setMovement(QListWidget.Static)
        self.grid.setIconSize(QSize(THUMB_SIZE, THUMB_H))
        self.grid.setGridSize(QSize(THUMB_SIZE + 22, THUMB_H + 18))   # room for the selected frame to grow
        self.grid.setFixedHeight(THUMB_H + 18 + 8 + 7 + 2)   # one row + padding + scrollbar, nothing more
        thumbs = ThumbDelegate(self.grid)
        thumbs.edited = self._frame_edited
        self.grid.setItemDelegate(thumbs)
        self.grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._frame_menu)
        self.grid.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.grid.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Cmd/Shift-click
        self._strip_guard = StripGuard(self.grid, self._center_current)
        self.grid.viewport().installEventFilter(self._strip_guard)
        self.grid.setObjectName("grid")
        self.grid.currentRowChanged.connect(self.on_select)
        self.grid.itemSelectionChanged.connect(lambda: self.refresh_exif())

        self.cards = QWidget()                        # one RollCard per roll (and one for single frames)
        self.cards_lay = QVBoxLayout(self.cards)
        self.cards_lay.setContentsMargins(0, 0, 0, 0)
        self.cards_lay.setSpacing(8)
        self.cards_lay.addStretch(1)
        cards_scroll = self.cards_scroll = QScrollArea()
        cards_scroll.setWidget(self.cards)
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cards_scroll.setFrameShape(QFrame.NoFrame)
        self._details_open = True                     # roll details expanded (remembered across rolls)

        self.roll_box = self._build_roll_details()     # lives inside the selected roll's card
        left_lay = QVBoxLayout()
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(12)
        hdr = QLabel("ROLLS & FRAMES")
        hdr.setObjectName("header")
        self.add_btn = IconButton(_icon_plus, None, self._add_menu, "Add a roll (a folder) or single frames")
        self.add_btn.setMinimumHeight(0)
        self.add_btn.setFixedSize(26, 24)
        self.save_btn = SaveButton(self.save_session)
        self.save_btn.setMinimumHeight(0)
        self.save_btn.setFixedSize(38, 28)            # in the tab bar, after the Settings gear
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(hdr)
        hdr_row.addStretch(1)
        hdr_row.addWidget(self.add_btn)
        left_lay.addLayout(hdr_row)
        left_lay.addWidget(cards_scroll, 1)
        self.left = QWidget()
        self.left.setLayout(left_lay)
        self.left.setFixedWidth(280)

        # ---- Center: title, preview, Edit Crop
        self.title = QLabel("")
        self.title.setObjectName("title")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # long names never widen the window
        self.title_info = InfoDot()
        self.title_box = QWidget()                    # in the top row, over the photo: the name and its details
        tl = QHBoxLayout(self.title_box)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(2)
        tl.addStretch(1)
        tl.addWidget(self.title, 0, Qt.AlignVCenter)
        tl.addWidget(self.title_info, 0, Qt.AlignVCenter)
        tl.addStretch(1)
        self.title_info.hide()
        self.preview = PreviewLabel()
        self.preview.setFocusPolicy(Qt.ClickFocus)
        self.toast = Toast(self.preview)
        self.preview.picked.connect(self.on_picked)
        self.preview.heal_clicked.connect(self.on_heal_click)
        self.preview.area_drawn.connect(self.on_area_drawn)
        self.preview.area_removed.connect(self.on_area_removed)
        self.preview.zoom_changed.connect(self.on_zoom)
        self.preview.brush_wheel.connect(self.on_brush_wheel)
        self.crop_btn = button("Edit", self.toggle_crop)
        self.stack = QStackedWidget()          # page 0: welcome, page 1: preview
        self.stack.addWidget(self._build_welcome())
        self.stack.addWidget(self.preview)
        self.toolbar = QWidget()
        tb = FlowLayout(self.toolbar, spacing=6)
        copy_btn = button("Copy", self.copy_settings, "tool")
        copy_btn.setToolTip(f"Copy this frame's adjustments ({keys.MOD}+C), to paste on other frames")
        self.paste_btn = button("Paste", self.paste_settings, "tool")
        self.paste_btn.setToolTip("Paste the copied adjustments on the selected frames (crop and rotation stay)")
        self.sync_btn = button("Sync", self.sync_roll, "tool")
        self._tip(self.sync_btn, "Sync Roll: send this frame's edits to every frame of the group (the arrow beside it picks which)",
                  "sync_roll")
        # Sync and its options are one split button: an outline holds both, the ▾ opens the choices
        self.sync_gear = IconButton(_icon_more, None, self._sync_choices, "Sync options: choose what Sync Roll sends")
        self.sync_gear.setObjectName("syncMore")
        self.sync_gear.setMinimumHeight(0)
        self.sync_gear.setFixedWidth(18)
        self.sync_gear.scale = 0.75
        self.sync_btn.setObjectName("syncMain")
        sync_split = QFrame()
        sync_split.setObjectName("syncSplit")
        split_row = QHBoxLayout(sync_split)
        split_row.setContentsMargins(0, 0, 0, 0)
        split_row.setSpacing(0)
        split_row.addWidget(self.sync_btn)
        split_div = QFrame()
        split_div.setObjectName("toolDiv")
        split_div.setFixedWidth(1)
        split_row.addWidget(split_div)
        split_row.addWidget(self.sync_gear)
        self.heal_btn = button("Heal", self.toggle_heal, "tool")
        self._tip(self.heal_btn, "Heal dust: click each speck to remove it; Alt+scroll changes the brush size", "heal")
        self.area_btn = button("Clean", self.toggle_heal_area, "tool")
        self._tip(self.area_btn, "Clean an area: draw around a part of the photo; dust and scratches inside it are removed", "heal_area")
        self.original_btn = button("Original", self.toggle_original, "tool")
        self._tip(self.original_btn, "The scan before conversion: the negative as your camera took it", "original")
        reset_btn = button("Reset", self.reset_current, "tool")
        self._tip(reset_btn, "Back to the automatic conversion", "reset")
        for b, draw in ((copy_btn, _icon_copy), (self.paste_btn, _icon_paste), (self.sync_btn, _icon_sync),
                        (self.heal_btn, _icon_heal), (self.area_btn, _icon_area), (self.original_btn, _icon_negative),
                        (reset_btn, _icon_reset)):
            b.setIcon(line_icon(draw))
            b.setIconSize(QSize(14, 14))
        self._bar_btns = (copy_btn, self.paste_btn, self.sync_btn, self.sync_gear, self.heal_btn, self.area_btn,
                          self.original_btn, reset_btn)
        tb.addWidget(tool_plate([[copy_btn, self.paste_btn, sync_split],
                                 [self.heal_btn, self.area_btn], [self.original_btn], [reset_btn]]))
        center = QVBoxLayout()
        center.addWidget(self.stack, 1)
        center.addWidget(self.toolbar)
        self.toolbar.setVisible(False)
        # While cropping, the strip under the photo offers film formats instead
        self.toolbar_crop = QWidget()
        tbc = FlowLayout(self.toolbar_crop, spacing=6)
        self.ratio_btns: dict[str, QPushButton] = {}
        for name, tip in self.CROP_FORMATS_TIPS.items():
            b = button(name, lambda n=name: self.set_crop_format(n), "tool")
            b.setCheckable(True)
            b.setToolTip(tip)
            self.ratio_btns[name] = b
        self.ratio_btns["Free"].setChecked(True)
        turn = IconButton(_icon_orientation, None, self.crop_turn, "Turn: portrait ↔ landscape")
        turn.setObjectName("tool")
        turn.setMinimumHeight(0)
        turn.setFixedSize(30, 26)
        reset = button("Reset", self.crop_restore, "tool")
        reset.setToolTip("Back to the frame's own crop: auto crop when new frames get it, else none")
        apply_ = button("Apply", self.toggle_crop, "tool")
        apply_.setToolTip(f"Apply the crop ({keys.RETURN}); Esc cancels")
        apply_.setProperty("go", True)               # Apply: the step that finishes
        tbc.addWidget(tool_plate([list(self.ratio_btns.values()), [turn, reset], [apply_]]))
        center.addWidget(self.toolbar_crop)
        self.toolbar_crop.setVisible(False)
        self.toolbar_exif = QWidget()                 # the Exif tab's actions under the photo
        tbe = FlowLayout(self.toolbar_exif, spacing=6)
        for text, slot, tip in (("Copy EXIF", self.exif_copy, "Copy this frame's EXIF"),
                                ("Paste to Selected", self.exif_paste, "Paste the copied EXIF onto the selected frames"),
                                ("Paste to Whole Roll", self.exif_paste_roll, "Paste the copied EXIF onto every frame of this roll"),
                                ("Set as Roll Defaults", self.exif_to_roll, "This frame's values become the roll's details"),
                                ("Clear Selected", self.exif_clear, "The selected frames go back to the roll's details")):
            b = button(text, slot, "bar")
            b.setToolTip(tip)
            tbe.addWidget(b)
        center.addWidget(self.toolbar_exif)
        self.toolbar_exif.setVisible(False)

        # ---- Right: the panels
        right = QVBoxLayout()
        right.setSpacing(14)

        p, lay = panel()                                  # the frame: turn, flip, crop, straighten
        lay.setSpacing(12)
        row = QHBoxLayout()
        row.setSpacing(6)                                 # three joined tiles
        for draw, arg, slot, tip in [(_icon_rotate, False, lambda: self.rotate(+1), "Rotate left"),
                                     (_icon_mirror, None, self.mirror, "Flip left to right"),
                                     (_icon_rotate, True, lambda: self.rotate(-1), "Rotate right")]:
            tile = IconButton(draw, arg, slot, tip)
            tile.setMinimumHeight(0)
            tile.setFixedHeight(26)
            row.addWidget(tile, 1)
        lay.addLayout(row)
        crop_label = QLabel("Crop")
        crop_label.setObjectName("rowLabel")
        self.autocrop_cb = button("Auto", None, "toggle")  # a switch: neon while on
        self.autocrop_cb.setCheckable(True)
        self._tip(self.autocrop_cb, "Find the picture edges and crop the film border away", "auto_crop")
        self.autocrop_cb.toggled.connect(self.on_autocrop)
        self.autocrop_refresh = IconButton(_icon_autocrop, None, self.refresh_auto_crop)
        self._tip(self.autocrop_refresh, "Find the picture edges again (after turning or straightening)",
                  "auto_crop_refresh")
        self.autocrop_refresh.setEnabled(False)
        self.autocrop_cb.toggled.connect(self.autocrop_refresh.setEnabled)
        self.crop_btn.setObjectName("toggle")
        self._tip(self.crop_btn, "Crop the photo", "crop")
        crop_row = QHBoxLayout()
        crop_row.setSpacing(6)
        crop_row.addWidget(crop_label)
        crop_row.addStretch(1)
        crop_row.addWidget(self.autocrop_refresh)
        crop_row.addWidget(self.autocrop_cb)
        crop_row.addWidget(self.crop_btn)
        lay.addLayout(crop_row)
        self.straighten = LabeledSlider("Straighten", -15.0, 15.0, 0.0)
        self.straighten.setToolTip("Small rotation to level the picture; empty corners are cropped away")
        self.straighten.changed.connect(self.on_straighten_slider)
        self.straighten.pressed.connect(lambda: self._show_grid(True))
        self.straighten.released.connect(self._straighten_released)
        self.auto_straighten_btn = button("Auto", self.auto_straighten, "toggle")
        self._tip(self.auto_straighten_btn, "Measures the film frame's edges and levels the picture", "auto_straighten")
        self.autocrop_cb.ensurePolished()
        self.straighten.value_input.ensurePolished()
        angle_input = self.straighten.value_input
        angle_width = max(self.autocrop_cb.sizeHint().width(),
                          angle_input.fontMetrics().horizontalAdvance("-15.0°") + 32)
        self.straighten.value_input.setFixedSize(angle_width, self.autocrop_cb.sizeHint().height())
        side = self.autocrop_cb.sizeHint().height()
        self.autocrop_refresh.setMinimumHeight(0)
        self.autocrop_refresh.setFixedSize(side, side)
        self.straighten.top.setSpacing(6)
        top = self.straighten.top
        top.insertWidget(top.indexOf(self.straighten.value_input), self.auto_straighten_btn)
        # The two rows line up, Auto under Auto. Each button sits in a holder of that width: the stylesheet
        # puts its own minimum back on a button whenever its state is restyled, a holder keeps its size.
        for b, row_lay in ((self.autocrop_cb, crop_row), (self.crop_btn, crop_row), (self.auto_straighten_btn, top)):
            holder = QWidget()
            hold = QHBoxLayout(holder)
            hold.setContentsMargins(0, 0, 0, 0)
            row_lay.replaceWidget(b, holder)
            hold.addWidget(b)
            b.setMinimumWidth(0)
            b.setMaximumWidth(16777215)
            holder.setFixedWidth(angle_width)
        lay.addWidget(self.straighten)
        right.addWidget(p)

        self.steppers: dict[str, Stepper] = {}
        p, lay = panel()                                  # all six faders, like a mixing desk
        row = QHBoxLayout()
        row.setSpacing(0)
        for key, title, up, down in [("density", "D", "dens+", "dens-"), ("contrast", "Con", "con+", "con-"),
                                     ("saturation", "Sat", "sat+", "sat-")]:
            self.steppers[key] = Stepper(title, up, down, step=0.05, lo=0.0, hi=3.0)
            row.addWidget(self.steppers[key])
        self.steppers["density"].plus.setToolTip("Darker")
        self.steppers["density"].minus.setToolTip("Lighter")
        for key in ("contrast", "saturation"):
            self.steppers[key].plus.setToolTip(f"More {key}")
            self.steppers[key].minus.setToolTip(f"Less {key}")
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet(f"color: {COLORS['line']};")
        row.addWidget(line)
        for key, title, plus, minus, other in [("cyan", "C", "plusC", "minusR", "red"),
                                               ("magenta", "M", "plusM", "minusG", "green"),
                                               ("yellow", "Y", "plusY", "minusB", "blue")]:
            # one control per column: down is the opposite colour (less cyan = more red)
            # 0.02 a step, and the track covers the whole range (±25 steps = 0.5 .. 1.5)
            self.steppers[key] = Stepper(title, plus, minus, step=0.02, lo=0.5, hi=1.5, span=25)
            self.steppers[key].plus.setToolTip(f"More {key}")
            self.steppers[key].minus.setToolTip(f"More {other} (less {key})")
            row.addWidget(self.steppers[key])
        lay.addLayout(row)
        row = QHBoxLayout()
        self.autowb_btn = IconButton(_icon_wand, None, self.auto_wb)
        self._tip(self.autowb_btn, "Auto WB: make the average colour of the photo neutral", "auto_wb")
        self.pick_btn = IconButton(_icon_dropper, None, self.toggle_pick_wb)
        self._tip(self.pick_btn, "Pick grey: click something in the photo that should be grey or white", "pick_wb")
        reset_wb = IconButton(_icon_reset, None, self.reset_wb)
        reset_wb.setToolTip("Reset WB: back to the automatic colour")
        row.setSpacing(6)
        for b in (self.autowb_btn, self.pick_btn, reset_wb):      # quiet tiles, like rotate and flip
            b.setMinimumHeight(0)
            b.setFixedHeight(26)
            row.addWidget(b, 1)
        lay.addLayout(row)
        right.addWidget(p)
        for s in self.steppers.values():
            s.changed.connect(lambda _: self.on_setting_changed())

        right.addWidget(self._build_more())               # collapsed sliders

        right.addStretch(1)

        right_w = QWidget()
        right_w.setLayout(right)
        scroll = QScrollArea()          # the column scrolls if the window is short
        right_w.setFixedWidth(360 - 10)                # the scrollbar's room is kept: nothing reflows when it appears
        scroll.setWidget(right_w)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        # The Exif tab swaps only this column; the rolls, photo and filmstrip stay
        self.exif_panel = exif_panel.ExifPanel(panel, button)
        self.exif_panel.edited.connect(self.on_exif_edited)
        self.exif_panel.apply_fields.connect(self.on_exif_apply)
        exif_scroll = QScrollArea()
        self.exif_panel.setFixedWidth(360 - 10)
        exif_scroll.setWidget(self.exif_panel)
        exif_scroll.setWidgetResizable(True)
        exif_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        exif_scroll.setFrameShape(QFrame.NoFrame)
        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(scroll)
        self.right_stack.addWidget(exif_scroll)
        self.right_stack.setFixedWidth(360)
        self.right_stack.setVisible(False)            # appears with the first pictures
        self._exif_clipboard: dict | None = None

        top = QHBoxLayout()
        top.setSpacing(22)
        top.addWidget(self.left)
        top.addLayout(center, 1)
        top.addWidget(self.right_stack)
        self._root = root = QVBoxLayout()
        self._root_margins = (18, 14, 18, 10)
        root.setContentsMargins(*self._root_margins)
        root.setSpacing(14)
        root.addLayout(top, 1)
        root.addWidget(self.grid)
        w = QWidget()
        w.setLayout(root)

        # ---- Top: tab switch between the converter and the key bindings
        self.bindings_page = keys.BindingsPage(keys.load())
        self.bindings_page.changed.connect(self.on_bindings_changed)
        self.main_stack = QStackedWidget()
        self.settings_page = self._build_settings()
        self.main_stack.addWidget(w)
        self.main_stack.addWidget(self.bindings_page)
        self.main_stack.addWidget(self.settings_page)
        tabs = QHBoxLayout()
        tabs.setContentsMargins(18, 10, 18, 0)
        tabs.setSpacing(0)
        self.tab_plate = plate = QFrame()
        plate.setObjectName("tabPlate")
        plate_lay = QHBoxLayout(plate)
        plate_lay.setContentsMargins(3, 3, 3, 3)
        plate_lay.setSpacing(2)
        self.tab_buttons = []
        for i, text in enumerate(("Convert", "Exif", "Keys", "")):    # the last one is the gear: Settings
            b = FadeButton(text) if text else GearButton()
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _=False, i=i: self.set_tab(i))
            b.setCursor(Qt.PointingHandCursor)
            plate_lay.addWidget(b)
            self.tab_buttons.append(b)
        self.tab_buttons[0].setChecked(True)
        sep = QFrame()                                # the save icon: part of the bar, after a hairline
        sep.setFixedSize(1, 16)
        sep.setStyleSheet(f"background: {COLORS['edge_lo']};")
        plate_lay.addSpacing(2)
        plate_lay.addWidget(sep, 0, Qt.AlignVCenter)
        plate_lay.addSpacing(2)
        self.save_btn.setParent(plate)
        self.save_btn.setObjectName("tab")
        self.save_btn.rest_color = COLORS["text"]    # brighter than the tab names: it is an action
        plate_lay.addWidget(self.save_btn)
        plate.setFixedWidth(280)                      # as wide as the rolls column under it
        tabs.addWidget(plate, 0, Qt.AlignVCenter)
        tabs.addSpacing(22)                           # the gaps of the columns below: the name sits over the photo
        tabs.addWidget(self.title_box, 1)
        tabs.addSpacing(22)
        # Export: top right on every tab, as tall as the tabs; lights up once there is something to export
        self.export_btn = button("Export", self.open_export)
        self.export_btn.setObjectName("export")
        self.export_btn.setProperty("joined", "left")
        self.export_btn.setFixedHeight(36)
        self.export_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.export_more = IconButton(_icon_chevron_down, None, self._export_menu, "More: contact sheet")
        self.export_more.setObjectName("export")
        self.export_more.setProperty("joined", "right")
        self.export_more.rest_color = COLORS["text"]
        self.export_more.setMinimumHeight(0)
        self.export_more.setFixedSize(36, 36)
        self.export_wrap = QWidget()                  # one control, as wide as the right column below it
        wrap = QHBoxLayout(self.export_wrap)
        wrap.setContentsMargins(11, 0, 21, 0)         # the edges of the panels below, not the column's scroll margins
        wrap.setSpacing(1)
        wrap.addWidget(self.export_btn)
        wrap.addWidget(self.export_more)
        keep = self.export_wrap.sizePolicy()
        keep.setRetainSizeWhenHidden(True)            # hidden, it still holds its space: the tab row never jumps
        self.export_wrap.setSizePolicy(keep)
        self.export_wrap.setFixedWidth(360)            # the right column's width
        self.export_wrap.setVisible(False)            # appears with the first pictures
        tabs.addWidget(self.export_wrap)
        self.backdrop = outer = Backdrop()
        outer.photo = self.preview
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(0)
        self.tab_row = QWidget()
        self.tab_row.setLayout(tabs)
        outer_lay.addWidget(self.tab_row)
        outer_lay.addWidget(self.main_stack, 1)
        self.setCentralWidget(outer)
        self.overlay = DropOverlay(outer)
        self.left.setVisible(False)
        self.grid.setVisible(False)
        self.statusBar().showMessage("Add a roll or single frames to start.")

    def _build_welcome(self) -> QWidget:
        """First screen: choose between a whole roll and single frames."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.addStretch(1)
        title = QLabel("What are you converting?")
        title.setObjectName("welcome")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)
        row = QHBoxLayout()
        row.addStretch(1)
        for text, sub, slot in [("A whole roll", "Frames of one film, analysed together so colours match.", self.open_roll),
                                ("Single frames", "Unrelated pictures, each analysed on its own.", self.open_frames)]:
            b = ChoiceCard(text, sub)
            b.setMinimumSize(120, 140)
            b.setMaximumSize(280, 160)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            b.clicked.connect(slot)
            if row.count() > 1:
                row.addSpacing(16)               # between the cards only, so the pair stays centred
            row.addWidget(b, 1)              # equal shares, so both cards are the same width
        row.addStretch(1)
        lay.addLayout(row)
        lay.addSpacing(10)
        hint = QLabel("…or drag folders or files into the window")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)
        lay.addStretch(2)
        return page

    def _build_roll_details(self) -> QWidget:
        """Film stock, development, camera, lens, date and location of the selected roll.
        Shown inside that roll's card when it is expanded."""
        p = QWidget(self.cards)                       # never a window of its own (that would steal activation)
        p.hide()
        lay = QVBoxLayout(p)
        lay.setContentsMargins(0, 10, 0, 12)
        lay.setSpacing(8)
        self._select_on_focus = SelectOnFocus(self)
        self.stock_box = exif_panel.ChevronCombo()
        self.stock_box.setEditable(True)
        self.stock_box.setInsertPolicy(QComboBox.NoInsert)
        self.stock_box.addItem("No film stock", "")
        for d in engine.film_stocks():
            self.stock_box.addItem(d["displayName"], d["id"])
        comp = self.stock_box.completer()
        comp.setFilterMode(Qt.MatchContains)          # type "gold" to find Kodak Gold 200
        comp.setCompletionMode(QCompleter.PopupCompletion)
        self.stock_box.lineEdit().setPlaceholderText("Choose the film stock: type to search")
        self.stock_box.installEventFilter(self._select_on_focus)
        comp.popup().setObjectName("completer")       # the suggestions: a solid list over the card
        self.stock_box.activated.connect(self.on_stock_changed)
        self.stock_box.setToolTip("The film decides the conversion: its dye profile, its Lab print colours, "
                                  "B&W or colour. Type to search")
        self.camera_edit = exif_panel.PresetField("cameras", "Camera (e.g. Nikon FM2)")
        self.camera_edit.committed.connect(lambda _: self.on_camera())
        self.lens_edit = exif_panel.PresetField("lenses", "Lens (e.g. 50mm f/1.8)")
        self.lens_edit.committed.connect(lambda _: self.on_roll_meta(self.lens_edit))
        self.date_edit = exif_panel.DateField()
        self.date_edit.changed_iso.connect(lambda _: self.on_roll_meta(self.date_edit))
        self.place_edit = exif_panel.LocationField("Location: search a place or paste GPS")
        self.place_edit.picked.connect(self.on_roll_place)
        self.place_edit.coords_typed.connect(self.on_roll_coords)
        self.push_box = exif_panel.ChevronCombo()
        for n in range(-3, 4):
            self.push_box.addItem("Developed normally" if n == 0 else f"{'Pushed' if n > 0 else 'Pulled'} {abs(n)} stop{'s' if abs(n) > 1 else ''}", n)
        self.push_box.setCurrentIndex(3)
        self.push_box.activated.connect(lambda _: self.on_roll_meta(self.push_box))
        self.roll_status = QLabel("")
        self.roll_status.setObjectName("hint")
        self.roll_status.setWordWrap(True)
        self.roll_status.hide()                       # kept for its text (tooltips, tests), not shown
        for c in (self.stock_box, self.push_box):        # the column's width, not the longest film name
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(6)
        for w in (self.stock_box, self.push_box, self.camera_edit, self.lens_edit, self.date_edit,
                  self.place_edit):
            w.setMinimumWidth(0)
            lay.addWidget(w)
        for w in (self.camera_edit, self.lens_edit, self.place_edit):
            w.returnPressed.connect(self.escape)     # Return: done typing, keys drive the scanner again
        p.setVisible(False)
        return p

    def _build_settings(self) -> QWidget:
        """The Settings tab: the few things that belong to you or your scanning setup, not to a photo."""
        col = QVBoxLayout()
        col.setSpacing(14)
        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        col.addWidget(title)
        col.addSpacing(4)

        def section(name: str) -> QVBoxLayout:
            p, lay = panel()
            lay.setSpacing(12)
            h = QLabel(name.upper())
            h.setObjectName("header")
            lay.addWidget(h)
            col.addWidget(p)
            return lay

        def row(lay, name: str, about: str, *controls) -> QLabel:
            """One setting: its name and a line about it on the left, its control(s) on the right."""
            text = QVBoxLayout()
            text.setSpacing(2)
            n = QLabel(name)
            a = QLabel(about)
            a.setObjectName("rollSub")
            a.setWordWrap(True)
            text.addWidget(n)
            text.addWidget(a)
            r = QHBoxLayout()
            r.setSpacing(6)
            r.addLayout(text, 1)
            r.addSpacing(18)
            for c in controls:
                r.addWidget(c, 0, Qt.AlignVCenter)
            lay.addLayout(r)
            return a

        def switch(key: str, on_change=None) -> QPushButton:
            b = button("", None, "toggle")
            b.setCheckable(True)
            b.setChecked(pref(key))
            b.setText("On" if b.isChecked() else "Off")
            b.setMinimumWidth(56)

            def flip(on: bool):
                b.setText("On" if on else "Off")
                set_pref(key, on)
                if on_change:
                    on_change(on)
            b.toggled.connect(flip)
            return b

        # Scanning setup
        lay = section("Scanning setup")
        flat_btn = button("Choose Photo…", self.choose_flat)
        flat_btn.setToolTip("A photo of your bare light panel, shot with the same lens and aperture as the scans")
        self.flat_clear = button("Remove", self.clear_flat)
        self.flat_clear.setVisible(False)
        self._flat_about = "Evens out the light and removes dark corners on every scan."
        self.flat_label = row(lay, "Flat field", self._flat_about, flat_btn, self.flat_clear)
        self.light_tabs = SlideTabs(["White", "RGB"])
        self.light_tabs.setFixedWidth(150)
        self.light_tabs.select(1 if pref("light") == "rgb" else 0, animate=False)
        self.light_tabs.chosen.connect(lambda i: self.set_light("rgb" if i else "white"))
        row(lay, "Scanning light", "White: a broadband light panel. RGB: separate red, green and blue LEDs. "
            "The conversion separates the film's colours to match.", self.light_tabs)

        # Look
        lay = section("Look")
        self.accent_picker = AccentPicker(pref("accent"))
        def set_accent(key: str):
            set_pref("accent", key)
            apply_accent(key)
            for g in self.groups:
                if g.card:
                    g.card._paint_state()
            self._refresh_stock_needed()
        self.accent_picker.chosen.connect(set_accent)
        row(lay, "Accent", "The colour that marks what is chosen: selection, the main buttons, progress.",
            self.accent_picker)

        # New frames
        lay = section("New frames")
        self.pref_autocrop = switch("auto_crop_new")
        row(lay, "Auto crop", "Crop the film border away on every frame you add (and after Reset).", self.pref_autocrop)

        # While working
        lay = section("While working")
        self.pref_warnings = switch("scan_warnings", lambda _on: self.update_nav())
        row(lay, "Scan warnings", "Say so next to the file name when a scan is very dark or blown out.", self.pref_warnings)
        def gpu(on: bool):
            engine.USE_GPU = on
            self.request_preview()
        self.pref_gpu = switch("gpu", gpu)
        row(lay, "Graphics card", "Render the preview and zoom on the graphics card: smoother editing. "
            "Off uses the processor only; exports always do.", self.pref_gpu)
        self.pref_border = switch("border", lambda _on: self._update_border())
        self.border_width = QSpinBox()
        self.border_width.setRange(1, 2000)
        self.border_width.setSuffix(" px")
        self.border_width.setValue(pref("border_px"))
        self.border_width.setFixedWidth(84)
        self.border_width.setToolTip("Width of the border on an exported file, in its pixels")

        def border_px(v: int):
            set_pref("border_px", v)
            self._update_border()
        self.border_width.valueChanged.connect(border_px)
        row(lay, "White border", "A white border around the photo, like a print's. O shows or hides it. "
            "Its width is in pixels of the exported file.", self.border_width, self.pref_border)

        # Exif presets
        lay = section("Exif presets")

        def edit(kind: str, what: str):
            store = exif_panel.presets()
            dlg = exif_panel.PresetDialog(self, what, store.presets[kind])
            if dlg.exec():
                store.set(kind, dlg.names())
        row(lay, "Cameras", "The cameras offered when you fill in a roll's details.",
            button("Edit…", lambda: edit("cameras", "Camera Presets")))
        row(lay, "Lenses", "The lenses offered when you fill in a roll's details.",
            button("Edit…", lambda: edit("lenses", "Lens Presets")))

        # About
        lay = section("About")
        try:
            base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent / "vendor" / "negpy"
            version = (base / "VERSION").read_text().strip()
        except OSError:
            version = ""
        from . import __version__
        about = QLabel(f"{APP_NAME} {__version__} converts with the NegPy engine{' ' + version if version else ''}. "
                       "Free software under the GPLv3 licence.")
        about.setObjectName("rollSub")
        about.setWordWrap(True)
        lay.addWidget(about)
        col.addStretch(1)

        inner = QWidget()
        inner.setLayout(col)
        inner.setFixedWidth(620)
        holder = QWidget()
        hl = QHBoxLayout(holder)
        hl.setContentsMargins(18, 18, 18, 18)
        hl.addStretch(1)
        hl.addWidget(inner)
        hl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        saved = pref("flat_field")                    # the flat field of last time, if the photo is still there
        if saved and Path(saved).exists():
            self.flat_field = saved
            self.flat_label.setText(f"{Path(saved).name}  ·  {self._flat_about}")
            self.flat_clear.setVisible(True)
        return scroll

    def choose_flat(self):
        exts = " ".join(f"*{x}" for x in sorted(engine.IMAGE_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(self, "Flat field photo (bare light panel)", "", f"Images ({exts})")
        if path:
            self.set_flat(path)

    def set_flat(self, path: str | None):
        self.flat_field = path
        set_pref("flat_field", path or "")
        self.flat_label.setText(f"{Path(path).name}  ·  {self._flat_about}" if path else self._flat_about)
        self.flat_clear.setVisible(bool(path))
        self._apply_lens_to_all()

    def clear_flat(self):
        self.set_flat(None)

    def set_light(self, light: str):
        """White or RGB light (Settings): every frame converts for it, and the rolls measure again."""
        set_pref("light", light)
        if self.light_tabs.current != ("rgb" == light):
            self.light_tabs.select(1 if light == "rgb" else 0, animate=False)
        for ph in self.all_photos():
            ph.settings.light = light
        self._thumb_timer.start()
        self._reanalyze_timer.start()           # the colour separation changed: rolls measure again
        self.request_preview()
        self.statusBar().showMessage(f"Converting for {'RGB' if light == 'rgb' else 'white'} light", 4000)

    def _apply_lens_to_all(self):
        for ph in self.all_photos():
            ph.settings.flat_field = self.flat_field
        self._thumb_timer.start()      # refresh thumbnails once the clicking stops
        self._reanalyze_timer.start()  # corners measured differently: rolls measure again
        what = Path(self.flat_field).name if self.flat_field else "off"
        self.statusBar().showMessage(f"Flat field for all images: {what}")

    def _refresh_all_thumbs(self):
        for ph in self.all_photos():
            if ph is not self.current:
                self.render_thumb(ph)

    def _build_more(self) -> QFrame:
        """A panel whose header button folds the extra sliders open/closed."""
        p, lay = panel()
        self.more_btn = FadeButton("Refine")              # the whole header row is the switch
        self.more_btn.setCheckable(True)
        self.more_btn.setObjectName("fold")
        self.more_btn.setCursor(Qt.PointingHandCursor)
        self.more_chevron = Chevron()
        self.more_chevron.clicked.connect(lambda _=False: self.more_btn.toggle())
        head_row = QHBoxLayout()
        head_row.setSpacing(0)
        head_row.addWidget(self.more_btn, 1)
        head_row.addWidget(self.more_chevron)
        lay.addLayout(head_row)

        self._refine_inner = QWidget()
        self.more_body = ClipBox(self._refine_inner)
        self.more_body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        outer = QVBoxLayout(self._refine_inner)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(10)
        self.sliders: dict[str, LineSlider] = {}
        defaults = engine.Settings()

        # Light · Color · Finish: one strip, its highlight gliding to the chosen tab
        self.refine_bar = SlideTabs(["Light", "Color", "Finish"])
        self.refine_bar.chosen.connect(self._refine_tab)
        self.refine_pages = FitStack()
        outer.addWidget(self.refine_bar)
        outer.addWidget(self.refine_pages)

        def page():
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)
            self.refine_pages.addWidget(w)
            return v

        def line(body, key, title, lo, hi, unipolar=False, minus="", plus=""):
            f = LineSlider(title, lo, hi, default=getattr(defaults, key), unipolar=unipolar,
                           minus=FADER_COLORS.get(minus, ""), plus=FADER_COLORS.get(plus, ""))
            f.setToolTip(self.TOOLTIPS.get(key, ""))
            f.changed.connect(lambda _: self.on_setting_changed())
            self.sliders[key] = f
            body.addWidget(f)

        def head(body, text, first=False):
            h = QLabel(text.upper())
            h.setObjectName("header")
            if not first:
                body.addSpacing(10)
            body.addWidget(h)
            body.addSpacing(2)

        # Light: how bright, and the two ends of the tonal range
        light = page()
        line(light, "exposure", "Exposure", -1.0, 1.0)
        self.exposure_slider = self.sliders["exposure"]
        line(light, "highlight_tone", "Highlights", -1.0, 1.0)
        line(light, "shadow_tone", "Shadows", -1.0, 1.0)
        line(light, "white_point", "Whites", -1.0, 1.0)
        line(light, "black_point", "Blacks", -1.0, 1.0)
        light.addSpacing(10)
        recalc = button("Re-measure Inside Crop", self.recalc)
        recalc.setToolTip("Measure the film again using only the area inside the crop, then Auto WB. "
                          "Helps when the film border or a light leak threw the colours off")
        light.addWidget(recalc)
        light.addStretch(1)

        # Color: the mixer (a colour, then its hue, saturation, luminance) and the two casts
        color = page()
        self.mixer = QWidget()
        mix = QVBoxLayout(self.mixer)
        mix.setContentsMargins(0, 0, 0, 0)
        mix.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        h = QLabel("MIXER")
        h.setObjectName("header")
        row.addWidget(h)
        row.addStretch(1)
        self.mixer_picker = IconButton(_icon_dropper, None, lambda: self.toggle_pick("mixer"),
                                       "Click a colour in the photo to choose it in the mixer")
        self.mixer_picker.setMinimumHeight(0)
        self.mixer_picker.setFixedSize(22, 18)
        self.mixer_picker.scale = 0.7
        self.mixer_picker.setStyleSheet("QPushButton { background: transparent; border: none; }")
        row.addWidget(self.mixer_picker)
        mix.addLayout(row)
        mix.addSpacing(2)
        dots = QHBoxLayout()
        dots.setSpacing(0)
        self.hsl_dots: list[ColorDot] = []
        for i, (name, colour) in enumerate(HSL_BANDS):
            d = ColorDot(name, colour)
            d.setAutoExclusive(True)
            d.clicked.connect(lambda _c=False, i=i: self._hsl_band_chosen(i))
            dots.addWidget(d)
            self.hsl_dots.append(d)
        dots.addStretch(1)
        mix.addLayout(dots)
        self._hsl_band = 0
        self.hsl_dots[0].setChecked(True)
        self.hsl_sliders: dict[str, LineSlider] = {}
        for key, title, tip in (("hsl_hue", "Hue", "Turn this colour toward its neighbours"),
                                ("hsl_saturation", "Saturation", "How strong this colour is: left takes it out"),
                                ("hsl_luminance", "Luminance", "How light or dark this colour is")):
            f = LineSlider(title, -1.0, 1.0, default=0.0)
            f.setToolTip(tip)
            f.changed.connect(lambda _v, key=key: self._on_hsl(key))
            self.hsl_sliders[key] = f
            mix.addWidget(f)
        color.addWidget(self.mixer)
        self.bw_joke = BwJoke()                    # B&W film: nothing to mix, so a smile instead
        self.bw_joke.setVisible(False)
        color.addWidget(self.bw_joke)
        # White balance, like Lightroom's Temp and Tint, with its dropper on the heading
        self.wb_box = QWidget()                    # hidden for B&W, like the mixer
        wbl = QVBoxLayout(self.wb_box)
        wbl.setContentsMargins(0, 0, 0, 0)
        wbl.setSpacing(2)
        wbl.addSpacing(10)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        h = QLabel("WHITE BALANCE")
        h.setObjectName("header")
        row.addWidget(h)
        row.addStretch(1)
        self.temp_picker = IconButton(_icon_dropper, None, lambda: self.toggle_pick("temp"),
                                      "Click something in the photo that should be grey or white")
        self.temp_picker.setMinimumHeight(0)
        self.temp_picker.setFixedSize(22, 18)
        self.temp_picker.scale = 0.7
        self.temp_picker.setStyleSheet("QPushButton { background: transparent; border: none; }")
        row.addWidget(self.temp_picker)
        wbl.addLayout(row)
        wbl.addSpacing(2)
        line(wbl, "temp", "Temp", -1.0, 1.0)
        line(wbl, "tint", "Tint", -1.0, 1.0)
        color.addWidget(self.wb_box)
        self.sliders["temp"].gradient = [QColor("#4a7bd0"), QColor("#8a8f96"), QColor("#d9a640")]
        self.sliders["tint"].gradient = [QColor("#4fa35c"), QColor("#8a8f96"), QColor("#c45aa8")]
        color.addStretch(1)

        # Finish: detail, grain colour, glow
        finish = page()
        line(finish, "sharpen", "Sharpen", 0.0, 1.0, unipolar=True)
        line(finish, "color_noise", "Colour noise", 0.0, 1.0, unipolar=True)
        line(finish, "glow", "Glow", 0.0, 1.0, unipolar=True)
        finish.addSpacing(6)
        look = QHBoxLayout()                       # the print look: a paper or a minilab scanner's rendering
        look.setContentsMargins(0, 0, 0, 0)
        look_name = QLabel("Print look")
        look_name.setStyleSheet(f"color: {COLORS['muted']}; font-size: 12px;")   # like a slider's name
        look_name.setFixedWidth(LineSlider.LABEL_W)
        look.addWidget(look_name)
        self.paper_box = exif_panel.ChevronCombo()
        self.paper_box.setToolTip("Lab print (the default) gives the colours the film takes on photo paper, "
                                  "measured from its data sheet: pick the film stock for its own. Or a photo paper, "
                                  "a cinema print film or a minilab scanner's rendering. None keeps the straight conversion")
        self.paper_box.setFocusPolicy(Qt.StrongFocus)  # the wheel scrolls the panel, not the looks
        self.paper_box.activated.connect(self._on_paper)
        look.addWidget(self.paper_box, 1)
        finish.addLayout(look)
        self._paper_grayscale: bool | None = None
        finish.addStretch(1)
        self.refine_pages.setCurrentIndex(0)
        self._hsl_band_chosen(0, load=False)
        self.more_body.setVisible(False)
        lay.addWidget(self.more_body)

        self._fold_anim = QPropertyAnimation(self.more_body, b"maximumHeight", self)
        self._fold_anim.setDuration(220)
        self._fold_anim.setEasingCurve(QEasingCurve.OutCubic)

        def fold_done():
            if self.more_btn.isChecked():
                self.more_body.setMaximumHeight(16777215)      # open: free to follow its content again
            else:
                self.more_body.setVisible(False)
        self._fold_anim.finished.connect(fold_done)

        def fold(open_: bool):
            """Slide the section open or shut (its height grows from, or shrinks to, nothing)."""
            self.more_chevron.setChecked(open_)
            self.more_chevron.update()
            self._fold_anim.stop()
            self._fold_anim.setDuration(220)
            if open_:
                self.more_body.setMaximumHeight(0)
                self.more_body.setVisible(True)
            self._fold_anim.setStartValue(self.more_body.height() if not open_ else 0)
            self._fold_anim.setEndValue(self.more_body.sizeHint().height() if open_ else 0)
            self._fold_anim.start()
        self.more_btn.toggled.connect(fold)
        return p

    # Window width (px) under which the rolls column hides. It must stay above the window's
    # minimum width *with* the column, or the window could never get narrow enough to hide it.
    LEFT_HIDE_BELOW = 1200

    def _fit_left_column(self):
        """The rolls column only shows when there's room; ↑ / ↓ still switch rolls without it."""
        self.left.setVisible(bool(self.groups) and self.width() >= self.LEFT_HIDE_BELOW and not self._fullscreen)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "left"):
            self._fit_left_column()

    # ----- Adding / removing files -----------------------------------------
    def _overlay_side(self, e) -> str:
        pos = self.overlay.mapFrom(self, e.position().toPoint())
        return self.overlay.side_at(pos.x())

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.overlay.setGeometry(self.centralWidget().rect())
            self.overlay.side = self._overlay_side(e)
            self.overlay.show()
            self.overlay.raise_()

    def dragMoveEvent(self, e):
        side = self._overlay_side(e)
        if side != self.overlay.side:
            self.overlay.side = side
            self.overlay.update()
        e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self.overlay.hide()

    def dropEvent(self, e):
        self.overlay.hide()
        self.set_tab(0)
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls()]
        if self._overlay_side(e) == "roll":
            self.add_rolls(paths)
        else:
            self.add_frames(paths)

    def open_roll(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose the folder of one roll")
        if folder:
            self.add_rolls([Path(folder)])

    def open_frames(self):
        exts = " ".join(f"*{x}" for x in sorted(engine.IMAGE_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(self, "Add single frames", "", f"Images ({exts})")
        if files:
            self.add_frames([Path(f) for f in files])

    @staticmethod
    def _images_in(paths: list[Path]) -> list[Path]:
        out = []
        for p in paths:
            for f in (sorted(p.iterdir()) if p.is_dir() else [p]):
                if f.is_file() and f.suffix.lower() in engine.IMAGE_EXTENSIONS and not f.name.startswith("."):
                    out.append(f)
        return out

    def add_rolls(self, paths: list[Path]):
        """Each dropped folder is one roll; loose files dropped together are one roll."""
        folders = [p for p in paths if p.is_dir()]
        files = [p for p in paths if not p.is_dir()]
        batches = [(f.name, self._images_in([f])) for f in folders]
        if files:
            batches.append((files[0].parent.name, self._images_in(files)))
        last = None
        for name, imgs in batches:
            if not imgs:
                continue
            g = Group(kind="roll", name=name or "Roll")
            self._add_group(g, imgs)
            last = g
        if last:
            self.show_group(last)
        else:
            self.statusBar().showMessage("No images found in what you dropped.")

    def add_frames(self, paths: list[Path]):
        imgs = self._images_in(paths)
        if not imgs:
            self.statusBar().showMessage("No images found in what you dropped.")
            return
        g = next((x for x in self.groups if x.kind == "frames"), None)
        if g is None:
            g = Group(kind="frames", name="Single frames")
            self._add_group(g, [])
        known = {ph.path for ph in g.photos}
        for f in imgs:
            if f not in known:
                self._add_photo(g, f)
        self._refresh_group_item(g)
        self.show_group(g)

    def _add_group(self, g: Group, imgs: list[Path]):
        self.groups.append(g)
        g.card = RollCard(g.kind == "roll", self.cards)
        g.card.clicked.connect(lambda g=g: self.show_group(g) if g is not self.group else None)
        g.card.toggled.connect(lambda on, g=g: self._toggle_details(g, on))
        g.card.menu.connect(lambda pos, g=g: self._group_menu(g, pos))
        g.card.renamed.connect(lambda name, g=g: self.rename_group(g, name))
        g.card.remove.connect(lambda g=g: self.confirm_remove_group(g))
        self.cards_lay.insertWidget(self.cards_lay.count() - 1, g.card)
        for f in imgs:
            self._add_photo(g, f)
        self._refresh_group_item(g)
        self._fit_left_column()
        self.right_stack.setVisible(True)
        self.grid.setVisible(True)
        self.export_wrap.setVisible(self.main_stack.currentIndex() == 0)
        self.stack.setCurrentIndex(1)
        self._update_toolbars()

    def _add_photo(self, g: Group, f: Path):
        photo = Photo(path=f, group=g)
        photo.settings.flat_field = self.flat_field
        photo.settings.light = pref("light")
        photo.settings.auto_crop = pref("auto_crop_new")
        engine.settings_for_stock(photo.settings, g.stock)
        photo.history, photo.hindex = [photo.settings.copy()], 0     # the automatic conversion: where undo ends
        g.photos.append(photo)
        self.render_thumb(photo)
        if photo.settings.auto_crop:
            self.auto_level(photo, automatic=True)

    def rename_group(self, g: Group, name: str):
        """The roll's own name instead of its folder's: shown here, and used for export subfolders
        and the {roll} name token."""
        g.name = name
        self._refresh_group_item(g)
        if g is self.group:
            self.update_nav()                     # the title above the photo
        self.statusBar().showMessage(f"Roll renamed to {name}.", 3000)

    def _refresh_group_item(self, g: Group):
        if g.card is not None:
            g.card.set_text(*g.label())

    def _toggle_details(self, g: Group, on: bool):
        """A roll's chevron: show or hide its details inside its card (selecting it first)."""
        self._details_open = on
        if g is not self.group:
            self.show_group(g)
        else:
            self._place_details()

    def _place_details(self):
        """Move the shared details widget into the selected roll's card (or hide it)."""
        g = self.group
        show = bool(g and g.kind == "roll" and self._details_open)
        if show and self.roll_box.parent() is not g.card:
            g.card.body.addWidget(self.roll_box)
        self.roll_box.setVisible(show)
        if show:                      # an opened roll low in the list: bring its details into view
            QTimer.singleShot(0, lambda c=g.card: self._scroll_to_card(c))
        for other in self.groups:
            other.card.set_selected(other is g)
            other.card.set_open(other is g and show)
            other.card.chevron.blockSignals(True)
            other.card.chevron.setChecked(other is g and show)
            other.card.chevron.update()
            other.card.chevron.blockSignals(False)

    def _scroll_to_card(self, card):
        """Show as much of the card as fits, its top (the roll's name) always."""
        if not card.isVisible():
            return
        bar, view = self.cards_scroll.verticalScrollBar(), self.cards_scroll.viewport().height()
        top, bottom = card.y(), card.y() + card.height()
        bar.setValue(min(top, max(bar.value(), bottom - view)))

    # ----- Removing frames and rolls (from the list only: files on disk are never touched) ------
    def _frame_menu(self, pos):
        if not self.grid.itemAt(pos):
            return
        n = len(self.grid.selectedItems())
        m = QMenu(self)
        m.addAction("Export Selected…", lambda: self.open_export(selected_only=True))
        m.addSeparator()
        m.addAction(f"Remove {n} Frames from List" if n > 1 else "Remove Frame from List", self.remove_selected_frames)
        m.exec(self.grid.viewport().mapToGlobal(pos))

    def _group_menu(self, g: Group, pos):
        m = QMenu(self)
        if g.kind == "roll":
            m.addAction("Contact Sheet…", lambda: self.open_contact_sheet(g))
            m.addSeparator()
        m.addAction("Remove Roll from List…" if g.kind == "roll" else "Remove All Single Frames from List…",
                    lambda: self.confirm_remove_group(g))
        m.exec(pos)

    def _forget(self, photos: list[Photo]):
        """Drop every reference the window keeps to these frames."""
        gone = set(photos)
        if self._stale_thumb in gone:
            self._stale_thumb = None
        if self._uv and self._uv[0] in gone:
            self._uv = None
        self._dirty_thumbs -= gone
        self._thumb_again -= gone
        self._pending_batch -= gone
        for ph in photos:
            ph.item = None

    def remove_selected_frames(self):
        """Cmd+Backspace, or right-click a frame: take the selected frames out of the list."""
        g = self.group
        if not g or self._typing() or self.main_stack.currentIndex() != 0:
            return
        rows = sorted(self.grid.row(i) for i in self.grid.selectedItems())
        if not rows:
            return
        if len(rows) == len(g.photos):
            self.remove_group(g)
            return
        if self.preview.crop_mode:
            self.toggle_crop(apply=False)
        gone = [g.photos[r] for r in rows]
        self._forget(gone)
        self.current = None                          # nothing left to finish for a frame that is going
        for ph in gone:
            g.photos.remove(ph)
        self.show_group(g)
        self.grid.setCurrentRow(min(rows[0], len(g.photos) - 1), QItemSelectionModel.ClearAndSelect)
        self._refresh_group_item(g)
        if g.kind == "roll":
            g.analyzed = False
            self.analyze_roll(g)                     # the roll's baseline was measured with those frames
        self.statusBar().showMessage(f"Removed {len(gone)} frame(s) from the list. The files are untouched.", 4000)

    def confirm_remove_group(self, g: Group):
        """The trash (or the right-click menu): ask first, then take the roll out of the list."""
        if g not in self.groups:
            return
        what = g.name if g.kind == "roll" else "all single frames"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setText(f"Remove {what} from the list?")
        box.setInformativeText("The scans on disk are not touched. Its edits in this session are lost.")
        go = box.addButton("Remove", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(go)
        box.exec()
        if box.clickedButton() is go:
            self.remove_group(g)

    def remove_group(self, g: Group):
        if g not in self.groups:
            return
        if self.preview.crop_mode:
            self.toggle_crop(apply=False)
        i = self.groups.index(g)
        self._forget(g.photos)
        if self.roll_box.parent() is g.card:         # the shared details widget lives in the open card
            self.roll_box.setVisible(False)
            self.roll_box.setParent(self.cards)
        self.groups.remove(g)
        self.cards_lay.removeWidget(g.card)
        g.card.deleteLater()
        g.card = None
        if g is self.group:
            self.current = None
            self.group, self.photos = None, []
            if self.groups:
                self.show_group(self.groups[min(i, len(self.groups) - 1)])
            else:
                self._show_empty()
        self.statusBar().showMessage(f"Removed {g.name} from the list. The files are untouched.", 4000)

    def _show_empty(self):
        """Nothing loaded any more: back to the first screen."""
        self.grid.clear()
        self.preview.set_image(None)
        self.preview.setText("")
        self.title.setText("")
        self.stack.setCurrentIndex(0)
        self.right_stack.setVisible(False)
        self.grid.setVisible(False)
        self.left.setVisible(False)
        self.export_wrap.setVisible(False)
        self._update_toolbars()

    # ----- Saving the session, and recovery after a crash ---------------------------------------
    # The save icon (Cmd+S) keeps the session (rolls, frames, adjustments) in one place of the app's
    # own, offered back at the next start. Nothing is saved without asking, except a recovery file
    # written every few seconds and removed again on a normal quit: if it is still there at the
    # next start, the app did not quit normally, and the work since the last save comes back.
    @staticmethod
    def _recovery_path() -> Path:
        from PySide6.QtCore import QStandardPaths
        return Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or Path.home() / ".dumbconvert") / "recovery.json"

    def _session(self) -> dict:
        from dataclasses import asdict
        return {"pid": os.getpid(), "groups": [
            {"kind": g.kind, "name": g.name, "stock_id": g.stock_id, "no_stock": g.no_stock, "camera": g.camera, "lens": g.lens, "date": g.date,
             "location": g.location, "location_label": g.location_label, "push_pull": g.push_pull,
             "photos": [{"path": str(ph.path), "settings": asdict(ph.settings), "exif": ph.exif} for ph in g.photos]}
            for g in self.groups]}

    def _session_path(self) -> Path:
        return self._recovery_path().with_name("session.json")

    def _session_text(self, data: dict | None = None) -> str:
        """The session's content, to compare: what a save keeps (no process id, no save time)."""
        import json
        groups = (data if data is not None else self._session()).get("groups", [])
        return json.dumps(groups, sort_keys=True, default=str)

    def session_dirty(self) -> bool:
        """Rolls are open and they changed since the last save (or were never saved)."""
        return bool(self.groups) and self._session_text() != self._saved_text

    def _update_saved_mark(self):
        if hasattr(self, "save_btn"):
            self.save_btn.set_dirty(self.session_dirty())

    def save_session(self) -> bool:
        """Cmd+S / the save icon: this session, ready to restore at the next start."""
        import datetime
        import json
        self._commit_history()
        path = self._session_path()
        if not self.groups:
            path.unlink(missing_ok=True)
            self._saved_text = ""
            self.statusBar().showMessage("Nothing open to save.", 3000)
            self._update_saved_mark()
            return True
        data = self._session()
        data.pop("pid", None)
        data["saved_at"] = datetime.datetime.now().isoformat(timespec="minutes")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(path)                         # never a half-written file
        except OSError as err:
            self.statusBar().showMessage(f"Could not save the session: {err}", 6000)
            return False
        self._saved_text = self._session_text(data)
        n = len(self.all_photos())
        self.statusBar().showMessage(f"Session saved: {n} frame{'s' if n != 1 else ''}.", 3000)
        self._update_saved_mark()
        return True

    def saved_session(self) -> dict | None:
        """The session saved with the save icon, if its photos are still there."""
        import json
        try:
            data = json.loads(self._session_path().read_text())
        except (OSError, ValueError):
            return None
        photos = [ph for g in data.get("groups", []) for ph in g.get("photos", []) if Path(ph.get("path", "")).exists()]
        return data if photos else None

    def offer_restore(self, ask=None):
        """At start: the work a crash left behind, else the saved session, each only if there is one.
        `ask(text, detail)` -> True to restore (a dialog; tests pass their own)."""
        ask = ask or self._ask_restore
        saved = self.saved_session()
        crashed = self.recovery_found()
        if crashed:
            n = sum(len(g.get("photos", [])) for g in crashed.get("groups", []))
            if ask(f"{APP_NAME} did not quit normally last time.",
                   f"Restore the {n} frame{'s' if n != 1 else ''} you were working on, with their adjustments?"):
                self.restore_session(crashed)
                self._saved_text = self._session_text(saved) if saved else ""
                self._update_saved_mark()
                return
        if saved:
            n = sum(len(g.get("photos", [])) for g in saved.get("groups", []))
            rolls = len(saved.get("groups", []))
            when = str(saved.get("saved_at", "")).replace("T", " at ")
            if ask("Restore your saved session?",
                   f"{n} frame{'s' if n != 1 else ''} in {rolls} roll{'s' if rolls != 1 else ''}"
                   + (f", saved on {when}." if when else ".")):
                self.restore_session(saved)
                self._saved_text = self._session_text()
                self._update_saved_mark()

    def _ask_restore(self, text: str, detail: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.NoIcon)
        box.setText(text)
        box.setInformativeText(detail)
        restore = box.addButton("Restore", QMessageBox.AcceptRole)
        box.addButton("Start Fresh", QMessageBox.RejectRole)
        box.setDefaultButton(restore)
        box.exec()
        return box.clickedButton() is restore

    def _write_recovery(self):
        self._update_saved_mark()
        if not self._recovery_on:
            return
        import json
        path = self._recovery_path()
        if not self.groups:
            path.unlink(missing_ok=True)
            self._recovery_text = ""
            return
        text = json.dumps(self._session())
        if text != self._recovery_text:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(text)
                tmp.replace(path)                    # never a half-written file
                self._recovery_text = text
            except OSError:
                pass

    def recovery_found(self) -> dict | None:
        """The session a crashed run left behind (None after a normal quit, or while another
        window of the app is still running and owns the file)."""
        import json
        path = self._recovery_path()
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        pid = int(data.get("pid") or 0)
        if pid and pid != os.getpid() and process_alive(pid):
            self._recovery_on = False                # that run is alive: the file is its, not ours
            return None
        photos = [ph for g in data.get("groups", []) for ph in g.get("photos", []) if Path(ph.get("path", "")).exists()]
        return data if photos else None

    def restore_session(self, data: dict):
        def tuples(v):                               # JSON turned every tuple into a list
            return tuple(tuples(x) for x in v) if isinstance(v, list) else v
        fields_ = engine.Settings.__dataclass_fields__
        last = None
        for gd in data.get("groups", []):
            entries = [ph for ph in gd.get("photos", []) if Path(ph.get("path", "")).exists()]
            if not entries:
                continue
            g = Group(kind=gd.get("kind", "frames"), name=gd.get("name", "Roll"))
            for k in ("stock_id", "no_stock", "camera", "lens", "date", "location", "location_label", "push_pull"):
                setattr(g, k, gd.get(k, getattr(g, k)))
            self._add_group(g, [])
            for e in entries:
                photo = Photo(path=Path(e["path"]), group=g)
                try:
                    photo.settings = engine.Settings(**{k: tuples(v) for k, v in e.get("settings", {}).items() if k in fields_})
                except TypeError:
                    pass
                photo.settings.flat_field = self.flat_field
                photo.settings.light = pref("light")       # the setup of now, not of last time
                photo.exif = dict(e.get("exif") or {})
                auto = engine.Settings()                         # undo still leads back to the automatic conversion
                for f in NOT_EDITS:
                    setattr(auto, f, getattr(photo.settings, f))
                photo.history, photo.hindex = [auto], 0
                if not same_edits(auto, photo.settings):
                    photo.history.append(photo.settings.copy())
                    photo.hindex = 1
                g.photos.append(photo)
                self.render_thumb(photo)
            self._refresh_group_item(g)
            last = g
        if last:
            self.show_group(self.groups[0])
            self.statusBar().showMessage("Restored what you were working on.", 5000)

    def _frame_edited(self, row: int) -> bool:
        """Was the frame in this row of the filmstrip adjusted by hand?"""
        if not 0 <= row < len(self.photos):
            return False
        ph = self.photos[row]
        return bool(ph.history) and not same_edits(ph.history[0], ph.settings)

    def unexported(self) -> list[Photo]:
        """Frames with adjustments that have not been exported as they are now."""
        out = []
        for ph in self.all_photos():
            edited = bool(ph.history) and not same_edits(ph.history[0], ph.settings)
            if ph.ok and edited and ph.exported != edit_key(ph.settings):
                out.append(ph)
        return out

    def _ask_quit(self, text: str, detail: str, can_save: bool) -> str:
        """The question before quitting: "save", "quit" or "cancel"."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.NoIcon)
        box.setText(text)
        box.setInformativeText(detail)
        save = box.addButton("Save and Quit", QMessageBox.AcceptRole) if can_save else None
        quit_btn = box.addButton("Quit Without Saving" if can_save else "Quit Anyway", QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel" if can_save else "Go Back", QMessageBox.RejectRole)
        box.setDefaultButton(save or cancel)
        box.exec()
        clicked = box.clickedButton()
        return "save" if clicked is save else "quit" if clicked is quit_btn else "cancel"

    def closeEvent(self, e):
        self._commit_history()
        running = []
        if self._sheets:
            running.append("A contact sheet is still being made: quitting leaves it unfinished.")
        if self._export_left > 0:
            done = self._export_total - self._export_left
            running.append(f"An export is still running: {done} of {self._export_total} files are written, "
                           "quitting stops the rest.")
        dirty = self.session_dirty()
        if dirty or running:
            if dirty:
                text = "Save your session before quitting?"
                n = len(self.unexported())
                lines = ["Saving keeps your rolls and adjustments for next time."]
                if n:
                    lines.append(f"{n} adjusted frame{'s have' if n > 1 else ' has'} not been exported.")
            else:
                text, lines = "Quit now?", []
            answer = self._ask_quit(text, "\n".join(running + lines), can_save=dirty)
            if answer == "cancel" or (answer == "save" and not self.save_session()):
                e.ignore()
                return
        self.export_pool.clear()                      # exports not started yet are dropped
        self._recovery_timer.stop()
        if self._recovery_on:
            self._recovery_path().unlink(missing_ok=True)   # a normal quit: nothing to recover
        e.accept()

    def all_photos(self) -> list[Photo]:
        return [p for g in self.groups for p in g.photos]

    def _leave_frame(self):
        """Before another frame becomes current: an edit made a moment ago is still waiting for
        its pause (undo step, full render, thumbnail). Finish it for the frame it belongs to."""
        p = self.current
        if p is None:
            return
        self._history_timer.stop()
        self._commit_history(p)
        if self._stale_thumb is p and p.ok:
            self.render_thumb(p)               # its filmstrip picture must show the edit
        self._stale_thumb = None

    def show_group(self, g: Group):
        """Put a group's photos in the thumbnail grid and show its details."""
        if self.preview.crop_mode:
            self.toggle_crop(apply=False)
        self._leave_frame()
        for ph in self.photos:          # items of the previous group are about to be deleted
            ph.item = None
        self.group, self.photos, self.current = g, g.photos, None
        self._place_details()
        self.grid.blockSignals(True)
        self.grid.clear()
        for ph in g.photos:
            ph.item = QListWidgetItem()
            ph.item.setToolTip(ph.path.name)
            ph.item.setIcon(self._placeholder())
            self.grid.addItem(ph.item)
            self.paint_thumb(ph)
        self.grid.blockSignals(False)
        self._show_roll_details(g)
        self._refresh_stock_needed()
        if g.photos:
            self.grid.setCurrentRow(0, QItemSelectionModel.ClearAndSelect)
        self.update_nav()
        if not self._typing():
            self.preview.setFocus()

    def _show_roll_details(self, g: Group):
        if g.kind != "roll":
            return
        self.stock_box.setCurrentIndex(max(self.stock_box.findData(g.stock_id), 0))
        if not g.stock_id:                            # not chosen yet: the placeholder, free to type
            self.stock_box.lineEdit().setText("No film stock" if g.no_stock else "")
        self._refresh_stock_needed()
        self.camera_edit.set_value(g.camera)
        self.lens_edit.set_value(g.lens)
        self.date_edit.set_iso(g.date)
        self.place_edit.setText(g.location_label or g.location)
        self.push_box.setCurrentIndex(max(self.push_box.findData(g.push_pull), 0))
        self._update_roll_status(g)

    def _update_roll_status(self, g: Group):
        if g is not self.group or g.kind != "roll":
            return
        if not g.stock_id:
            text = "Choose the film stock: the roll is analysed for it"
        elif g.analyzing:
            text = "Analysing the roll…"
        elif g.analyzed:
            text = f"Roll analysed: one colour baseline for {len(g.photos)} frame{'s' if len(g.photos) != 1 else ''}"
            if g.outliers:
                text += f" ({g.outliers} unusual frame(s) keep their own)"
        else:
            text = "Waiting for the frames to load…"
        self.roll_status.setText(text)

    def _roll_saved(self, widget: QWidget, what: str):
        """A roll detail changed: light the field up, say so, and refresh the frames' defaults."""
        exif_panel.flash(widget)
        g = self.group
        self.statusBar().showMessage(f"✓ Saved {what} for {g.name} ({len(g.photos)} frames)", 2500)
        self.refresh_exif()

    def on_camera(self):
        if self.group and self.group.kind == "roll":
            self.group.camera = self.camera_edit.text().strip()
            self._roll_saved(self.camera_edit, "camera")

    def on_roll_meta(self, widget: QWidget | None = None):
        g = self.group
        if g and g.kind == "roll":
            g.lens = self.lens_edit.text().strip()
            g.date = self.date_edit.iso()                 # the picker only gives real dates
            g.push_pull = int(self.push_box.currentData() or 0)
            if widget is not None:
                what = {self.lens_edit: "lens", self.date_edit: "date", self.push_box: "push / pull"}.get(widget, "roll details")
                self._roll_saved(widget, what)

    def on_roll_place(self, label: str, lat: float, lon: float):
        g = self.group
        if g and g.kind == "roll":
            g.location, g.location_label = f"{lat:.6f}, {lon:.6f}", label
            self._roll_saved(self.place_edit, f"location {label}")

    def on_roll_coords(self, text: str):
        g = self.group
        if g and g.kind == "roll":
            c = engine.parse_gps(text)
            g.location, g.location_label = (f"{c[0]:.6f}, {c[1]:.6f}" if c else ""), ""
            self._roll_saved(self.place_edit, "location")

    def on_stock_changed(self, _index: int):
        g = self.group
        if not g or g.kind != "roll":
            return
        stock_id = self.stock_box.currentData() or ""
        was_chosen = g.film_chosen
        g.no_stock = not stock_id                  # picking "No film stock" is a choice too
        if stock_id == g.stock_id:
            if not was_chosen:                     # nothing changes in the frames: just measure now
                self._refresh_group_item(g)
                self.analyze_roll(g)
        else:
            self._set_roll_stock(g, stock_id)
        if not stock_id:
            self.stock_box.lineEdit().setText("No film stock")
        self._refresh_stock_needed()

    def _refresh_stock_needed(self):
        """The film stock field stands out (accent edge and placeholder) while the roll has none."""
        g = self.group
        need = bool(g and g.kind == "roll" and not g.film_chosen)
        box = self.stock_box
        box.setProperty("needed", need)
        box.setStyleSheet(f"QComboBox {{ border: 1px solid {ACCENT}; }} "
                          f"QLineEdit {{ placeholder-text-color: {ACCENT2}; }}" if need else "")
        self._show_roll_progress()

    def _set_roll_stock(self, g: Group, stock_id: str):
        """A roll's film stock changed: its frames convert for that film, and the roll is measured again."""
        if stock_id == g.stock_id:
            return
        g.stock_id = stock_id
        for ph in g.photos:
            engine.settings_for_stock(ph.settings, g.stock)
            self._commit_history(ph)
            self.render_thumb(ph)
        if self.current in g.photos:
            self.load_controls(self.current.settings)
            self._sync_base = self.current.settings.copy()
            self.request_preview()
        self._refresh_group_item(g)
        self.analyze_roll(g)   # a different film changes what is measured: a running analysis restarts
        self._refresh_stock_needed()

    # ----- Roll analysis ---------------------------------------------------
    def analyze_roll(self, g: Group):
        """Measure the roll for one colour baseline. Only once its film stock is chosen (the film
        decides what is measured). A newer analysis cancels the one running: it stops at its next
        frame and its result is dropped."""
        if g.kind != "roll" or not g.photos:
            return
        if not g.film_chosen:                     # the film first: nothing to measure for yet
            g.analysis_id += 1                    # one running for a film taken back stops
            g.analyzing = g.analyzed = False
            self._refresh_group_item(g)
            self._update_roll_status(g)
            self._show_roll_progress()
            return
        members = [ph for ph in g.photos if not ph.failed]
        frames = [(ph.path, ph.settings.copy()) for ph in members]
        if not frames:
            return
        g.analysis_id += 1
        mine = g.analysis_id
        g.analyzing, g.analyzed = True, False
        g.progress = 0.0
        self._refresh_group_item(g)
        self._update_roll_status(g)
        self.statusBar().clearMessage()           # a message would cover the progress line
        self._show_roll_progress()
        emit = self.signals.roll_progress.emit

        def tick(frac: float):
            if g.analysis_id != mine:             # a newer analysis started: stop here
                raise AnalysisCancelled
            emit(g, frac)

        def job():
            try:
                # reading the white balances is quick (file headers); measuring the frames is the work
                wb = engine.roll_white_balance([path for path, _st in frames],
                                               progress=lambda i, n: tick(0.1 * i / max(n, 1)))
                for _path, st in frames:          # the roll is measured on the white balance it will decode on
                    st.raw_wb = wb
                result = engine.analyze_roll(frames, progress=lambda i, n: tick(0.1 + 0.9 * i / max(n, 1)))
                return mine, result + (members, wb)
            except AnalysisCancelled:
                return mine, None
        self.pool.start(Job(job, lambda r: self.signals.roll.emit(g, r)))

    def on_roll_progress(self, g: Group, frac: float):
        if g.analyzing:
            g.progress = max(g.progress, frac)
            self._show_roll_progress()

    def _show_roll_progress(self):
        """The progress line follows the roll on screen if it is being analysed, else any roll that
        is; a roll on screen with no film yet asks for it there."""
        busy = [g for g in self.groups if g.kind == "roll" and g.analyzing]
        if busy:
            g = self.group if self.group in busy else busy[0]
            self.roll_progress.show_progress(g.name, g.progress)
        elif self.group and self.group.kind == "roll" and not self.group.film_chosen and self.group in self.groups:
            self.roll_progress.show_waiting(self.group.name)
        elif self.roll_progress.waiting:
            self.roll_progress.hide()

    def on_roll_analyzed(self, g: Group, payload):
        mine, result = payload if isinstance(payload, tuple) else (g.analysis_id, payload)
        if mine != g.analysis_id or result is None:   # cancelled by a newer one: that one reports
            return
        g.analyzing = False
        if g not in self.groups:
            self.roll_progress.hide()
            self._show_roll_progress()
            return
        if isinstance(result, str):
            self.roll_progress.hide()
            self.statusBar().showMessage(f"Roll analysis failed: {result}")
        else:
            floors, ceils, slope, outliers, axis, members, wb = result
            g.analyzed, g.outliers = True, sum(outliers)
            unusual = {ph for ph, out in zip(members, outliers) if out}
            for ph in g.photos:
                st = ph.settings
                st.raw_wb = wb                    # every frame of the roll decodes on the same white balance
                if ph in unusual:      # far from the rest of the roll: its own measurement fits it better
                    st.roll_floors = st.roll_ceils = st.roll_axis = st.roll_slope = None
                else:
                    st.roll_floors, st.roll_ceils, st.roll_axis, st.roll_slope = floors, ceils, axis, slope
                self.render_thumb(ph)
            if self.current in g.photos:
                self.request_preview()
            if any(o.kind == "roll" and o.analyzing for o in self.groups):
                self._show_roll_progress()        # another roll is still going
            else:
                self.roll_progress.show_done(g.name)
        self._refresh_group_item(g)
        self._update_roll_status(g)

    def _reanalyze_rolls(self):
        for g in self.groups:
            if g.kind == "roll" and not g.analyzing and g.photos:
                self.analyze_roll(g)

    # ----- Thumbnails ------------------------------------------------------
    def _placeholder(self) -> QIcon:
        pix = QPixmap(THUMB_SIZE, THUMB_SIZE * 2 // 3)
        pix.fill(QColor("#2a2c31"))
        return QIcon(pix)

    def render_thumb(self, p: Photo):
        if p in self._thumb_busy:                 # one at a time per frame: it renders again, newest settings
            self._thumb_again.add(p)
            return
        self._thumb_busy.add(p)
        st = p.settings.copy()
        self.pool.start(Job(lambda: engine.thumbnail(p.path, st, THUMB_SIZE * 2),
                            lambda r: self.signals.thumb.emit(p, r)))

    def on_thumb(self, p: Photo, result):
        self._thumb_busy.discard(p)
        if p in self._thumb_again:
            self._thumb_again.discard(p)
            self.render_thumb(p)
        if isinstance(result, str):
            p.failed = True
            if p.item is not None:
                p.item.setText("⚠ " + p.path.name)
                p.item.setToolTip(result)
        else:
            p.ok = True
            p.thumb = result
            self.paint_thumb(p)
        g = p.group
        if g and all(ph.ok or ph.failed for ph in g.photos):
            # a roll starts its shared analysis once every frame has loaded
            if g.kind == "roll" and not g.analyzed and not g.analyzing:
                self.analyze_roll(g)

    def label_item(self, p: Photo):
        """The filmstrip shows only the frame number, drawn on the thumbnail (see paint_thumb)."""
        if p.item is not None and not p.failed and p in self.photos and p.thumb is not None:
            self.paint_thumb(p)

    def paint_thumb(self, p: Photo):
        if p.thumb is None or p.item is None:
            return
        # twice the pixels of its size on screen: sharp on a Retina display, and when it grows
        pix = to_pixmap(p.thumb).scaled(THUMB_SIZE * 2, THUMB_H * 2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pix.setDevicePixelRatio(2)
        if p in self.photos:                              # the frame number in a small corner tag
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            f = QFont()
            f.setPixelSize(10)
            painter.setFont(f)
            text = str(self.photos.index(p) + 1)
            w = painter.fontMetrics().horizontalAdvance(text) + 10
            tag = QRectF(4, pix.height() / 2 - 18, w, 14)
            painter.fillRect(tag, QColor(0, 0, 0, 170))
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(tag, Qt.AlignCenter, text)
            painter.end()
        p.item.setIcon(QIcon(pix))

    # ----- Selecting -------------------------------------------------------
    def _center_current(self):
        if self.current is not None and self.current.item is not None:
            self.grid.scrollToItem(self.current.item, QAbstractItemView.PositionAtCenter)

    def on_select(self, row: int):
        if self.preview.crop_mode:
            self.toggle_crop(apply=False)
        if row < 0 or row >= len(self.photos):
            return
        if self.photos[row] is not self.current:
            self._leave_frame()
        self.current = self.photos[row]
        self.preview.reset_zoom()                         # a new frame starts at fit
        if self._strip_guard.pressed:                     # a click: centred once the mouse is released
            self._strip_guard.center_on_release = True
        else:
            self._center_current()
        if not self.current.history:
            self.current.history, self.current.hindex = [self.current.settings.copy()], 0
        self._sync_base = self.current.settings.copy()
        self.load_controls(self.current.settings)
        self.update_nav()
        p = self.current
        if p.thumb is not None and not self.preview.crop_mode and not self._show_original and not self._show_before:
            self.preview.set_image(to_pixmap(p.thumb), draft=True, fade=True)   # at once, while the sharp one renders
        self.request_preview()                        # sharp, from the prepared decode when there is one
        self.refresh_exif()
        if self.current.note is None:             # checked once per frame, quietly, off the UI thread
            p = self.current
            p.note = ""
            self.pool.start(Job(lambda: engine.scan_note(p.path), lambda r: self.signals.note.emit(p, r)))

    def _prefetch_neighbours(self, p: Photo):
        """Prepare the frames either side of this one in the background (their scans decoded and shrunk
        to the viewer's size): switching to one of them then renders sharp at once."""
        if p not in self.photos:
            return
        i = self.photos.index(p)
        for j in (i + 1, i - 1):
            if 0 <= j < len(self.photos):
                nb = self.photos[j]
                if nb.path in self._prefetched or nb.failed:
                    continue
                self._prefetched.add(nb.path)
                st = nb.settings.copy()
                self.pool.start(Job(lambda nb=nb, st=st: engine.prefetch_preview(nb.path, st, PREVIEW_SIZE),
                                    lambda _r: None))

    def step_image(self, direction: int):
        if self._typing():
            return
        row = self.grid.currentRow() + direction
        if 0 <= row < len(self.photos):
            self.grid.setCurrentRow(row, QItemSelectionModel.ClearAndSelect)   # just this frame

    def update_nav(self):
        row = self.grid.currentRow()
        n = len(self.photos)
        if self.current:
            # the frame (the roll's name is on the left already), then how the scan was exposed
            muted, text = COLORS["muted"], COLORS["text"]
            gap = "&nbsp;&nbsp;&nbsp;"
            icons = (Path(__file__).parent / "icons").as_posix()
            exposure = "".join(
                f"{gap}<img src='{icons}/exp-{kind}.svg' width='12' height='12' style='vertical-align:middle'>&nbsp;"
                f"<span style='color:{muted}'>{value}</span>"
                for kind, value in zip(("shutter", "aperture", "iso"), engine.scan_exposure(self.current.path)) if value)
            note = (f"{gap}<span style='color:{COLORS['warn']}'>{self.current.note}</span>"
                    if self.current.note and self.pref_warnings.isChecked() else "")
            self.title.setText(f"<span style='color:{text}; font-weight:600'>{self.current.path.name}</span>")
            self.title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            self.title.setMaximumWidth(max(120, self.title_box.width() - 30))
            self.title_info.set_info(self._frame_info(row, n, exposure, note), bool(note))
            self.title_info.show()

    def _frame_info(self, row: int, n: int, exposure: str, note: str) -> str:
        """What the info dot beside the file name shows, in three parts: the picture (film and how
        it was developed, camera, lens, date, place), the scan (the camera that photographed the
        negative, its exposure, the file), and the conversion (print look, light). A scan warning first."""
        p = self.current
        muted, text = COLORS["muted"], COLORS["text"]
        sid = self.exif_value(p, "film_stock_id")
        stock = engine.stock_by_id(sid) if sid else None
        g = p.group
        film = f"<span style='color:{COLORS['accent2']}'>not set</span>"
        if stock:
            film = stock["displayName"]
            iso, push = stock.get("iso"), (g.push_pull if g else 0)
            if iso and push:
                film += f"  ·  ISO {iso} {'pushed' if push > 0 else 'pulled'} {abs(push)}, EI {round(iso * 2 ** push)}"
            elif push:
                film += f"  ·  {'pushed' if push > 0 else 'pulled'} {abs(push)} stop{'s' if abs(push) > 1 else ''}"
        place = str(p.exif.get("gps_label") or "") or (g.location_label if g and not p.exif.get("gps") else "") \
            or self.exif_value(p, "gps")
        d = engine.scan_details(p.path)
        w, h = d["size"]
        file_ = "  ·  ".join(x for x in (d["type"], f"{w} × {h} px ({w * h / 1e6:.0f} MP)" if w else "",
                                          f"{d['bytes'] / 1e6:.1f} MB" if d["bytes"] else "") if x)
        st = p.settings
        look = dict(engine.print_looks(st.grayscale)).get(st.paper, "None")
        conversion = "  ·  ".join(x for x in (look if not st.grayscale else "B&W" + (f", {look}" if look != "None" else ""),
                                               "RGB light" if st.light == "rgb" else "white light",
                                               "roll colour" if st.roll_floors else "own colour") if x)

        def section(title, rows):
            body = "".join(f"<tr><td style='color:{muted}; padding-right:14px'>{k}</td><td style='color:{text}'>{v}</td></tr>"
                           for k, v in rows if v)
            return (f"<tr><td colspan='2' style='color:{COLORS['dim']}; font-size:10px; letter-spacing:1px; "
                    f"padding-top:6px'>{title}</td></tr>{body}") if body else ""
        scan_exp = exposure.replace("&nbsp;&nbsp;&nbsp;", "", 1) if exposure else ""
        table = (section("PICTURE", [("Frame", f"{row + 1} of {n}"), ("Film", film),
                                     ("Camera", self.exif_value(p, "camera_model")), ("Lens", self.exif_value(p, "lens_model")),
                                     ("Date", self.exif_value(p, "capture_date")), ("Place", place)])
                 + section("SCAN", [("Camera", d["camera"]), ("Lens", d["lens"]), ("Exposure", scan_exp), ("File", file_)])
                 + section("CONVERSION", [("Look", conversion)]))
        head = f"<div style='margin-bottom:4px'>{note.strip().replace('&nbsp;', '')}</div>" if note else ""
        return f"{head}<table cellspacing='0' cellpadding='2'>{table}</table>"

    def on_note(self, p: Photo, note):
        p.note = "" if note.startswith("ERROR") else note
        if p is self.current:
            self.update_nav()

    def load_controls(self, st: engine.Settings):
        """Show a photo's settings in the controls (without triggering renders)."""
        for key, s in self.steppers.items():
            s.set_value(getattr(st, key))
        for key, s in self.sliders.items():
            s.set_value(getattr(st, key))
        self.sliders["color_noise"].setVisible(not st.grayscale)    # B&W has no colour noise to remove
        self.mixer.setVisible(not st.grayscale)                     # nor colours to mix
        self._fill_paper_box(st)
        self.wb_box.setVisible(not st.grayscale)                    # nor a white balance to set
        if self.bw_joke.isHidden() != (not st.grayscale):
            before = self.more_body.height()
            self.bw_joke.setVisible(st.grayscale)
            self.refine_pages.updateGeometry()
            self._fit_more(before)
        for key, f in self.hsl_sliders.items():
            f.set_value(getattr(st, key)[self._hsl_band])
        self._mark_hsl_dots(st)
        self.straighten.set_value(st.fine_rotation)
        for cb, val in ((self.autocrop_cb, st.auto_crop),):
            cb.blockSignals(True)
            cb.setChecked(val)
            cb.blockSignals(False)
        self.autocrop_refresh.setEnabled(st.auto_crop)
        self._update_undo_buttons()

    # ----- Editing ---------------------------------------------------------
    def on_setting_changed(self):
        if not self.current:
            return
        st = self.current.settings
        for key, s in self.steppers.items():
            setattr(st, key, s.value())
        for key, s in self.sliders.items():
            setattr(st, key, s.value())
        self._apply_to_selection()
        self.edited()

    # ----- Refine: Light · Color · Finish ----------------------------------
    def _refine_tab(self, i: int):
        """Show a Refine page; the section grows or shrinks smoothly to its height."""
        if self.refine_bar.current != i:
            self.refine_bar.select(i)                 # comes back here through `chosen`
            return
        if self.refine_pages.currentIndex() == i:
            return
        body = self.more_body
        before = body.height()
        self.refine_pages.setCurrentIndex(i)
        self.refine_pages.updateGeometry()
        if not (self.more_btn.isChecked() and body.isVisible()):
            return
        self._fit_more(before)

    def _fit_more(self, before: int | None = None):
        """The open Refine section glides to its content's height (a page switch, the mixer hiding)."""
        body = self.more_body
        if not (self.more_btn.isChecked() and body.isVisible()):
            return
        before = body.height() if before is None else before
        self._refine_inner.layout().activate()
        body._place()                                 # the new page at its full height now, not an event later
        after = body.sizeHint().height()
        if abs(after - before) < 2:
            return
        self._fold_anim.stop()
        self._fold_anim.setDuration(260)
        self._fold_anim.setStartValue(before)
        self._fold_anim.setEndValue(after)
        body.setMaximumHeight(before)
        self._fold_anim.start()

    def _fill_paper_box(self, st: engine.Settings):
        """The print looks of this kind of film (colour or B&W), the frame's own chosen."""
        if self._paper_grayscale != st.grayscale:
            self._paper_grayscale = st.grayscale
            self.paper_box.blockSignals(True)
            self.paper_box.clear()
            for key, name in engine.print_looks(st.grayscale):
                self.paper_box.addItem(name, key)
            self.paper_box.blockSignals(False)
        i = self.paper_box.findData(st.paper)
        self.paper_box.blockSignals(True)
        self.paper_box.setCurrentIndex(max(i, 0))      # a look of the other kind shows (and renders) as None
        self.paper_box.blockSignals(False)

    def _on_paper(self, _i: int = 0):
        if not self.current:
            return
        self.current.settings.paper = self.paper_box.currentData() or "neutral"
        self._apply_to_selection()
        self.edited()

    def _hsl_band_chosen(self, i: int, load: bool = True):
        """A colour of the mixer chosen: its three sliders show its values; Hue fades to its neighbours."""
        self._hsl_band = i
        for k, d in enumerate(self.hsl_dots):
            d.setChecked(k == i)
        for key, colours in hsl_gradients(i).items():
            self.hsl_sliders[key].gradient = colours
        st = self.current.settings if (load and self.current) else engine.Settings()
        for key, f in self.hsl_sliders.items():
            f.set_value(getattr(st, key)[i])
            f.update()

    def _on_hsl(self, key: str):
        if not self.current:
            return
        st = self.current.settings
        values = list(getattr(st, key))
        values[self._hsl_band] = self.hsl_sliders[key].value()
        setattr(st, key, tuple(values))
        self._mark_hsl_dots(st)
        self._apply_to_selection()
        self.edited()

    def _mark_hsl_dots(self, st: engine.Settings):
        for k, d in enumerate(self.hsl_dots):
            d.edited = any(getattr(st, key)[k] != 0.0 for key in self.hsl_sliders)
            d.update()

    # ----- Batch editing, undo/redo ----------------------------------------
    def selected_others(self) -> list[Photo]:
        rows = sorted(self.grid.row(i) for i in self.grid.selectedItems())
        return [self.photos[r] for r in rows if 0 <= r < len(self.photos) and self.photos[r] is not self.current]

    def _apply_to_selection(self):
        """With several frames selected, an adjustment changes all of them (crop/rotation excepted)."""
        if not self.current or self._sync_base is None:
            return
        st, base = self.current.settings, self._sync_base
        changed = [f for f in st.__dataclass_fields__
                   if f not in NOT_EDITS and f not in PER_FRAME and getattr(st, f) != getattr(base, f)]
        self._sync_base = st.copy()
        others = self.selected_others()
        if not changed or not others:
            return
        for ph in others:
            for f in changed:
                setattr(ph.settings, f, getattr(st, f))
            self._pending_batch.add(ph)           # one undo step with this frame's
            self._dirty_thumbs.add(ph)            # re-rendered once the slider rests
        self._notice_batch(len(others) + 1)

    def _notice_batch(self, n: int):
        """Say so on the photo when a change reaches more than this frame."""
        if n > 1:
            self.toast.show_text(f"Updating {n} photos")

    # ----- Sync Roll -------------------------------------------------------
    def _sync_keys(self) -> list[str]:
        import json
        try:
            keys = json.loads(pref("sync") or "null")
        except ValueError:
            keys = None
        return [k for k in keys if k in SYNC_GROUPS] if isinstance(keys, list) else list(SYNC_DEFAULT)

    def _sync_choices(self):
        """The gear beside Sync Roll: tick what Sync sends (remembered)."""
        import json
        pop = QWidget(self, Qt.Popup)
        pop.setAttribute(Qt.WA_TranslucentBackground)      # the rounded plate below shows its corners
        outer = QVBoxLayout(pop)
        outer.setContentsMargins(0, 0, 0, 0)
        plate = QFrame()
        plate.setObjectName("syncPop")
        outer.addWidget(plate)
        lay = QVBoxLayout(plate)
        lay.setContentsMargins(18, 16, 20, 16)
        lay.setSpacing(11)
        title = QLabel("SYNC ROLL SENDS")
        title.setObjectName("rowLabel")
        lay.addWidget(title)
        lay.addSpacing(2)
        chosen = self._sync_keys()

        def save():
            set_pref("sync", json.dumps([k for k, cb in boxes.items() if cb.isChecked()]))
        boxes = {}
        for key, (label, _fields) in SYNC_GROUPS.items():
            cb = QCheckBox(label)
            cb.setChecked(key in chosen)
            cb.toggled.connect(lambda _on: save())
            boxes[key] = cb
            lay.addWidget(cb)
        lay.addSpacing(2)
        hint = QLabel("From this frame to every frame of the group.")
        hint.setObjectName("hint")
        lay.addWidget(hint)
        pop.adjustSize()
        at = self.sync_gear.mapToGlobal(QPoint(0, 0))
        pop.move(at.x(), at.y() - pop.height() - 6)
        self._sync_pop = pop                       # kept while it is open
        pop.show()

    def sync_roll(self):
        """Send the chosen edits of this frame to every other frame of its group."""
        p = self.current
        g = p.group if p else None
        if not g:
            return
        keys = self._sync_keys()
        fields = [f for k in keys for f in SYNC_GROUPS[k][1]]
        if not fields:
            self.statusBar().showMessage("Nothing to sync: tick something in Sync's options (the arrow beside Sync).", 5000)
            return
        if self._history_timer.isActive():        # this frame's edit in progress counts
            self._history_timer.stop()
            self._commit_history(p)
        src = p.settings
        others = [ph for ph in g.photos if ph is not p and not ph.failed]
        before, steps = self._seq_now, {}
        for ph in others:
            st = ph.settings
            for f in fields:
                setattr(st, f, getattr(src, f))
            if "framing" in keys:
                st.auto_rect, st.auto_key = None, ""      # an auto crop is found again on each frame
                if st.auto_crop:
                    st.crop = None
                self._autocrop_asked.pop(ph, None)
            self._step(ph)
            steps[ph] = ph.hindex
            self.render_thumb(ph)
        self._record_batch(steps, before)         # Cmd+Z right after undoes the whole sync
        if others:
            self.toast.show_text(f"Synced to {len(others) + 1} photos")

    def edited(self):
        """Call after any edit of the current frame: a draft render now, a full one and one undo
        step after a pause.

        The draft timer is only started when idle, never restarted: restarting it on every
        slider movement would postpone the render until the slider stopped.
        """
        if not self.render_timer.isActive():
            self.render_timer.start()
        if self._dirty_thumbs and not self._thumb_timer.isActive():
            self._thumb_timer.start()             # the other selected frames' thumbnails, while you edit
        self._settle_timer.start()
        self._history_timer.start()
        self._stale_thumb = self.current       # until a full render of this edit has painted it

    def _render_key(self, st: engine.Settings, size: int, uncropped: bool) -> tuple:
        """What a render shows: the frame's settings, its size and the view it was made for."""
        return (repr(st), size, uncropped, self._show_before, self._show_original)

    def _settled(self):
        p = self.current
        sharp = p is not None and self._shown_key == self._render_key(p.settings, PREVIEW_SIZE, self.preview.crop_mode)
        if not sharp or self.preview_busy or self.preview_pending:
            self.request_preview()                # the edits came to rest: their sharp render
        self._check_sharp_fit()                   # full view: and its full-resolution picture
        self._flush_thumbs()

    def _flush_thumbs(self):
        for ph in self._dirty_thumbs:
            self.render_thumb(ph)
        self._dirty_thumbs.clear()

    def _commit_history(self, p: Photo | None = None):
        p = p or self.current
        if p is None:
            return
        self._update_saved_mark()                 # the save icon's dot: changes not saved yet
        before = self._seq_now
        self._step(p)
        if p is self.current and self._pending_batch:   # the selected frames changed with it: one step
            frames = {p: p.hindex}
            for ph in self._pending_batch:
                self._step(ph)
                frames[ph] = ph.hindex
            self._pending_batch.clear()
            self._record_batch(frames, before)
        if p is self.current:
            self._update_undo_buttons()

    def _step(self, p: Photo):
        """One undo step for this frame, if its settings changed since its last one."""
        if not p.history:
            p.history, p.hindex = [p.settings.copy()], 0
        elif not same_edits(p.history[p.hindex], p.settings):
            p.history = p.history[:p.hindex + 1] + [p.settings.copy()]
            p.hindex = len(p.history) - 1
            self._history_seq += 1
            self._seq_now = self._history_seq

    def _record_batch(self, frames: dict, before: int):
        """Several frames changed together: Cmd+Z undoes it on all of them, group after group.
        `before` is where undo stood before the group, so undoing it lands exactly there."""
        frames = {ph: i for ph, i in frames.items() if i > 0}
        if len(frames) < 2 and not (frames and self.current not in frames):
            return                                # this frame alone: its own undo covers it
        self._batches = self._batches[-49:] + [{"frames": frames, "seq": self._seq_now, "prev": before}]
        self._redo_batches.clear()

    def _restore_photo(self, p: Photo, idx: int):
        snap = p.history[idx].copy()
        for f in NOT_EDITS:                       # keep what was measured since
            setattr(snap, f, getattr(p.settings, f))
        p.settings, p.hindex = snap, idx
        if p is self.current:
            self._sync_base = snap.copy()
            self.load_controls(snap)
            self.request_preview()
        elif p.group is not None and p in p.group.photos:
            self.render_thumb(p)

    def _restore(self, idx: int):
        self._restore_photo(self.current, idx)

    def undo(self):
        if not self.current:
            return
        if self._history_timer.isActive():        # finish the edit in progress first
            self._history_timer.stop()
            self._commit_history()
        b = self._batches[-1] if self._batches else None
        if b and b["seq"] == self._seq_now:       # the last thing done reached several frames
            self._batches.pop()
            for ph, idx in b["frames"].items():
                if ph.hindex == idx:
                    self._restore_photo(ph, idx - 1)
            self._seq_now = b["prev"]
            self._redo_batches.append(b)
            self.toast.show_text(f"Undone on {len(b['frames'])} photos")
            return
        if self.current.hindex > 0:
            self._restore(self.current.hindex - 1)

    def redo(self):
        b = self._redo_batches[-1] if self._redo_batches else None
        if b and b["prev"] == self._seq_now:
            self._redo_batches.pop()
            for ph, idx in b["frames"].items():
                if ph.hindex == idx - 1 and idx < len(ph.history):
                    self._restore_photo(ph, idx)
            self._seq_now = b["seq"]
            self._batches.append(b)
            self.toast.show_text(f"Redone on {len(b['frames'])} photos")
            return
        if self.current and self.current.hindex < len(self.current.history) - 1:
            self._restore(self.current.hindex + 1)

    def _update_undo_buttons(self):
        pass                                      # undo / redo live on Cmd+Z / Cmd+Shift+Z only

    def _set_before(self, on: bool):
        self._show_before = on
        self.request_preview()

    def copy_settings(self):
        if self.current:
            self._clipboard = self.current.settings.copy()
            self.statusBar().showMessage("Settings copied. Select frames and press Paste to Selected.")

    def paste_settings(self):
        if not self._clipboard:
            self.statusBar().showMessage("Copy settings from a frame first.")
            return
        targets = self.selected_others() + ([self.current] if self.current else [])
        if self.current and self._history_timer.isActive():
            self._history_timer.stop()
            self._commit_history()
        before, steps = self._seq_now, {}
        for ph in targets:
            new = self._clipboard.copy()
            for f in NOT_EDITS + PER_FRAME:
                setattr(new, f, getattr(ph.settings, f))
            ph.settings = new
            self._step(ph)
            steps[ph] = ph.hindex
            self.render_thumb(ph)
        self._record_batch(steps, before)
        if self.current:
            self._sync_base = self.current.settings.copy()
            self.load_controls(self.current.settings)
            self.request_preview()
        self.statusBar().showMessage(f"Settings pasted to {len(targets)} frame(s).")

    def on_straighten_slider(self):
        if self.current:
            self.preview.reset_zoom()
            st = self.current.settings
            st.fine_rotation = round(self.straighten.value(), 1)
            # instant: turn the picture already on screen; the real render follows after a pause
            self.preview.turn_fills = not (st.crop or (st.auto_crop and st.auto_rect))
            self.preview.angle = st.fine_rotation - self._rendered_angle
            self.preview.refresh()
            self._history_timer.start()
            if engine.gpu_active():                   # quick enough: the real straightened render as you drag
                self.edited()
            else:
                self._straighten_timer.start()

    def _transform_shown(self, t: QTransform):
        """Turn/flip the picture on screen right away; the real render replaces it shortly."""
        if self.preview.source is not None and not self.preview.crop_mode:
            self.preview.set_image(self.preview.source.transformed(t, Qt.SmoothTransformation), fade=True)
            if t.m11() < 0 or abs(t.m12()) > 0.5:     # mirrored or turned: a live tilt changes direction
                self._rendered_angle = -self._rendered_angle if t.m11() < 0 else self._rendered_angle

    def _show_grid(self, on: bool):
        self.preview.grid = on
        self.preview.refresh()

    def _straighten_released(self):
        self._show_grid(False)
        self._straighten_timer.stop()
        self.request_preview()

    def auto_straighten(self):
        if not self.current:
            return
        p, st = self.current, self.current.settings.copy()
        self.auto_straighten_btn.setEnabled(False)
        self.statusBar().showMessage("Measuring the frame edges…")
        self.pool.start(Job(lambda: engine.auto_straighten(p.path, st), lambda r: self.signals.straighten.emit(p, r)),
                        URGENT)

    def on_straightened(self, p: Photo, angle):
        self.auto_straighten_btn.setEnabled(True)
        if isinstance(angle, str) or angle is None:
            self.statusBar().showMessage("No straight frame edge found to level against. Use the slider instead.")
            return
        p.settings.fine_rotation = round(angle, 1)
        if p is self.current:
            self.straighten.set_value(p.settings.fine_rotation)
            self.edited()
        else:
            self._commit_history(p)
        self.statusBar().showMessage(f"Straightened by {angle:+.1f}°")

    def on_autocrop(self, on: bool):
        if self.current:
            self.preview.reset_zoom()
            others = self.selected_others()             # the selected frames too
            for ph in others:
                st = ph.settings
                st.auto_crop = on
                if on:
                    st.crop, st.auto_rect, st.auto_key = None, None, ""
                    self.auto_level(ph)
                else:
                    self._drop_levelling(ph)
                self._pending_batch.add(ph)
                self._dirty_thumbs.add(ph)
            self._notice_batch(len(others) + 1)
            self.current.settings.auto_crop = on
            if on:
                self.current.settings.crop = None   # auto replaces a manual crop
                self.auto_level(self.current)
            else:                                   # off: the frame as it was, the levelling included
                self._drop_levelling(self.current)
                self.straighten.set_value(self.current.settings.fine_rotation)
            self.edited()

    def refresh_auto_crop(self):
        """Look for the auto crop again, on the frame as it is turned and straightened now."""
        p = self.current
        if not p or not p.settings.auto_crop:
            return
        st = p.settings
        st.auto_rect, st.auto_key = None, ""
        self._autocrop_asked.pop(p, None)
        self.auto_level(p)                        # levels it first, unless you straightened it yourself
        self.edited()
        self.statusBar().showMessage("Auto crop: looking for the picture edges again.", 3000)

    def auto_level(self, p: Photo, automatic: bool = False):
        """Auto crop also levels a tilted scan (unless you straightened it yourself): a crop of a
        tilted frame would keep slivers of the light around the film in its corners.
        `automatic`: the frame was just loaded, so the levelling is part of its automatic conversion."""
        if abs(p.settings.fine_rotation) > 0.05:
            self._ensure_auto_crop(p)             # straightened by you: no levelling, just the crop
            return
        st = p.settings.copy()
        self._levelling.add(p)
        self.pool.start(Job(lambda: engine.auto_level(p.path, st), lambda r: self.signals.level.emit(p, r, automatic)),
                        URGENT if p is self.current else 0)

    def on_levelled(self, p: Photo, angle, automatic: bool):
        self._levelling.discard(p)
        st = p.settings
        if p is self.current:
            self._ensure_auto_crop(p)             # its crop is looked for once the levelling is known
        if isinstance(angle, str) or angle is None or not st.auto_crop or st.crop or abs(st.fine_rotation) > 0.05:
            return                                # nothing to level, or you changed your mind meanwhile
        if p.group is None or p not in p.group.photos:
            return                                # removed from the list meanwhile
        untouched = len(p.history) == 1 and same_edits(p.history[0], st)
        st.fine_rotation = p.levelled = angle
        if automatic and untouched:               # where undo ends, and not an edit to be asked about on quit
            p.history[0].fine_rotation = angle
        last = p.history[p.hindex] if p.history else None
        if last is not None and p.hindex > 0 and last.auto_crop and abs(last.fine_rotation) < 0.05:
            last.fine_rotation = angle            # one undo step with the Auto crop that asked for it
        if p is self.current:
            self.straighten.set_value(angle)
            self.edited()
            self.statusBar().showMessage(f"Auto crop: levelled by {angle:+.1f}° to clear the borders.", 5000)
        else:
            self._commit_history(p)
            self.render_thumb(p)

    # ----- Auto crop, found once ----------------------------------------------
    # Auto crop is looked for once, when it is turned on (or the frame loads with it on), and then
    # kept: turning and mirroring move it with the picture, straightening leaves it where it is.
    # Only Refresh (the icon beside Auto, or its key) looks again, on the frame as it is framed now.
    def _ensure_auto_crop(self, p: Photo):
        st = p.settings
        if not st.auto_crop or st.crop or p in self._levelling:
            return
        key = engine.auto_crop_key(st)
        if st.auto_rect or self._autocrop_asked.get(p) == key:
            return
        self._autocrop_asked[p] = key
        snap = st.copy()
        self.pool.start(Job(lambda: engine.find_auto_crop(p.path, snap),
                            lambda r: self.signals.autocrop.emit(p, key, r)), URGENT)

    def _auto_crop_rect(self, p: Photo) -> QRectF | None:
        """The frame's auto crop as a crop-tool rectangle, when it is known for its framing now."""
        st = p.settings
        if st.auto_crop and not st.crop and st.auto_rect:
            x0, y0, x1, y1 = st.auto_rect
            return QRectF(x0, y0, x1 - x0, y1 - y0)
        return None

    def on_auto_crop_found(self, p: Photo, key: str, result):
        if self._autocrop_asked.get(p) == key:
            del self._autocrop_asked[p]
        st = p.settings
        if isinstance(result, str) or result is None or not st.auto_crop or st.crop:
            return                                # no border found (NegPy keeps the whole frame), or Auto went off
        rect, found_key = result
        if engine.auto_crop_key(st) != found_key:
            if p is self.current:
                self._ensure_auto_crop(p)         # framed differently meanwhile: look again
            return
        changed = tuple(rect) != tuple(st.auto_rect or ())
        st.auto_rect, st.auto_key = rect, found_key
        if changed and p is self.current and not self.preview.crop_mode:
            self.request_preview()                # its exact framing now, not at the next slider touch
        pv = self.preview
        if p is self.current and pv.crop_mode and self._crop_seed is not None and pv._grab is None \
                and pv.crop == self._crop_seed:   # the crop tool opened before it was known: start from it
            self._crop_seed = pv.crop = self._auto_crop_rect(p)
            pv.format_area = QRectF(pv.crop)
            pv.refresh()

    def rotate(self, quarter_turns: int):
        """+1 = left (counter-clockwise), -1 = right. The crop box turns with the image."""
        if not self.current:
            return
        self.preview.reset_zoom()

        def turn(r):
            if r is None:
                return None
            x0, y0, x1, y1 = r
            return (y0, 1 - x1, y1, 1 - x0) if quarter_turns > 0 else (1 - y1, x0, 1 - y0, x1)
        frames = [self.current] + self.selected_others()      # the selected frames turn too
        for ph in frames:
            st = ph.settings
            st.rotation = (st.rotation + quarter_turns) % 4
            st.crop, st.analysis_rect, st.auto_rect = turn(st.crop), turn(st.analysis_rect), turn(st.auto_rect)
            if ph is not self.current:
                self._pending_batch.add(ph)
                self._dirty_thumbs.add(ph)
        self._notice_batch(len(frames))
        self._transform_shown(QTransform().rotate(-90 * quarter_turns))
        self.edited()
        self._history_timer.stop()                # a turn is one undo step of its own, however quick
        self._commit_history()

    def mirror(self):
        if not self.current:
            return
        self.preview.reset_zoom()
        flip = lambda r: None if r is None else (1 - r[2], r[1], 1 - r[0], r[3])
        frames = [self.current] + self.selected_others()      # the selected frames flip too
        for ph in frames:
            st = ph.settings
            st.mirror = not st.mirror
            st.crop, st.analysis_rect, st.auto_rect = flip(st.crop), flip(st.analysis_rect), flip(st.auto_rect)
            if ph is not self.current:
                self._pending_batch.add(ph)
                self._dirty_thumbs.add(ph)
        self._notice_batch(len(frames))
        self._transform_shown(QTransform().scale(-1, 1))
        self.edited()
        self._history_timer.stop()                # a flip is one undo step of its own
        self._commit_history()

    def _reset_settings(self, p: Photo) -> engine.Settings:
        """This frame's automatic conversion: what was measured and the film stock are kept."""
        old = p.settings
        new = engine.Settings(flat_field=old.flat_field, light=old.light, raw_wb=old.raw_wb, roll_floors=old.roll_floors,
                              roll_ceils=old.roll_ceils, roll_axis=old.roll_axis, roll_slope=old.roll_slope)
        engine.settings_for_stock(new, p.group.stock if p.group else None)
        new.auto_crop = pref("auto_crop_new")
        return new

    def reset_current(self):
        """Back to the automatic conversion: this frame and every other selected one."""
        if self.current:
            others = self.selected_others()
            for ph in others:
                ph.settings = self._reset_settings(ph)
                if ph.settings.auto_crop:
                    self.auto_level(ph)
                self._pending_batch.add(ph)       # one undo step with this frame's
                self._dirty_thumbs.add(ph)
            self._notice_batch(len(others) + 1)
            new = self._reset_settings(self.current)
            self.current.settings = new
            self._sync_base = new.copy()
            self.load_controls(self.current.settings)
            if new.auto_crop:
                self.auto_level(self.current)
            self.edited()

    # ----- White balance ---------------------------------------------------
    def auto_wb(self):
        """Balance the frame, and every other selected frame, each on its own picture."""
        if not self.current:
            return
        self.autowb_btn.setEnabled(False)
        self.statusBar().showMessage("Balancing colours…")
        self._notice_batch(1 + len(self.selected_others()))
        for p in [self.current] + self.selected_others():
            st = p.settings.copy()
            self._wb_running += 1
            self.pool.start(Job(lambda p=p, st=st: engine.auto_wb(p.path, st), lambda r, p=p: self.signals.wb.emit(p, r)),
                            URGENT if p is self.current else 1)

    def _set_wb_base(self, p: Photo):
        """A white balance set on the current frame alone must not travel to the selected frames with
        the next slider move (that copies whatever differs from this base)."""
        if p is self.current and self._sync_base is not None:
            for f in ("cyan", "magenta", "yellow"):
                setattr(self._sync_base, f, getattr(p.settings, f))

    def on_wb(self, p: Photo, result):
        self._wb_running = max(0, self._wb_running - 1)
        if not self._wb_running:
            self.autowb_btn.setEnabled(True)
        if isinstance(result, str):
            self.statusBar().showMessage(result)
            return
        p.settings.cyan, p.settings.magenta, p.settings.yellow = result
        self._set_wb_base(p)
        if p is self.current:
            self.load_controls(p.settings)
            self.edited()
        else:
            self._commit_history(p)
            self.render_thumb(p)
        self.statusBar().showMessage(f"White balance: C {result[0]:.2f}  M {result[1]:.2f}  Y {result[2]:.2f}")

    # ----- EXIF ---------------------------------------------------------------
    # The roll details on the left are every frame's defaults; the Exif tab sets a frame's own.
    ROLL_DEFAULTS = {"camera_model": "camera", "lens_model": "lens", "capture_date": "date",
                     "gps": "location", "film_stock_id": "stock_id"}

    def exif_targets(self) -> list[Photo]:
        return ([self.current] if self.current else []) + self.selected_others()

    def exif_fallback(self, p: Photo, key: str) -> str:
        attr = self.ROLL_DEFAULTS.get(key)
        return str(getattr(p.group, attr, "") or "") if attr and p.group else ""

    def exif_value(self, p: Photo, key: str) -> str:
        return str(p.exif.get(key) or "") or self.exif_fallback(p, key)

    def exif_fields(self, p: Photo) -> dict:
        """Everything written into this frame's exported file."""
        f = {k: self.exif_value(p, k) for k in exif_panel.KEYS}
        f["push_pull"] = p.group.push_pull if p.group else 0
        return f

    def refresh_exif(self):
        if self.right_stack.currentIndex() != 1:
            return
        ps = self.exif_targets()
        if not ps:
            self.exif_panel.show_values("No frame selected", {})
            return
        values = {}
        for k in exif_panel.KEYS:
            effective = {self.exif_value(p, k) for p in ps}
            own, fallback = str(ps[0].exif.get(k) or ""), self.exif_fallback(ps[0], k)
            if k == "gps":                        # show the picked place's name rather than the numbers
                own = str(ps[0].exif.get("gps_label") or own)
                g = ps[0].group
                fallback = (g.location_label or g.location) if g else ""
            values[k] = (own, fallback, len(effective) == 1)
        n = len(ps)
        info = (f"{n} frames" if n > 1 else
                f"Frame {self.photos.index(ps[0]) + 1}" if ps[0] in self.photos else ps[0].path.stem)
        self.exif_panel.show_values(info, values)

    def on_exif_edited(self, key: str, value):
        if key == "capture_date":
            iso = engine.normalize_date(str(value))
            if iso is None:
                self.statusBar().showMessage(f"“{value}” is not a date I understand. Try 21/06/2024 or 2024-06-21.")
                self.refresh_exif()
                return
            value = iso                           # always written as YYYY-MM-DD: what Photos reads
        if key == "gps" and str(value).strip() and engine.parse_gps(str(value)) is None:
            self.statusBar().showMessage("Those coordinates were not understood. Paste e.g. 35.0116, 135.7681")
            self.refresh_exif()
            return
        if key == "gps" and str(value).strip():
            lat, lon = engine.parse_gps(str(value))
            value = f"{lat:.6f}, {lon:.6f}"          # a pasted link becomes clean coordinates
        ps = self.exif_targets()
        for p in ps:
            if key == "gps":
                p.exif.pop("gps_label", None)     # typed coordinates: no place name to show
            if str(value).strip():
                p.exif[key] = value
            else:
                p.exif.pop(key, None)              # empty: back to the roll's value
        self.refresh_exif()
        self.exif_panel.saved([key], len(ps))

    @staticmethod
    def _apply_exif(p: Photo, fields: dict):
        """Set several fields on a frame; coordinates always travel with their place name (or none)."""
        p.exif.update({k: v for k, v in fields.items() if str(v).strip()})
        if str(fields.get("gps") or "").strip():
            if str(fields.get("gps_label") or "").strip():
                p.exif["gps_label"] = fields["gps_label"]
            else:
                p.exif.pop("gps_label", None)

    def on_exif_apply(self, fields: dict):
        ps = self.exif_targets()
        for p in ps:
            self._apply_exif(p, fields)
        self.refresh_exif()
        self.exif_panel.saved(list(fields), len(ps))

    def exif_copy(self):
        if self.current:
            self._exif_clipboard = {k: self.exif_value(self.current, k) for k in exif_panel.KEYS}
            label = self.current.exif.get("gps_label") or (self.current.group.location_label
                                                          if self.current.group and not self.current.exif.get("gps") else "")
            self._exif_clipboard["gps_label"] = label
            self.statusBar().showMessage("EXIF copied. Select frames and press Paste EXIF.")

    def exif_paste(self):
        if not self._exif_clipboard:
            self.statusBar().showMessage("Copy EXIF from a frame first.")
            return
        self.on_exif_apply(self._exif_clipboard)

    def exif_paste_roll(self):
        """Paste the copied EXIF onto every frame of this roll."""
        if not self._exif_clipboard or not self.group:
            self.statusBar().showMessage("Copy EXIF from a frame first.")
            return
        for p in self.group.photos:
            self._apply_exif(p, self._exif_clipboard)
        self.statusBar().showMessage(f"EXIF pasted onto all {len(self.group.photos)} frames of {self.group.name}.")
        self.refresh_exif()

    def exif_to_roll(self):
        """This frame's values become the roll's details (the defaults of every frame)."""
        p, g = self.current, self.group
        if not p or not g or g.kind != "roll":
            self.statusBar().showMessage("Roll defaults need a roll (single frames have none).")
            return
        g.camera, g.lens, g.date = (self.exif_value(p, k) for k in ("camera_model", "lens_model", "capture_date"))
        stock_id = self.exif_value(p, "film_stock_id")
        g.location = self.exif_value(p, "gps")
        g.location_label = str(p.exif.get("gps_label") or (g.location_label if not p.exif.get("gps") else ""))
        for k in ("camera_model", "lens_model", "capture_date", "film_stock_id", "gps", "gps_label"):
            p.exif.pop(k, None)                  # now the roll's, so not this frame's own
        self._set_roll_stock(g, stock_id)        # the film also decides how the roll converts
        self._refresh_group_item(g)
        self._show_roll_details(g)
        self.statusBar().showMessage(f"Roll details of {g.name} set from frame {self.photos.index(p) + 1}.")
        self.refresh_exif()

    def exif_clear(self):
        ps = self.exif_targets()
        for p in ps:
            p.exif.clear()
        self.statusBar().showMessage(f"{len(ps)} frame(s) back to the roll's details.")
        self.refresh_exif()

    # ----- Zoom -------------------------------------------------------------
    def _settings_key(self, p: Photo):
        return (id(p), repr(p.settings))

    def _update_zoom_100(self):
        """The zoom at which one scan pixel is one screen pixel (refined once the full render arrives)."""
        pv, p = self.preview, self.current
        fit = pv.image_rect()
        if not p or fit.isEmpty():
            return
        dpr = pv.devicePixelRatioF()
        if pv.hires is not None:
            full = max(pv.hires.width(), pv.hires.height())
        else:
            if p.path not in self._src_long:
                self._src_long[p.path] = engine.source_long_edge(p.path)
            full = self._src_long[p.path] or 0
        pv.zoom_100 = (full / dpr) / max(fit.width(), fit.height()) if full else 0.0

    def _update_border(self):
        """The preview's white border: the same share of the picture as on a full-resolution export."""
        pv, p = self.preview, self.current
        frac = 0.0
        if p and pref("border") and pref("border_px") > 0 and not self._show_original:
            if p.path not in self._src_long:
                self._src_long[p.path] = engine.source_long_edge(p.path)
            st = p.settings
            rect = st.crop or (st.auto_rect if st.auto_crop else None) or (0.0, 0.0, 1.0, 1.0)
            long_px = self._src_long[p.path] * max(rect[2] - rect[0], rect[3] - rect[1])
            frac = pref("border_px") / long_px if long_px else 0.0
        if abs(frac - pv.border) > 1e-9:
            pv.border = frac
            pv.update()

    def toggle_original(self):
        """Original (and its key): the scan before conversion, for comparing; again to go back."""
        self._show_original = not self._show_original
        self._fade_next = True
        self._mark(self.original_btn, self._show_original)
        self.preview.reset_zoom()
        self._update_border()
        self.request_preview()
        self.statusBar().showMessage("Original scan: before conversion." if self._show_original else "", 0 if self._show_original else 1)

    def toggle_border(self):
        """O: the white border on or off (the switch in Settings follows)."""
        on = not pref("border")
        set_pref("border", on)
        if hasattr(self, "pref_border"):
            self.pref_border.blockSignals(True)
            self.pref_border.setChecked(on)
            self.pref_border.setText("On" if on else "Off")
            self.pref_border.blockSignals(False)
        self._update_border()
        self.statusBar().showMessage(f"White border {'on' if on else 'off'}.", 2000)

    def on_zoom(self, z: float):
        pv = self.preview
        self._update_zoom_100()
        if z <= 1.0:
            if self._fullscreen:                          # full view keeps its full-resolution picture
                if self._hires_key == self._settings_key(self.current) if self.current else False:
                    pv.fit_hires = pv.hires
                self._check_sharp_fit()
                return
            if pv.hires is not None or self._hires_key is not None:
                pv.hires, self._hires_key = None, None
                self.zoom_pool.start(Job(engine.release_cache, lambda r: None))   # free the full decode
            return
        need = pv.view_rect().width() * pv.devicePixelRatioF() > (pv.source.width() if pv.source else 0) * 1.02
        if need and self.current and self._hires_key != self._settings_key(self.current):
            self._request_hires()

    def _request_hires(self):
        p = self.current
        if not p or self._hires_busy:
            return
        self._hires_busy = True
        self.preview.loading = True
        self.preview.refresh()
        key, st = self._settings_key(p), p.settings.copy()
        self.zoom_pool.start(Job(lambda: engine.preview_full(p.path, st), lambda r: self.signals.hires.emit(p, key, r)))

    def on_hires(self, p: Photo, key, result):
        self._hires_busy = False
        pv = self.preview
        if p is not self.current or (pv.zoom <= 1.0 and not self._sharp_fit_wanted()):
            pv.loading = False
            return
        if key != self._settings_key(p):                  # edited meanwhile: render again
            self._request_hires()
            return
        pv.loading = False
        if not isinstance(result, str):
            was_100 = abs(pv.zoom - pv.zoom_100) < 0.02 if pv.zoom_100 else False
            pv.hires, self._hires_key = to_pixmap(result), key
            pv.fit_hires = pv.hires if pv.zoom <= 1.0 else None
            self._update_zoom_100()
            if was_100:                                   # stay at an exact 100% now that it is known
                pv.zoom_at(pv.zoom_100)
        pv.refresh()

    def toggle_zoom_key(self):
        """Z: Fit <-> 100%, around the mouse when it is over the picture."""
        pv = self.preview
        pos = pv.mapFromGlobal(QCursor.pos())
        self._update_zoom_100()
        pv.toggle_zoom(QPointF(pos) if pv.rect().contains(pos) else None)

    # ----- Dust healing -----------------------------------------------------
    @staticmethod
    def _geom_key(st: engine.Settings) -> tuple:
        return (st.rotation, round(st.fine_rotation, 2), st.mirror, st.crop, st.auto_crop)

    def toggle_heal(self):
        """H: click dust specks to heal them. Each click is a search area: only what stands out is repaired."""
        if not self.current:
            return
        on = not self.preview.heal_mode
        if on:
            if self.preview.crop_mode:
                self.toggle_crop(apply=False)
            if self.preview.picking:
                self.stop_pick()
            if self.preview.area_mode:
                self.toggle_heal_area()
        self.preview.heal_mode = on
        self.preview.setCursor(Qt.CrossCursor)
        if not on:
            self.preview._update_cursor()
            self._heal_queue = []                 # clicks still waiting for the map belong to this session
        self.heal_btn.setText("Done Healing" if on else "Heal Dust")
        self.heal_btn.setProperty("on", on)
        self.heal_btn.style().unpolish(self.heal_btn)
        self.heal_btn.style().polish(self.heal_btn)
        self._tool_lock(on, ("heal", "zoom", "escape", "undo", "redo"), self.heal_btn)
        if on:
            self.statusBar().showMessage("Heal dust: click each speck · Alt+scroll for brush size · zoom with "
                                         f"{self.key_for('zoom') or 'a click'} or {keys.MOD}+scroll · {self.key_for('heal') or 'the button'} or Esc when done")
            self._ensure_uv()
        else:
            self.statusBar().clearMessage()
        self._refresh_heal_marks()

    def _ensure_uv(self):
        """Measure where the shown picture's pixels sit in the raw scan (again after a crop or turn)."""
        p = self.current
        if not p or self._uv_busy:
            return
        key = self._geom_key(p.settings)
        if self._uv and self._uv[0] is p and self._uv[1] == key:
            return
        self._uv_busy = True
        st = p.settings.copy()
        self.pool.start(Job(lambda: engine.uv_grid(p.path, st), lambda r: self.signals.uv.emit(p, key, r)), URGENT)

    def on_uv(self, p: Photo, key, grid):
        self._uv_busy = False
        if isinstance(grid, str):
            self.statusBar().showMessage("Could not map the picture for healing: " + grid)
            return
        self._uv = (p, key, grid)
        if p is not self.current or key != self._geom_key(p.settings):
            self._ensure_uv()                     # the frame or its framing changed meanwhile
            return
        queue, self._heal_queue = self._heal_queue, []
        for pos in queue:
            self._add_heal(pos)
        self._refresh_heal_marks()
        shapes, self._area_queue = self._area_queue, []
        for shape in shapes:
            self._add_area(shape)
        self._refresh_area_marks()

    def _uv_ready(self) -> bool:
        p = self.current
        return bool(p and self._uv and self._uv[0] is p and self._uv[1] == self._geom_key(p.settings))

    def on_heal_click(self, pos: QPointF):
        if self._uv_ready():
            self._add_heal(pos)
        else:
            self._heal_queue.append(pos)
            self._ensure_uv()

    def _add_heal(self, pos: QPointF):
        rx, ry = engine.raw_from_view(self._uv[2], pos.x(), pos.y())
        st = self.current.settings
        st.heal_spots = tuple(st.heal_spots) + ((round(rx, 5), round(ry, 5), round(self.heal_size, 1)),)
        self.edited()
        self._history_timer.stop()
        self._commit_history()                    # every spot is its own undo step
        self._refresh_heal_marks()
        self.statusBar().showMessage(f"{len(st.heal_spots)} spot(s) healed on this frame · {keys.MOD}+Z removes the last")

    def _brush_radius_px(self, size: float) -> float:
        """A heal size (diameter at 1600 px of the raw scan) as a radius on screen."""
        r = self.preview.image_rect()
        if not self._uv_ready() or r.isEmpty():
            return max(4.0, size / 1600 * max(r.width(), r.height()) / 2)
        grid = self._uv[2]
        span = max(float(np.ptp(grid[..., 0])), float(np.ptp(grid[..., 1])), 1e-3)   # how much of the scan shows
        return max(3.0, size / 1600 * max(r.width(), r.height()) / span / 2)

    def on_brush_wheel(self, direction: int):
        self.heal_size = float(min(max(self.heal_size * (1.15 if direction > 0 else 1 / 1.15), 6.0), 160.0))
        self._refresh_heal_marks()

    def _refresh_heal_marks(self):
        pv = self.preview
        if not pv.heal_mode or not self.current:
            pv.heal_marks = []
            pv.refresh()
            return
        if not self._uv_ready():
            self._ensure_uv()
            pv.heal_marks = []
        else:
            grid = self._uv[2]
            pv.heal_marks = [(QPointF(*engine.view_from_raw(grid, x, y)), self._brush_radius_px(size))
                             for x, y, size in self.current.settings.heal_spots]
        pv.brush_px = self._brush_radius_px(self.heal_size)
        pv.refresh()

    # ----- Cleaning an area: dust and scratches found for you, but only where you say ----------
    AREA_POINTS = 64          # a drawn shape is kept as at most this many corners

    def toggle_heal_area(self):
        """A: draw around a part of the photo; what looks like dust or a scratch inside it is repaired."""
        if not self.current:
            return
        pv = self.preview
        on = not pv.area_mode
        if on:
            if pv.crop_mode:
                self.toggle_crop(apply=False)
            if pv.picking:
                self.stop_pick()
            if pv.heal_mode:
                self.toggle_heal()
        pv.area_mode, pv._lasso = on, None
        pv.setCursor(Qt.CrossCursor)
        if not on:
            pv._update_cursor()
            self._area_queue = []
        self.area_btn.setText("Done Cleaning" if on else "Clean Area")
        self._mark(self.area_btn, on)
        self._tool_lock(on, ("heal_area", "zoom", "escape", "undo", "redo"), self.area_btn)
        if on:
            self.statusBar().showMessage("Clean area: draw around the dust or scratches · right-click an area to remove it · "
                                         f"{keys.MOD}+Z undoes the last · {self.key_for('heal_area') or 'the button'} or Esc when done")
            self._ensure_uv()
        else:
            self.statusBar().clearMessage()
        self._refresh_area_marks()

    def on_area_drawn(self, shape: list):
        if self._uv_ready():
            self._add_area(shape)
        else:
            self._area_queue.append(shape)
            self._ensure_uv()

    def _add_area(self, shape: list):
        step = max(1, -(-len(shape) // self.AREA_POINTS))        # thin a long, wobbly line out
        grid = self._uv[2]
        raw = tuple((round(x, 5), round(y, 5)) for x, y in
                    (engine.raw_from_view(grid, q.x(), q.y()) for q in shape[::step]))
        if len(raw) < 3:
            return
        st = self.current.settings
        st.heal_areas = tuple(st.heal_areas) + (raw,)
        self.edited()
        self._history_timer.stop()
        self._commit_history()                    # every area is its own undo step
        self._refresh_area_marks()
        n = len(st.heal_areas)
        self.statusBar().showMessage(f"{n} area{'s' if n > 1 else ''} cleaned on this frame · right-click one to remove it")

    def on_area_removed(self, pos: QPointF):
        """Right-click: the area under the pointer goes (the topmost one, if they overlap)."""
        if not self.current:
            return
        for i in range(len(self.preview.area_marks) - 1, -1, -1):
            if QPolygonF(self.preview.area_marks[i]).containsPoint(pos, Qt.OddEvenFill):
                st = self.current.settings
                st.heal_areas = tuple(a for j, a in enumerate(st.heal_areas) if j != i)
                self.edited()
                self._history_timer.stop()
                self._commit_history()
                self._refresh_area_marks()
                self.statusBar().showMessage("Area removed: that part of the photo is as it was scanned.", 4000)
                return

    def _refresh_area_marks(self):
        pv = self.preview
        if not pv.area_mode or not self.current:
            pv.area_marks = []
        elif not self._uv_ready():
            self._ensure_uv()
            pv.area_marks = []
        else:
            grid = self._uv[2]
            pv.area_marks = [[QPointF(*engine.view_from_raw(grid, x, y)) for x, y in area]
                             for area in self.current.settings.heal_areas]
        pv.refresh()

    def toggle_pick_wb(self):
        """W: the grey picker on or off."""
        self.toggle_pick("wb")

    def stop_pick(self):
        """Another mode starts, or Esc: whichever picker waits for a click stops."""
        if self.preview.picking:
            self.toggle_pick(self._pick_target)

    def _pick_button(self, target: str):
        return {"wb": self.pick_btn, "temp": self.temp_picker, "mixer": self.mixer_picker}[target]

    def toggle_pick(self, target: str = "wb"):
        """The next click on the photo picks: a spot that should be neutral ("wb"), or the colour
        to tint the shadows or the highlights with. Clicking the lit picker again stops it."""
        if not self.current:
            return
        if self.preview.picking:                  # one picker at a time: stop the one that waits
            was = self._pick_target
            self.preview.picking = False
            self.preview._update_cursor()
            self._mark(self._pick_button(was), False)
            self.statusBar().showMessage("", 1)
            if was == target:
                return
        if self.preview.area_mode:
            self.toggle_heal_area()
        if self.preview.crop_mode:
            self.toggle_crop(apply=False)
        self._pick_target = target
        self.preview.picking = True
        self.preview.setCursor(Qt.CrossCursor)
        self._mark(self._pick_button(target), True)  # lit while waiting for the click
        self.statusBar().showMessage("Click a colour in the photo to adjust it." if target == "mixer" else
                                     "Click something that should be grey or white.", 0)

    def on_picked(self, pos: QPointF):
        target = self._pick_target
        self.toggle_pick(target)                  # one click per pick
        if not self.current:
            return
        if target == "temp":
            self._pick_temp(pos)
            return
        if target == "mixer":
            self._pick_band(pos)
            return
        p, st, pt = self.current, self.current.settings.copy(), (pos.x(), pos.y())
        self.statusBar().showMessage("Balancing on the spot you clicked…")
        self.pool.start(Job(lambda: engine.auto_wb(p.path, st, point=pt), lambda r: self.signals.wb.emit(p, r)), URGENT)

    def _colour_at(self, pos: QPointF):
        """The average colour of a small spot of the picture shown (sRGB 0..255), or None."""
        src = self.preview.source
        if src is None or not self.current:
            return None
        img = src.toImage().convertToFormat(QImage.Format_RGB888)
        a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())
        a = a[:, :img.width() * 3].reshape(img.height(), img.width(), 3)
        x, y = round(pos.x() * (img.width() - 1)), round(pos.y() * (img.height() - 1))
        return a[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].reshape(-1, 3).astype(float).mean(0)

    def _pick_band(self, pos: QPointF):
        """The mixer chooses the colour under the click, ready for Hue, Saturation and Luminance."""
        rgb = self._colour_at(pos)
        if rgb is None:
            return
        band = engine.mixer_band(rgb)
        if band is None:
            self.statusBar().showMessage("That spot is grey: click something with colour.", 4000)
            return
        self._hsl_band_chosen(band)
        self.statusBar().showMessage(f"{HSL_BANDS[band][0]} chosen in the mixer.", 4000)

    def _pick_temp(self, pos: QPointF):
        """Temp and Tint make the spot under the click grey (a small average of the picture shown)."""
        rgb = self._colour_at(pos)
        if rgb is None:
            return
        st = self.current.settings
        temp, tint = engine.neutral_temp_tint(rgb, st.temp, st.tint)
        self.sliders["temp"].set_value(temp)
        self.sliders["tint"].set_value(tint)
        self.on_setting_changed()
        self.statusBar().showMessage("Temp and Tint set from the spot you clicked.", 4000)

    def recalc(self):
        """Measure the film base and white point again, only inside the crop."""
        if not self.current:
            return
        st = self.current.settings
        st.analysis_rect = st.crop
        st.cyan = st.magenta = st.yellow = 1.0
        self.load_controls(st)
        self.auto_wb()

    def reset_wb(self):
        if self.current:
            self._notice_batch(1 + len(self.selected_others()))
            for p in self.selected_others():      # the selected frames too, like the sliders
                p.settings.cyan = p.settings.magenta = p.settings.yellow = 1.0
                self._pending_batch.add(p)
                self._dirty_thumbs.add(p)
            st = self.current.settings
            st.cyan = st.magenta = st.yellow = 1.0
            self._set_wb_base(self.current)
            self.load_controls(st)
            self.edited()

    @staticmethod
    def _mark(b: QPushButton, on: bool):
        """Light a button up while its mode is active (the stylesheet's [on="true"])."""
        b.setProperty("on", on)
        b.style().unpolish(b)
        b.style().polish(b)

    # ----- Crop ------------------------------------------------------------
    # Film formats, as width : height of a landscape frame (the frame's own direction is kept)
    CROP_FORMATS = {"Free": None, "3:2": 3 / 2, "4:3": 4 / 3, "5:4": 5 / 4, "6:7": 7 / 6, "1:1": 1.0, "65:24": 65 / 24}
    CROP_FORMATS_TIPS = {"Free": "Any shape", "3:2": "35 mm (24 × 36) and 6 × 9", "4:3": "6 × 4.5 and half frame",
                         "5:4": "4 × 5 and 8 × 10 sheet film", "6:7": "6 × 7 medium format", "1:1": "6 × 6 square",
                         "65:24": "Panoramic 24 × 65"}

    def set_crop_format(self, name: str, portrait: bool | None = None):
        """Hold the crop frame to a film format (keeping it upright if it is upright now)."""
        for n, b in self.ratio_btns.items():
            b.setChecked(n == name)
        ratio = self.CROP_FORMATS[name]
        if ratio:
            upright = self.preview.crop_is_portrait() if portrait is None else portrait
            ratio = 1 / ratio if upright else ratio
        self.preview.set_ratio(ratio)

    def crop_turn(self):
        """Upright <-> landscape: the same format, turned (the frame is refitted around its centre)."""
        pv = self.preview
        if not pv.crop_mode or not pv.crop:
            return
        name = next((n for n, b in self.ratio_btns.items() if b.isChecked()), "Free")
        c, a = pv.crop, pv._aspect or 1.0
        if name == "Free" or not pv.ratio:           # free: swap the frame's own width and height
            w, h = c.height() / a, c.width() * a
            scale = min(1.0, 1.0 / w, 1.0 / h)
            w, h = w * scale, h * scale
        else:
            pv.ratio = 1 / pv.ratio
            pv.crop = pv._largest(pv._k(), c.center(), pv.format_area)   # as large as the picture allows
            pv.refresh()
            return
        x = min(max(c.center().x() - w / 2, 0.0), 1.0 - w)
        y = min(max(c.center().y() - h / 2, 0.0), 1.0 - h)
        pv.crop = QRectF(x, y, w, h)
        pv.refresh()

    def _crop_room(self, on: bool):
        pv = self.preview
        r = pv.view_rect()
        before = QRectF(QPointF(pv.mapToGlobal(r.topLeft().toPoint())), r.size())
        self._fade_next = True                    # the whole (or the cropped) picture, faded in
        self._crop_room_now(on)
        # once the layout has made room, the picture glides from where it was
        QTimer.singleShot(0, lambda: pv.glide_from(QRectF(QPointF(pv.mapFromGlobal(before.topLeft().toPoint())),
                                                          before.size())))

    def _crop_room_now(self, on: bool):
        """While cropping, the photo takes the room of the rolls column, the filmstrip and the right
        column (all locked while cropping): a bigger picture to crop on."""
        self.left.setVisible(not on)
        self.grid.setVisible(not on)
        self.right_stack.setVisible(not on)

    def crop_restore(self):
        """The crop toolbar's Reset: no crop of your own; auto crop again if new frames get it."""
        p = self.current
        if not p:
            return
        if self.preview.crop_mode:
            self.toggle_crop(apply=False)
        st = p.settings
        st.crop = None
        st.auto_crop = pref("auto_crop_new")
        if st.auto_crop:
            st.auto_rect, st.auto_key = None, ""
            self._autocrop_asked.pop(p, None)
            self.auto_level(p)
        else:
            self._drop_levelling(p)
            self.straighten.set_value(st.fine_rotation)
        self.load_controls(st)
        self.edited()
        self.statusBar().showMessage("Crop reset.", 2000)

    def crop_reset(self):
        if self.preview.crop_mode:
            self.preview.crop = self.preview.format_area = QRectF(0, 0, 1, 1)   # the border included
            self.set_crop_format("Free")

    def _tool_lock(self, on: bool, keys_: tuple = (), keep: QPushButton | None = None):
        """While cropping, healing or cleaning an area, only the photo and that tool work: the rest
        of the window steps back (dimmed, and deaf to the mouse and the keys) until you are done.
        `keys_`: the actions that still answer; `keep`: the tool's own button under the photo."""
        dimmed = [self.left, self.right_stack, self.grid, self.tab_plate, self.export_wrap, self.title]
        dimmed += [b for b in self._bar_btns if b is not keep]
        if on:
            self._locked = {w: w.isEnabled() for w in dimmed}
        for w in dimmed:
            w.setEnabled(False if on else self._locked.get(w, True))
            if on:
                dim = QGraphicsOpacityEffect(w)
                dim.setOpacity(0.22)
                w.setGraphicsEffect(dim)
            else:
                w.setGraphicsEffect(None)
        for sc in self._shortcuts + self._fixed_shortcuts:
            sc.setEnabled(not on or sc.property("aid") in keys_)

    def _drop_levelling(self, p: Photo):
        """Take back the angle Auto crop straightened the frame by (not one you set yourself)."""
        if p.levelled and abs(p.settings.fine_rotation - p.levelled) < 0.05:
            p.settings.fine_rotation = 0.0
        p.levelled = 0.0

    def toggle_crop(self, apply: bool = True):
        if not self.current:
            return
        pv = self.preview
        if not pv.crop_mode:                      # start editing: the frame lies on the whole picture
            if pv.area_mode:
                self.toggle_heal_area()
            if pv.heal_mode:
                self.toggle_heal()
            pv.reset_zoom()
            pv.crop_mode = True
            self._crop_room(True)
            c = self.current.settings.crop
            pv.crop = QRectF(c[0], c[1], c[2] - c[0], c[3] - c[1]) if c else QRectF(0, 0, 1, 1)
            # Auto crop on: start from where it put the frame, not from the whole scan with its border
            self._crop_seed = None
            pv.format_area = QRectF(0, 0, 1, 1)
            if not c and self.current.settings.auto_crop:
                pv.crop = self._auto_crop_rect(self.current) or QRectF(0, 0, 1, 1)
                self._crop_seed = pv.format_area = QRectF(pv.crop)
                self._ensure_auto_crop(self.current)
            self.crop_btn.setText("Apply")
            self.crop_btn.setProperty("go", True)
            self._mark(self.crop_btn, False)
            self._tool_lock(True, ("accept", "crop", "escape"))
            self.crop_btn.setToolTip(f"Pull the corners of the frame, drag inside to move it; {keys.RETURN} applies, Esc cancels")
            self._update_toolbars()
            self.statusBar().showMessage(f"Crop: pull a corner · drag inside to move · pick a film format below · {keys.RETURN} applies · Esc cancels")
            self.request_preview()
            return
        pv.crop_mode = False                      # finish editing
        self._crop_room(False)
        self._tool_lock(False)
        pv._grab = None
        pv._update_cursor()
        self.crop_btn.setText("Edit")
        self.crop_btn.setProperty("go", False)
        self._mark(self.crop_btn, False)
        self._refresh_tips()                      # "Crop the photo", with its key
        self._update_toolbars()
        self.statusBar().clearMessage()
        seed, self._crop_seed = self._crop_seed, None
        if apply and seed is not None and pv.crop is not None and self.current.settings.auto_crop and all(
                abs(a - b) < 0.002 for a, b in zip(pv.crop.getCoords(), seed.getCoords())):
            self.request_preview()                # opened and applied untouched: it stays Auto
            return
        if apply:
            r = pv.crop
            whole = r is None or (r.left() < 0.002 and r.top() < 0.002 and r.right() > 0.998 and r.bottom() > 0.998)
            st = self.current.settings
            st.crop = None if whole else (r.left(), r.top(), r.right(), r.bottom())
            # the crop is now what you set here, "Whole Frame" included: Auto is off
            if whole and st.auto_crop:
                self._drop_levelling(self.current)       # the whole frame, as it was scanned
            st.auto_crop = False
            self.load_controls(st)
            self.edited()
            return
        self.request_preview()

    # ----- Preview ---------------------------------------------------------
    def request_preview(self, draft: bool = False):
        if not self.current:
            return
        if not draft:
            self._ensure_auto_crop(self.current)
        if self.preview_busy:          # render the newest settings when this one ends
            self.preview_pending = True
            self._pending_full = self._pending_full or not draft
            return
        self.preview_busy = True
        self.preview_request += 1
        p, st, rid = self.current, self.current.settings.copy(), self.preview_request
        if self._show_before:                     # the automatic conversion: no adjustments, same framing
            before = engine.Settings()
            for f in NOT_EDITS + PER_FRAME:
                setattr(before, f, getattr(st, f))
            engine.settings_for_stock(before, p.group.stock if p.group else None)
            st = before
        uncropped = self.preview.crop_mode
        # On the graphics card a sharp render costs what the CPU's draft did: edits render sharp,
        # with no soft draft to swap out when the slider stops. A frame switch's draft stays quick.
        sharp_drafts = draft and engine.gpu_active() and not self._show_original
        size = PREVIEW_SIZE if (not draft or sharp_drafts) else PREVIEW_DRAFT
        if self._show_original:
            fn = lambda: engine.original_preview(p.path, st, size)
        else:
            fn = lambda: engine.preview(p.path, st, size, uncropped=uncropped, keep_source=True, gpu=True)
        self._request_key[rid] = self._render_key(st, size, uncropped)
        self._request_angle[rid] = st.fine_rotation
        self._request_draft[rid] = draft
        self._request_geom[rid] = (st.rotation % 4, st.mirror)
        self.preview_pool.start(Job(fn, lambda r: self.signals.preview.emit(p, r, rid)))

    def on_preview(self, p: Photo, result, request_id: int):
        self.preview_busy = False
        geom = self._request_geom.pop(request_id, None)
        if p is self.current and not isinstance(result, str) and geom != (p.settings.rotation % 4, p.settings.mirror):
            # turned or flipped again while this rendered: the screen already shows the newer turn
            # (_transform_shown); showing this would step back. The newest render follows.
            self._request_angle.pop(request_id, None)
            self._request_draft.pop(request_id, None)
            self.preview_pending = True
        elif p is self.current and request_id == self.preview_request:
            if isinstance(result, str):
                self.preview.set_image(None)
                self.preview.setText(f"Could not convert {p.path.name}\n\n{result}")
            else:
                self.preview.setText("")
                self._update_border()
                self.preview.set_image(to_pixmap(result), draft=self._request_draft.get(request_id, False),
                                       fade=self._fade_next)
                self._fade_next = False
                self._shown_key = self._request_key.get(request_id)
                # this render already contains its rotation; show only what changed since
                self._rendered_angle = self._request_angle.get(request_id, 0.0)
                self.preview.angle = p.settings.fine_rotation - self._rendered_angle
                if not self._request_draft.get(request_id, False):
                    self._prefetch_neighbours(p)
                if not self.preview.crop_mode and not self._show_before and not self._show_original:
                    p.thumb = result
                    self.paint_thumb(p)
                    if not self._request_draft.get(request_id, False) and not self._settle_timer.isActive() \
                            and not self.preview_pending:
                        self._stale_thumb = None    # the finished edit is on the filmstrip
                if self.preview.heal_mode:
                    self._refresh_heal_marks()      # undo, a new frame or a new crop moves the marks
                if self.preview.area_mode:
                    self._refresh_area_marks()
                if self.preview.zoom > 1.0 and self._hires_key != self._settings_key(p):
                    self.preview.hires = None       # its colours are out of date: show the preview meanwhile
                    self.on_zoom(self.preview.zoom)
                if self._hires_key != self._settings_key(p):
                    self.preview.fit_hires = None   # out of date: the new preview shows until it is redone
                if not self._request_draft.get(request_id, False):
                    self._check_sharp_fit()         # full view: the full-resolution picture of this edit
        self._request_angle.pop(request_id, None)
        self._request_draft.pop(request_id, None)
        self._request_key.pop(request_id, None)
        if self.preview_pending:
            self.preview_pending = False
            full, self._pending_full = self._pending_full, False
            self.request_preview(draft=not full)

    # ----- Export ----------------------------------------------------------
    def _add_menu(self):
        """The + beside "Rolls & frames". Dropping folders or files on the window does the same."""
        m = QMenu(self)
        roll = m.addAction("Add Roll…", self.open_roll)
        roll.setToolTip("A folder of one film: its frames share a colour baseline and the roll's details")
        frames = m.addAction("Add Frames…", self.open_frames)
        frames.setToolTip("Loose pictures, each converted on its own")
        m.setToolTipsVisible(True)
        m.exec(self.add_btn.mapToGlobal(QPoint(0, self.add_btn.height() + 4)))

    def _export_menu(self):
        """The small arrow beside Export: what else can be made from the pictures."""
        m = QMenu(self)
        m.addAction("Export…", self.open_export)
        sheet = m.addAction("Contact Sheet of This Roll…", lambda: self.open_contact_sheet())
        sheet.setEnabled(bool(self.group and self.group.kind == "roll"))
        m.exec(self.export_more.mapToGlobal(QPoint(self.export_more.width() - m.sizeHint().width(),
                                                   self.export_more.height() + 4)))

    def open_export(self, selected_only: bool = False):
        """The export window. Several frames selected (or `selected_only`, from the filmstrip's
        menu): "Selected (n)" is offered and chosen first."""
        if not self.all_photos():
            self.statusBar().showMessage("Nothing to export yet.")
            return
        missing = [g for g in self.groups if g.kind == "roll" and g.photos and not g.film_chosen]
        if missing:                               # the film decides the conversion: choose it first
            g = missing[0]
            if g is not self.group:
                self.show_group(g)
            exif_panel.flash(self.stock_box)
            names = ", ".join(m.name for m in missing)
            self.statusBar().showMessage(f"Choose the film stock first: {names}", 5000)
            return
        sel = [self.photos[self.grid.row(i)] for i in self.grid.selectedItems()]
        extra = dict(self.export_extra)
        offer = len(sel) if (selected_only or len(sel) > 1) else 0
        if offer:
            extra["scope"] = "selection"
        elif extra.get("scope") == "selection":
            extra["scope"] = "all"
        if not extra.get("dest"):
            first = self.all_photos()[0]
            extra["dest"] = str(first.path.parent / "converted")
        if self.current:
            extra["sample"] = self._tokens(self.current)
        dlg = ExportDialog(self, self.export_opts, extra, selected=offer,
                           has_roll=any(g.kind == "roll" for g in self.groups))
        if not dlg.exec():
            return
        self.export_opts, self.export_extra = dlg.options()
        import json
        from dataclasses import asdict
        set_pref("export", json.dumps({"opts": asdict(self.export_opts), "per_roll": self.export_extra["per_roll"],
                                       "reveal": self.export_extra["reveal"]}))
        if not self.export_extra["dest"]:
            self.statusBar().showMessage("Choose a folder to export to.")
            return
        scope = self.export_extra["scope"]
        photos = (sel if scope == "selection" else
                  self.photos if scope == "group" else self.all_photos())
        trash = self.export_extra.pop("trash", False)
        if trash and not self._confirm_trash(len([p for p in photos if p.ok])):
            return
        self._export(photos, trash=trash)

    def _confirm_trash(self, n: int) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setText(f"Move {n} original scan{'s' if n != 1 else ''} to the {TRASH_NAME} after export?")
        box.setInformativeText("Each scan moves only after its own file exported without an error. "
                               f"You can put them back from the {TRASH_NAME}.")
        go = box.addButton(f"Export and Move to {TRASH_NAME}", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        return box.clickedButton() is go

    def _tokens(self, p: Photo) -> dict:
        g = p.group
        roll = g and g.kind == "roll"
        return {"name": p.path.stem,
                "roll": g.name if roll else "",
                "film": (g.stock or {}).get("displayName", "") if roll else "",
                "frame": f"{g.photos.index(p) + 1:02d}" if g else "",
                "camera": g.camera if roll else ""}

    def _export(self, photos: list[Photo], trash: bool = False):
        """Export these frames. `trash`: move each original to the Trash once its own export succeeded."""
        photos = [p for p in photos if p.ok]
        if not photos:
            self.statusBar().showMessage("Nothing to export.")
            return
        out_dir = Path(self.export_extra["dest"])
        opts = self.export_opts
        if self._export_left > 0:                 # an export is still running: this one joins it
            self._export_total += len(photos)
            self._export_left += len(photos)
        else:
            self._export_total, self._export_left, self._export_errors = len(photos), len(photos), 0
            self._export_failed = []
            self._to_trash = []
            engine.new_export_batch()
        self._export_dir = str(out_dir)
        self.statusBar().showMessage(f"Exporting {self._export_total - self._export_left} of {self._export_total}…")
        for p in photos:
            st = p.settings.copy()
            p._exporting = edit_key(p.settings)
            p._trash_after = trash
            g = p.group
            is_roll = bool(g and g.kind == "roll")
            dest = out_dir / g.name if is_roll and self.export_extra.get("per_roll", True) else out_dir
            fields = self.exif_fields(p)          # film, camera, lens, date, place, GPS go into the file
            meta = (engine.metadata_from_exif(fields, g.name if is_roll else "",
                                              g.photos.index(p) + 1 if is_roll else None)
                    if is_roll or any(v for k, v in fields.items() if k != "push_pull") else None)
            tokens = self._tokens(p)
            self.export_pool.start(Job(lambda p=p, st=st, dest=dest, meta=meta, tokens=tokens:
                                str(engine.export(p.path, dest, st, opts, metadata=meta, tokens=tokens)),
                                lambda r, p=p: self.signals.exported.emit(p, r)))

    def on_exported(self, photo: Photo, msg: str):
        if msg.startswith("ERROR"):
            self._export_errors += 1
            self._export_failed.append(f"{photo.path.name}: {msg[7:]}")
        else:
            photo.exported = getattr(photo, "_exporting", "")
            out = Path(msg)
            # only when the new file is really there, and is not the scan itself
            if getattr(photo, "_trash_after", False) and out.is_file() and out.stat().st_size > 0 \
                    and out.resolve() != photo.path.resolve():
                self._to_trash.append(photo)
        photo._trash_after = False
        self._export_left -= 1
        done = self._export_total - self._export_left
        if self._export_left:
            self.statusBar().showMessage(f"Exporting {done} of {self._export_total}…")
        else:
            errs = f" ({self._export_errors} failed)" if self._export_errors else ""
            self.statusBar().showMessage(f"Exported {done - self._export_errors} image(s){errs} → {self._export_dir}")
            if self._export_failed:                   # say which files, and why, where it can be read
                box = QMessageBox(self)
                box.setIcon(QMessageBox.NoIcon)
                box.setText(f"{len(self._export_failed)} file(s) could not be exported.")
                box.setInformativeText("\n".join(self._export_failed[:12]) + ("\n…" if len(self._export_failed) > 12 else ""))
                box.addButton("OK", QMessageBox.AcceptRole)
                self._export_failed = []
                box.open()
            if self.export_extra.get("reveal") and done > self._export_errors:
                show_in_file_manager(self._export_dir)
            if self._to_trash:
                self._trash_originals(self._to_trash)
                self._to_trash = []

    def _trash_originals(self, photos: list[Photo]):
        """Move the exported frames' scans to the Trash and take those frames out of the list."""
        moved, kept = [], []
        for p in photos:
            (moved if move_to_trash(p.path) else kept).append(p)
        if moved:
            self._drop_frames(moved)
        text = f"Exported. Moved {len(moved)} original scan(s) to the {TRASH_NAME}."
        if kept:
            text += f" {len(kept)} could not be moved and are still in place."
            box = QMessageBox(self)
            box.setIcon(QMessageBox.NoIcon)
            box.setText(f"{len(kept)} original scan(s) could not be moved to the {TRASH_NAME}.")
            box.setInformativeText("They were exported and are still in place:\n"
                                   + "\n".join(p.path.name for p in kept[:12]) + ("\n…" if len(kept) > 12 else ""))
            box.addButton("OK", QMessageBox.AcceptRole)
            box.open()
        self.statusBar().showMessage(text, 8000)

    def _drop_frames(self, photos: list[Photo]):
        """Take these frames out of the list, from whichever roll they are in."""
        if self.preview.crop_mode:
            self.toggle_crop(apply=False)
        gone = set(photos)
        self._forget(photos)
        for g in list(self.groups):
            if not gone.intersection(g.photos):
                continue
            if gone.issuperset(g.photos):
                self.remove_group(g)
                continue
            for ph in [ph for ph in g.photos if ph in gone]:
                g.photos.remove(ph)
            self._refresh_group_item(g)
            if g is self.group:
                self.current = None
                self.show_group(g)
            if g.kind == "roll":
                g.analyzed = False
                self.analyze_roll(g)

    # ----- Contact sheet (per roll) ----------------------------------------
    def open_contact_sheet(self, g: Group | None = None):
        """The roll's frames on one sheet of paper. From the roll's details, its right-click menu or a key."""
        g = g or self.group
        if not g or g.kind != "roll":
            self.statusBar().showMessage("A contact sheet is made from a roll: select one on the left.", 5000)
            return
        if g in self._sheets:
            self.statusBar().showMessage(f"The contact sheet of {g.name} is already being made.", 5000)
            return
        photos = [p for p in g.photos if not p.failed]
        if not photos:
            self.statusBar().showMessage("This roll has no frames that can be printed.", 5000)
            return
        import json
        from dataclasses import asdict
        opts, reveal = engine.SheetOptions(), True
        try:                                          # the options of your last contact sheet
            saved = json.loads(pref("sheet") or "{}")
            fields_ = engine.SheetOptions.__dataclass_fields__
            opts = engine.SheetOptions(**{k: v for k, v in saved.get("opts", {}).items() if k in fields_})
            reveal = bool(saved.get("reveal", True))
        except (TypeError, ValueError):
            pass
        # next to this roll's exported pictures
        base = Path(self.export_extra.get("dest") or photos[0].path.parent / "converted")
        dest = base / engine.file_name("{name}", {"name": g.name}) if self.export_extra.get("per_roll", True) else base
        dlg = ContactSheetDialog(self, g.name, len(photos), opts, str(dest), reveal)
        if not dlg.exec():
            return
        opts, dest, reveal = dlg.options()
        set_pref("sheet", json.dumps({"opts": asdict(opts), "reveal": reveal}))
        if not dest:
            self.statusBar().showMessage("Choose a folder to save the contact sheet to.")
            return
        self._commit_history()
        frames = [(p.path, p.settings.copy()) for p in photos]
        label = engine.sheet_label(g.name, g.stock, g.push_pull, g.camera, g.date)
        stock = g.stock
        self._sheets[g] = reveal
        self.statusBar().showMessage(f"Contact sheet of {g.name}: frame 0 of {len(frames)}…")
        if self._export_left <= 0 and len(self._sheets) == 1:
            engine.new_export_batch()        # names of earlier exports are free again
        self.export_pool.start(Job(
            lambda: engine.contact_sheet(frames, Path(dest), g.name, opts, label, stock,
                                         progress=lambda i, n: self.signals.sheet_progress.emit(g, i, n)),
            lambda r: self.signals.sheet.emit(g, r)))

    def on_sheet_progress(self, g: Group, done: int, total: int):
        if self._export_left <= 0:                    # a running export keeps the status line
            self.statusBar().showMessage(f"Contact sheet of {g.name}: frame {done} of {total}…")

    def on_sheet(self, g: Group, result):
        reveal = self._sheets.pop(g, False)
        if isinstance(result, str):                   # "ERROR: ..."
            box = QMessageBox(self)
            box.setIcon(QMessageBox.NoIcon)
            box.setText(f"The contact sheet of {g.name} could not be made.")
            box.setInformativeText(result[7:])
            box.addButton("OK", QMessageBox.AcceptRole)
            box.open()
            self.statusBar().showMessage("")
            return
        n = len(result)
        self.statusBar().showMessage(f"Contact sheet of {g.name}: {n} sheet{'s' if n > 1 else ''} → {result[0].parent}")
        if reveal:
            show_in_file_manager(str(result[0] if n == 1 else result[0].parent))


# The look: a grey window with black, borderless sections; a deep purple for anything chosen.
# Change a colour here and it changes everywhere (the stylesheet below and the painted widgets).
COLORS = {
    # Dark neutral greys: panels a touch lighter than the window, buttons lighter than their panel,
    # fields recessed. Colour comes from the faders; a deep purple marks what is chosen.
    "bg": "#0f1012",          # window (under the cutting-mat grid, see Backdrop)
    "dots": "#26272b",        # the dots of the "dots" pattern
    "grid_fine": "#16171a",   # the mat's thin lines, one per cell
    "grid": "#202125",        # its stronger lines, every fourth cell
    "panel": "#161719",       # sections
    "panel_top": "#191a1d",   # sections: a faint fall of light
    "field": "#0c0d0e",       # text fields: recessed below their section
    "button": "#1f2023",      # buttons: lighter than the section they sit on (the bottom of their gradient)...
    "button_top": "#27282c",  # ...and its top: the same fall of light gives them body
    "button_hover": "#25262a",
    "button_hover_top": "#2e2f34",
    "button_down": "#151618",
    "edge_hi": "#303136",     # the lit top edge of a raised shape
    "edge_lo": "#070809",     # its outline; a sunken field has them the other way round
    "track": "#33343a",       # a slider's track...
    "tick": "#38393e",        # ...the ticks standing on it...
    "fill": "#c2c2c6",        # ...its fill from zero...
    "pointer": "#76767b",     # ...and its pointer, untouched...
    "pointer_on": "#d0d0d4",  # ...and moved
    "knob": "#0e0f11",        # slider handles: dark caps...
    "knob_edge": "#4e4f55",   # ...with a quiet edge
    "text": "#b8b8bc",        # body text: a light grey, never white
    "soft_solid": "#9a9a9f",  # icons at rest: the second text level, opaque
    "muted": "#808085",
    "faint": "#4a4b50",
    "soft": "rgba(184, 184, 188, 70%)",    # text, second level: labels, quiet buttons
    "dim": "rgba(184, 184, 188, 44%)",     # third level: hints
    "ghost": "rgba(184, 184, 188, 28%)",   # captions, untouched values
    "placeholder": "rgba(184, 184, 188, 26%)",   # what a field would hold: barely there
    "danger": "#ff5a52",      # removing something: the trash under the mouse
    "warn": "#f0a24a",        # a quiet warning (the scan itself is too dark / clipped), the batch notice
    "line": "#222327",        # hairlines inside sections
    "accent": "#7350c8",      # deep purple: selection, focus, the main action, the progress line
    "accent_hover": "#9a80dc",
    "accent2": "#9a80dc",     # the lighter accent: an adjusted value, hover over something that is on,
                              # the field being typed in
    "select": "#d8d8dc",      # a chosen plate where one is still used...
    "on_select": "#000000",   # ...with its text reversed to black
    "on_select_muted": "#55575d",
    "raise": "#1d1e21",       # a quiet lift
    "on_accent": "#ffffff",
}
ACCENT = COLORS["accent"]
ACCENT2 = COLORS["accent2"]

# Settings › Accent: the one colour that marks what is chosen. (name, accent, lighter, text on it),
# all of about the same strength on the dark window.
ACCENTS = {
    "purple": ("Purple", "#7350c8", "#9a80dc", "#ffffff"),
    "blue": ("Blue", "#4a70cf", "#7b98e0", "#ffffff"),
    "gold": ("Gold", "#c39a2e", "#dcbb62", "#17120a"),
    "teal": ("Teal", "#2a9688", "#5bbcae", "#ffffff"),
}


def apply_accent(key: str, restyle: bool = True):
    """Make `key` the app's accent: the colour table, what widgets paint with, and the stylesheet."""
    global ACCENT, ACCENT2
    _name, accent, light, on = ACCENTS.get(key, ACCENTS["purple"])
    COLORS.update(accent=accent, accent_hover=light, accent2=light, on_accent=on)
    ACCENT, ACCENT2 = accent, light
    exif_panel.ACCENT = accent
    qapp = QApplication.instance()
    if restyle and qapp:
        qapp.setStyleSheet(make_style())
        for w in qapp.allWidgets():
            w.update()

# Type scale (px): one size per role; names and what is chosen are semibold, the rest regular.
#   title 20 (light)  ·  card title 15  ·  body 13  ·  small 12  ·  caption 10 (spaced capitals)
STYLE_TEMPLATE = """
QWidget { background: transparent; color: @text; font-size: 13px; }
QMainWindow, QDialog { background: @bg; }
QToolTip { background: @panel; color: @text; border: 1px solid @line; border-radius: 6px; padding: 5px 8px; font-size: 12px; }
QFrame#panel { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @panel_top, stop:1 @panel); border: 1px solid @edge_lo; border-top-color: @line; border-radius: 6px; }

QLabel#header { color: @ghost; font-size: 10px; font-weight: 600; letter-spacing: 1.6px; }
QLabel#hint { color: @dim; font-size: 12px; padding-top: 4px; }
QLabel#welcome { font-size: 22px; font-weight: 300; color: @text; padding-bottom: 22px; letter-spacing: 0.3px; }
QLabel#pageTitle { font-size: 17px; font-weight: 600; color: @text; }
QLabel#title { color: @muted; font-size: 12px; padding: 3px 10px; }
QLabel#title[over="true"] { background: rgba(0, 0, 0, 170); color: @text; }
QLabel#faderTitle { color: @text; font-size: 12px; font-weight: 600; }
QLabel#rowLabel { color: @soft; font-size: 12px; font-weight: 500; }
QLabel#key { font-size: 12px; color: @accent; }
QLabel#faderValue { font-size: 11px; color: @ghost; }
QLabel#faderValue[edited="true"] { color: @text; font-weight: 600; }
QFrame#card { background: @panel; border: none; border-radius: 8px; }
QLabel#cardTitle { font-size: 16px; font-weight: 500; color: @text; border-bottom: 2px solid @accent; padding-bottom: 5px; }
QLabel#cardText { font-size: 12px; color: @dim; }
QLabel#exifTitle { font-size: 15px; font-weight: 600; color: @text; }
QToolButton#presetMenu { background: @bg; color: @muted; font-size: 13px; padding: 0; }
QToolButton#presetMenu:hover { color: @accent; }
QToolButton#presetMenu::menu-indicator { image: none; }
QMenu { background: @panel; color: @text; border: 1px solid @line; border-radius: 8px; padding: 5px; }
QMenu::item { padding: 6px 14px; border-radius: 5px; }
QMenu::item:selected { background: @accent; color: @on_accent; }
QMenu::item:disabled { color: @faint; }
QMenu::separator { height: 1px; background: @line; margin: 4px 6px; }

QPushButton { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_top, stop:1 @button); color: @text; border: 1px solid @edge_lo; border-top-color: @edge_hi; border-radius: 4px; padding: 6px 12px; font-size: 12px; font-weight: 500; }
QFrame#panel QPushButton { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_top, stop:1 @button); }
QPushButton:hover, QFrame#panel QPushButton:hover { color: @text; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_hover_top, stop:1 @button_hover); }
QPushButton:pressed, QFrame#panel QPushButton:pressed { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_down, stop:1 @button); color: @text; border-top-color: @edge_lo; }
QPushButton:disabled, QFrame#panel QPushButton:disabled { color: @faint; }
QPushButton#square { min-width: 36px; }
QFrame#panel QPushButton[on="true"] { color: @accent; }
QPushButton#tile, QFrame#panel QPushButton#tile { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_top, stop:1 @button); padding: 0; }
QPushButton#tile:hover, QFrame#panel QPushButton#tile:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_hover_top, stop:1 @button_hover); }
QPushButton#tile[on="true"], QFrame#panel QPushButton#tile[on="true"] { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_down, stop:1 @button); border-top-color: @edge_lo; }
QPushButton#tile:pressed, QFrame#panel QPushButton#tile:pressed { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_down, stop:1 @button); border-top-color: @edge_lo; }
QPushButton#toggle, QFrame#panel QPushButton#toggle { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_top, stop:1 @button); color: @soft; padding: 5px 12px; min-width: 34px; }
QPushButton#toggle:hover, QFrame#panel QPushButton#toggle:hover { color: @text; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_hover_top, stop:1 @button_hover); }
QPushButton#toggle:pressed, QFrame#panel QPushButton#toggle:pressed { color: @text; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_down, stop:1 @button); border-top-color: @edge_lo; }
QPushButton#toggle:checked, QFrame#panel QPushButton#toggle:checked,
QPushButton#toggle[on="true"], QFrame#panel QPushButton#toggle[on="true"] { color: @accent_hover; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_down, stop:1 @button); border-top-color: @edge_lo; font-weight: 600; }
QPushButton#toggle[go="true"], QFrame#panel QPushButton#toggle[go="true"] { color: @accent2; background: @raise; font-weight: 600; }
QPushButton#toggle[go="true"]:hover, QFrame#panel QPushButton#toggle[go="true"]:hover { color: @text; }
QPushButton#toggle:checked:hover, QFrame#panel QPushButton#toggle:checked:hover,
QPushButton#toggle[on="true"]:hover, QFrame#panel QPushButton#toggle[on="true"]:hover { color: @accent2; }
QPushButton#bar { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_top, stop:1 @button); color: @soft; padding: 6px 10px; font-size: 12px; letter-spacing: 0.2px; }
QPushButton#bar:hover { color: @text; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_hover_top, stop:1 @button_hover); }
QPushButton#bar:pressed { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_down, stop:1 @button); color: @text; border-top-color: @edge_lo; }
QPushButton#bar[on="true"], QPushButton#bar:checked { color: @accent_hover; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_down, stop:1 @button); border-top-color: @edge_lo; font-weight: 600; }
QPushButton#bar[on="true"]:hover, QPushButton#bar:checked:hover { color: @accent2; }
QPushButton#barGo { background: @raise; color: @accent2; padding: 8px 22px; font-size: 12px; font-weight: 600;
                    letter-spacing: 0.2px; }
QPushButton#barGo:hover { color: @text; }
QPushButton#barGo:pressed { background: @button_down; color: @text; }
QFrame#tabPlate { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @panel_top, stop:1 @panel); border: 1px solid @edge_lo; border-top-color: @line; border-radius: 6px; }
QPushButton#primary, QFrame#panel QPushButton#primary { background: @accent; color: @on_accent; font-size: 13px;
                                                         font-weight: 600; padding: 9px 16px; }
QPushButton#primary:hover, QFrame#panel QPushButton#primary:hover { background: @accent_hover; color: @on_accent; }
QPushButton#primary:pressed, QFrame#panel QPushButton#primary:pressed { background: @accent; color: @on_accent; }
QPushButton#primary:disabled, QFrame#panel QPushButton#primary:disabled { background: @bg; color: @faint; }
QPushButton#export { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_top, stop:1 @button); color: @text; border: 1px solid @accent; font-size: 13px; font-weight: 600;
                     padding: 0; border-radius: 5px; letter-spacing: 0.2px; }
QPushButton#export:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_hover_top, stop:1 @button_hover); color: @text; border-color: @accent_hover; }
QPushButton#export:disabled { background: @panel; color: @faint; }
QPushButton#tab { background: transparent; border: 1px solid transparent; border-radius: 4px;
                  color: @muted; padding: 6px 12px; font-size: 12px; letter-spacing: 0.3px; }
QPushButton#tab:checked { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_top, stop:1 @button); border: 1px solid @edge_lo; border-top-color: @edge_hi; color: @text; font-weight: 600; }
QPushButton#tab:hover { color: @text; }
QPushButton#tab:checked:hover { color: @accent2; }
QToolButton { border: none; border-radius: 0; font-size: 14px; }
QPushButton#fold, QFrame#panel QPushButton#fold { background: transparent; border: none; color: @soft; font-size: 12px;
                                                  font-weight: 600; letter-spacing: 0; text-align: left; padding: 4px 0; }
QPushButton#fold:hover, QFrame#panel QPushButton#fold:hover,
QPushButton#fold:checked, QFrame#panel QPushButton#fold:checked { background: transparent; color: @text; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: @field; border: 1px solid @edge_lo; border-bottom-color: @line;
                                                  border-radius: 4px; padding: 4px 8px;
                                                  font-size: 12px; selection-background-color: @accent;
                                                  selection-color: @on_accent; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateEdit:focus { background: @field;
                                                                                         border-color: @accent; }
QLineEdit, QComboBox, QDateEdit { placeholder-text-color: @placeholder; }
QDateEdit[empty="true"] { color: @placeholder; }
QListView#completer { background: @panel; color: @text; border: 1px solid @edge_lo; border-top-color: @line;
                      border-radius: 6px; padding: 4px; outline: none; font-size: 12px; }
QListView#completer::item { padding: 5px 8px; border-radius: 4px; }
QListView#completer::item:selected, QListView#completer::item:hover { background: @accent; color: @on_accent; }
QDoubleSpinBox#straightenAngle { padding: 4px 22px 4px 4px; }
QDoubleSpinBox#straightenAngle QLineEdit { background: transparent; border: none; padding: 0; }
QDoubleSpinBox#straightenAngle::up-button, QDoubleSpinBox#straightenAngle::down-button {
    subcontrol-origin: padding; width: 16px; border: none; background: transparent; margin-right: 2px; }
QDoubleSpinBox#straightenAngle::up-button { subcontrol-position: top right; border-top-right-radius: 4px; }
QDoubleSpinBox#straightenAngle::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 4px; }
QDoubleSpinBox#straightenAngle::up-button:hover, QDoubleSpinBox#straightenAngle::down-button:hover { background: @raise; }
QDoubleSpinBox#straightenAngle::up-button:pressed, QDoubleSpinBox#straightenAngle::down-button:pressed { background: @button_down; }
QDoubleSpinBox#straightenAngle::up-arrow, QDoubleSpinBox#straightenAngle::down-arrow { width: 7px; height: 5px; }
QDoubleSpinBox#straightenAngle::up-arrow { image: url("@icons/angle-up.svg"); }
QDoubleSpinBox#straightenAngle::down-arrow { image: url("@icons/angle-down.svg"); }
QDoubleSpinBox#straightenAngle::up-arrow:disabled, QDoubleSpinBox#straightenAngle::down-arrow:disabled { width: 0; height: 0; }
QDateEdit { background: @field; border: 1px solid @edge_lo; border-bottom-color: @line; border-radius: 4px;
            padding: 4px 8px; font-size: 12px; }
QDateEdit[inherited="true"] { color: @muted; }
QDateEdit::drop-down { border: none; width: 24px; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: none; width: 0; }
QDateEdit::down-arrow { image: none; width: 0; }
QLineEdit[saved="true"], QDateEdit[saved="true"], QComboBox[saved="true"] { background: @raise; color: @accent; }
QLabel#savedNote { color: @accent; font-size: 12px; }
QListWidget#placeList { background: @panel; border: 1px solid @line; border-radius: 8px; padding: 4px; }
QListWidget#placeList::item { border: none; border-radius: 5px; }
QListWidget#placeList::item:selected { background: @raise; }
QCalendarWidget QWidget { background: @panel; color: @text; }
QCalendarWidget QToolButton { background: transparent; color: @text; font-size: 13px; padding: 4px 8px; }
QCalendarWidget QToolButton:hover { color: @accent; }
QCalendarWidget QAbstractItemView { background: @panel; color: @text; selection-background-color: @accent;
                                    selection-color: @on_accent; outline: none; }
QCalendarWidget QAbstractItemView:disabled { color: @faint; }
QComboBox QAbstractItemView { background: @panel; border: 1px solid @line; selection-background-color: @accent;
                              selection-color: @on_accent; }
QCheckBox { spacing: 10px; font-size: 12px; color: @soft; }
QCheckBox:checked { color: @text; }
QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px; background: @button; border: none; }
QCheckBox::indicator:hover { background: @button_down; }
QCheckBox::indicator:checked { image: url("@dots/check-dot.svg"); }
QCheckBox::indicator:checked:hover { image: url("@dots/check-dot-hover.svg"); }
QPushButton#bar[joined="left"], QPushButton#export[joined="left"] { border-top-right-radius: 0; border-bottom-right-radius: 0; }
QPushButton#export[joined="right"] { border-top-left-radius: 0; border-bottom-left-radius: 0; padding: 0; }
QPushButton#bar[joined="right"] { border-top-left-radius: 0; border-bottom-left-radius: 0; padding: 0; }
QFrame#toolPlate { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @button_top, stop:1 @button);
                   border: 1px solid @edge_lo; border-top-color: @edge_hi; border-radius: 6px; }
QFrame#toolDiv { background: @edge_lo; border: none; }
QPushButton#tool { background: transparent; border: 1px solid transparent; color: @soft; padding: 4px 9px;
                   border-radius: 4px; font-size: 12px; }
QPushButton#tool:hover { background: rgba(255, 255, 255, 6%); color: @text; }
QPushButton#tool:pressed { background: rgba(0, 0, 0, 35%); color: @text; }
QPushButton#tool[go="true"] { color: @accent_hover; font-weight: 600; }
QPushButton#tool[on="true"], QPushButton#tool:checked { background: rgba(0, 0, 0, 40%); color: @accent_hover;
                                                       border-color: @edge_lo; }
QFrame#syncPop { background: @panel; border: 1px solid @line; border-radius: 6px; }
QFrame#syncSplit { background: transparent; border: 1px solid @edge_lo; border-radius: 4px; margin-left: 2px; }
QFrame#syncSplit:hover { border-color: @line; }
QPushButton#syncMain { background: transparent; border: none; color: @soft; padding: 3px 8px;
                       border-top-left-radius: 3px; border-bottom-left-radius: 3px; font-size: 12px; }
QPushButton#syncMore { background: transparent; border: none; padding: 0;
                       border-top-right-radius: 3px; border-bottom-right-radius: 3px; }
QPushButton#syncMain:hover, QPushButton#syncMore:hover { background: rgba(255, 255, 255, 6%); color: @text; }
QPushButton#syncMain:pressed, QPushButton#syncMore:pressed { background: rgba(0, 0, 0, 35%); }

QFrame#rollCard { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @panel_top, stop:1 @panel); border: 1px solid @edge_lo; border-top-color: @line; border-radius: 6px; }
QFrame#rollHead { background: transparent; }
QLabel#rollTitle { font-size: 13px; font-weight: 600; color: @text; }
QLabel#rollSub { font-size: 11px; color: @dim; }
QPushButton#rollEdit { background: transparent; padding: 0; border: none; }
QLineEdit#rollRename { font-size: 13px; font-weight: 600; padding: 1px 4px; }
QListWidget#groups { background: @panel; border: none; padding: 2px; font-size: 12px; }
QListWidget#groups::item { padding: 7px 8px; }
QListWidget#groups::item:selected { background: @button_down; color: @text; }
QListWidget#grid { background: @panel; border: 1px solid @edge_lo; border-top-color: @line; border-radius: 6px; padding: 4px 8px;
                   font-size: 10px; color: @muted; }
QListWidget#grid::item, QListWidget#grid::item:selected { border: none; background: transparent; }

QSlider::groove:horizontal { height: 3px; background: @button; border-radius: 1px; }
QSlider::sub-page:horizontal { background: @button; }
QSlider::handle:horizontal { background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 @muted, stop:0.34 @muted, stop:0.42 @knob, stop:1 @knob); border: 1px solid @knob_edge;
                              width: 11px; margin: -5px 0; border-radius: 6px; }
QSlider::handle:horizontal:hover, QSlider::handle:horizontal:pressed { background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 @accent, stop:0.34 @accent, stop:0.42 @knob, stop:1 @knob); }
QScrollBar:vertical { background: transparent; width: 7px; margin: 0; }
QScrollBar::handle:vertical { background: @button; min-height: 30px; border-radius: 3px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: @faint; }
QScrollBar:horizontal { background: transparent; height: 7px; margin: 0; }
QScrollBar::handle:horizontal { background: @button; min-width: 30px; border-radius: 3px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QStatusBar { background: transparent; color: @dim; font-size: 11px; padding-left: 14px; }
QStatusBar::item { border: none; }
QScrollArea { border: none; }
"""


def _accent_dots() -> str:
    """The checkbox's dot in the current accent (an image the stylesheet points at)."""
    import tempfile
    folder = Path(tempfile.gettempdir()) / "dumbconvert-accent" / COLORS["accent"].lstrip("#")
    folder.mkdir(parents=True, exist_ok=True)
    for name, colour in (("check-dot.svg", COLORS["accent"]), ("check-dot-hover.svg", COLORS["accent2"])):
        f = folder / name
        if not f.exists():
            f.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
                         f'<circle cx="6" cy="6" r="2.5" fill="{colour}"/></svg>')
    return folder.as_posix()


def make_style() -> str:
    """One typeface everywhere: the system's (San Francisco on a Mac), numbers included."""
    style = STYLE_TEMPLATE.replace("@icons", (Path(__file__).parent / "icons").as_posix())
    style = style.replace("@dots", _accent_dots())
    for name in sorted(COLORS, key=len, reverse=True):     # accent_hover before accent
        style = style.replace("@" + name, COLORS[name])
    return style


STYLE = make_style()


def self_test(path: str) -> int:
    """`DUMBCONVERT_SELFTEST=1 DumbConvert <scan>`: every feature that needs a library, run without a
    window on one scan, each file written read back. Proves a packaged (and slimmed) app has all it
    needs. Prints PASS / FAIL per check; returns 1 when anything failed."""
    import tempfile
    import time
    import traceback as tb

    import numpy as np

    src, failures = Path(path), []

    def check(name, fn):
        t = time.time()
        try:
            note = fn() or ""
            print(f"SELFTEST PASS {name} ({time.time() - t:.1f}s) {note}", flush=True)
        except Exception as exc:
            failures.append(name)
            print(f"SELFTEST FAIL {name}: {exc}", flush=True)
            tb.print_exc()

    def readable(out: Path, fmt: str):
        """Open the written file with the library a viewer would use; returns (width, height)."""
        if fmt == "jxl":
            import imagecodecs
            a = imagecodecs.jpegxl_decode(out.read_bytes())
            return a.shape[1], a.shape[0]
        if fmt in ("dng", "dng16"):
            import rawpy
            with rawpy.imread(str(out)) as raw:
                return raw.sizes.width, raw.sizes.height
        if fmt == "tif":
            import tifffile
            a = tifffile.imread(out)
            return a.shape[1], a.shape[0]
        if fmt == "heic":
            import pillow_heif
            pillow_heif.register_heif_opener()
        from PIL import Image
        with Image.open(out) as im:
            im.load()
            return im.size

    st = engine.Settings()
    found = {}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        def preview():
            a = engine.preview(src, st, 800)
            assert a.ndim == 3 and a.mean() > 1, "empty picture"
            return f"{a.shape[1]}x{a.shape[0]}, {len(engine.film_stocks())} film stocks"
        check("preview (CPU)", preview)

        def auto_crop():
            r = engine.find_auto_crop(src, engine.Settings(auto_crop=True))
            found["auto"] = r
            a = engine.preview(src, engine.Settings(auto_crop=True, auto_rect=r[0] if r else None,
                                                    auto_key=r[1] if r else ""), 800)
            return f"{'found ' + str(tuple(round(v, 3) for v in r[0])) if r else 'no border (film fills the scan)'}, {a.shape[1]}x{a.shape[0]}"
        check("auto crop", auto_crop)

        def gpu():
            if not engine.gpu_active() or engine._gpu_processor() is None:
                return "no graphics card: skipped"
            c, g = engine.preview(src, st, 1600).astype(float), engine.preview(src, st, 1600, gpu=True).astype(float)
            assert c.shape == g.shape, f"shapes {c.shape} {g.shape}"
            diff = float(np.abs(c - g).mean())
            assert diff < 1.0, f"GPU differs from CPU by {diff:.2f}"
            return f"matches the CPU (mean difference {diff:.2f})"
        check("preview (graphics card)", gpu)

        check("original scan", lambda: "%dx%d" % engine.original_preview(src, st, 800).shape[1::-1])

        def heal_and_clean():
            plain = engine.preview(src, st, 800).astype(float)
            spots = engine.preview(src, engine.Settings(heal_spots=((0.5, 0.5, 60.0), (0.3, 0.6, 60.0))), 800)
            area = engine.preview(src, engine.Settings(heal_areas=(((0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)),)), 800)
            for name, a in (("heal", spots), ("clean area", area)):
                assert a.shape == plain.shape, f"{name}: picture changed shape"
            return "rendered (any dust found there is repaired)"
        check("heal and clean area", heal_and_clean)

        check("thumbnail + prefetch", lambda: (engine.prefetch_preview(src, st, 1600),
                                                "%dx%d" % engine.thumbnail(src, st, 240).shape[1::-1])[1])

        check("scan exposure (EXIF)", lambda: " ".join(v for v in engine.scan_exposure(src) if v) or "none recorded")

        meta = engine.metadata_from_exif({"capture_date": "1997-09-30", "camera_model": "Test FM2",
                                          "film_stock": "Kodak Portra 400"})
        for fmt in engine.EXPORT_FORMATS:
            for with_meta in (False, True):
                def export(fmt=fmt, with_meta=with_meta):
                    out = engine.export(src, d / f"{fmt}-{with_meta}", st,
                                        engine.ExportOptions(fmt=fmt, size="long_edge", long_edge=1000),
                                        meta if with_meta else None)
                    w, h = readable(out, fmt)
                    assert w > 0 and h > 0
                    return f"{w}x{h}, {out.stat().st_size // 1024} KB"
                check(f"export {fmt}{' + metadata' if with_meta else ''}", export)

        def border():
            plain = engine.export(src, d / "b0", st, engine.ExportOptions(fmt="png", overwrite=True))
            framed = engine.export(src, d / "b40", st, engine.ExportOptions(fmt="png", border_px=40, overwrite=True))
            (w0, h0), (w1, h1) = readable(plain, "png"), readable(framed, "png")
            assert (w1 - w0, h1 - h0) == (80, 80), f"border adds {w1 - w0}x{h1 - h0} px, not 80x80"
            return "40 px on every side"
        check("export with white border", border)

        def icc():
            for cs in engine.COLOR_SPACES:
                assert engine._icc(cs), f"no ICC profile for {cs}"
            return ", ".join(engine.COLOR_SPACES)
        check("colour profiles", icc)

        def tiff_codecs():
            import tifffile
            scan = engine.preview(src, st, 600)
            done = []
            for comp in ("lzw", "zlib", "zstd", "jpeg"):
                p = d / f"scan-{comp}.tif"
                tifffile.imwrite(p, scan, compression=comp)
                back = tifffile.imread(p)
                assert back.shape == scan.shape, comp
                done.append(comp)
            return "reads " + ", ".join(done)
        check("compressed TIFF scans", tiff_codecs)

        def roll():
            floors, ceils, slope, outliers, axis = engine.analyze_roll([(src, st), (src, st)])
            assert len(floors) == 3 and len(slope) == 3
            return f"baseline found, {sum(outliers)} outlier(s)"
        check("roll analysis", roll)

        def sheet():
            pages = engine.contact_sheet([(src, st)] * 3, d / "sheet", "Selftest roll",
                                         label="Selftest roll · Test FM2")
            assert pages, "no page written"
            w, h = readable(pages[0], "jpg")
            return f"{len(pages)} page(s), {w}x{h}"
        check("contact sheet", sheet)

    print(f"SELFTEST {'OK' if not failures else 'FAILED: ' + ', '.join(failures)}", flush=True)
    return 1 if failures else 0


def _log_without_console():
    """Started without a console (Windows runs the app with pythonw), errors go to dumbconvert.log
    next to the recovery file, so it can be sent with a bug report. The previous run's log is kept
    as dumbconvert.old.log."""
    if sys.stderr is not None and sys.stdout is not None:
        return
    try:
        log = MainWindow._recovery_path().with_name("dumbconvert.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists():
            log.replace(log.with_name("dumbconvert.old.log"))
        f = open(log, "w", encoding="utf-8", buffering=1)
    except OSError:
        return
    from . import __version__
    print(f"{APP_NAME} {__version__} on {sys.platform}, Python {sys.version.split()[0]}", file=f)
    sys.stdout = sys.stdout or f
    sys.stderr = sys.stderr or f


def main():
    import os
    if os.environ.get("DUMBCONVERT_SELFTEST") and len(sys.argv) > 1:
        sys.exit(self_test(sys.argv[1]))
    # Memory: by default macOS keeps freed large blocks cached in the app, and a converted scan
    # frees gigabytes of them. MallocSpaceEfficient=1 hands them back at once. The setting is
    # read when a process starts, so without it we start again with it. The packaged app gets it
    # from its Info.plist (packaging/build.sh), run.command sets it too.
    if sys.platform == "darwin" and not os.environ.get("MallocSpaceEfficient") and not getattr(sys, "frozen", False):
        os.environ["MallocSpaceEfficient"] = "1"
        try:
            os.execv(sys.executable, [sys.executable, "-m", "simpleneg", *sys.argv[1:]])
        except OSError:
            pass                          # could not restart: carry on as before
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    _log_without_console()
    # The system UI font: San Francisco on a Mac. It is used from macOS, never shipped with the app.
    from PySide6.QtGui import QFontDatabase
    ui = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    app.setFont(ui)
    apply_accent(pref("accent"), restyle=False)
    app.setStyleSheet(make_style())
    win = MainWindow()
    win.show()
    win.offer_restore()                              # a crash's work, else the saved session; else nothing to ask
    win._recovery_timer.start()
    if len(sys.argv) > 1:          # allow: python -m simpleneg file1.tif folder/ ...
        win.add_rolls([Path(a) for a in sys.argv[1:]])
    sys.exit(app.exec())
