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
                           QPainter, QPainterPath, QPen, QPolygonF,
                           QPixmap, QShortcut, QTransform)
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QCompleter, QDoubleSpinBox, QFormLayout, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLayout, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget,
    QStyledItemDelegate, QToolButton, QVBoxLayout, QWidget,
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
    camera: str = ""          # camera the film was shot with, rolls only
    lens: str = ""
    date: str = ""            # "2024", "2024-06" or "2024-06-21"
    location: str = ""        # GPS coordinates "lat, lon" for the whole roll
    location_label: str = ""  # the place picked in the search, shown instead of the numbers
    push_pull: int = 0        # stops: -3 .. +3
    analyzed: bool = False
    analyzing: bool = False
    outliers: int = 0
    card: "RollCard | None" = None

    @property
    def stock(self) -> dict | None:
        return engine.stock_by_id(self.stock_id) if self.stock_id else None

    def label(self) -> tuple[str, str]:
        """(title, subtitle) for its card in the left column."""
        n = len(self.photos)
        if self.kind == "frames":
            return "Single frames", f"{n} frame{'s' if n != 1 else ''}"
        film = self.stock["displayName"] if self.stock else "Film not set"
        state = "  ·  analysing…" if self.analyzing else ""
        return self.name, f"{film}  ·  {n} frame{"s" if n != 1 else ""}{state}"


# Settings that come from measuring or from the scanning setup, not from editing a frame.
NOT_EDITS = ("roll_floors", "roll_ceils", "roll_axis", "flat_field", "film", "grayscale",  # from the setup / film stock
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

    QUIET = ("toggle", "bar", "tab", "fold")      # these rest in the muted colour
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
PREF_DEFAULTS = {"auto_crop_new": False, "scan_warnings": True, "flat_field": "", "export": "", "sheet": "",
                 "histogram": "rgb"}


def pref(key: str):
    v = QSettings("SimpleNeg", "SimpleNeg").value("pref/" + key, PREF_DEFAULTS[key])
    if isinstance(PREF_DEFAULTS[key], bool):       # stored as text: "true" / "false"
        return v if isinstance(v, bool) else str(v).lower() == "true"
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


CHANNELS = ((236, 64, 64), (64, 210, 96), (72, 120, 255))   # red, green, blue: the histogram and its switch


class ChannelButton(FadeButton):
    """The histogram's mode switch: three dots, red, green and blue, for the colour channels.
    Clicked, they glide together into one grey dot (brightness only), and apart again."""

    mode_changed = Signal(str)
    GREY = (196, 196, 200)
    TIPS = {"rgb": "Showing red, green and blue: click for brightness only",
            "mono": "Showing brightness: click for red, green and blue"}

    def __init__(self, mode: str = "rgb"):
        super().__init__("")
        self.setObjectName("toggle")                      # the quiet plate the other switches sit on
        self.setFixedHeight(28)                          # as wide as the other switches (their style sets it)
        self.setCursor(Qt.PointingHandCursor)
        self._hover = 0.0
        self._merge = QVariantAnimation(self)             # 0: three dots, 1: one grey dot
        self._merge.setDuration(340)
        self._merge.setEasingCurve(QEasingCurve.InOutCubic)
        self._merge.valueChanged.connect(lambda _v: self.update())
        self.mode = ""
        self.set_mode(mode, animate=False)
        self.clicked.connect(lambda _checked=False: self.set_mode("rgb" if self.mode == "mono" else "mono"))

    @property
    def merged(self) -> float:
        v = self._merge.currentValue()
        return float(v) if v is not None else (1.0 if self.mode == "mono" else 0.0)

    def set_mode(self, mode: str, animate: bool = True):
        mode = "mono" if mode == "mono" else "rgb"
        if mode == self.mode:
            return
        start, end = self.merged, (1.0 if mode == "mono" else 0.0)
        self.mode = mode
        self.setToolTip(self.TIPS[mode])
        self._merge.stop()
        self._merge.setStartValue(start if animate else end)
        self._merge.setEndValue(end)
        if animate:
            self._merge.start()
        else:
            self._merge.setCurrentTime(self._merge.duration())
        self.update()
        self.mode_changed.emit(mode)

    def _tint(self, t: float):
        self._hover = t
        self.update()

    def paintEvent(self, e):
        import math
        super().paintEvent(e)                             # the plate, from the stylesheet
        t = self.merged
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.translate(self.width() / 2, self.height() / 2 + 0.5)
        spread = 4.4 * (1 - t) + 0.4 * self._hover * (1 - t)   # hovering parts them a touch
        r = 3.1 + 1.9 * t                                 # the grey dot is a little bigger than one channel
        light = 0.8 + 0.2 * self._hover                   # a little dimmer at rest, full colour under the mouse
        grey = self.GREY
        if t > 0.999:                                     # one dot: three stacked would thicken its soft edge
            p.setBrush(QColor(*(round(g * light) for g in grey)))
            p.drawEllipse(QPointF(0, 0), r, r)
        else:
            for rgb, angle in zip(CHANNELS, (-90, 150, 30)):     # red on top, green left, blue right
                a = math.radians(angle)
                p.setBrush(QColor(*(round((c + (g - c) * t) * light) for c, g in zip(rgb, grey))))
                p.drawEllipse(QPointF(spread * math.cos(a), spread * math.sin(a)), r, r)
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
        self.setMinimumHeight(34)
        self.setToolTip(tip)
        if slot:
            self.clicked.connect(lambda _checked=False: slot())

    def paintEvent(self, e):
        super().paintEvent(e)                     # the tile's background, from the stylesheet
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = (COLORS["faint"] if not self.isEnabled() else COLORS["on_select"] if self.isDown() else
                 mix(COLORS["text"], ACCENT, self._t))
        p.setPen(QPen(QColor(color), 1.3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.translate(self.width() / 2, self.height() / 2)
        p.scale(1.15, 1.15)
        p.translate(-10, -10)
        self._draw(p, self._arg)
        p.end()


class LineSlider(QWidget):
    """One control as a thin horizontal line: its name, the line with a small marker, the value.

    A two-sided control (soft ↔ hard, cool ↔ warm) has its zero in the middle and can tint each
    side; an amount (sharpen, glow) starts at the left. Drag or click the line; double-click resets.
    """

    changed = Signal(float)
    LABEL_W, VALUE_W = 92, 30

    def __init__(self, title: str, lo: float, hi: float, default: float = 0.0, step: float = 0.05,
                 unipolar: bool = False, minus: str = "", plus: str = ""):
        super().__init__()
        self.title, self.lo, self.hi, self.default, self.step = title, lo, hi, default, step
        self.unipolar = unipolar
        self.origin = lo if unipolar else 0.0          # where the filled part starts
        self.minus_color, self.plus_color = minus or COLORS["text"], plus or COLORS["text"]
        self._value = default
        self._drag = False
        self.setFixedHeight(26)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

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
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cy = self.height() / 2
        f = p.font()
        f.setPixelSize(12)
        p.setFont(f)
        moved = abs(self._value - self.default) > 1e-6
        p.setPen(QColor(COLORS["text"] if moved or self._drag else COLORS["muted"]))
        p.drawText(QRectF(0, 0, self.LABEL_W, self.height()), Qt.AlignVCenter | Qt.AlignLeft, self.title)
        x0, x1 = self._track()
        p.setPen(QPen(QColor(COLORS["button"]), 1))
        p.drawLine(QPointF(x0, cy), QPointF(x1, cy))
        xo, xv = self._x(self.origin), self._x(self._value)
        if not self.unipolar:                          # the zero mark
            p.drawLine(QPointF(xo, cy - 3), QPointF(xo, cy + 3))
        color = QColor(self.plus_color if self._value >= self.origin else self.minus_color)
        if abs(xv - xo) > 0.5:
            p.setPen(QPen(color, 1.4))
            p.drawLine(QPointF(xo, cy), QPointF(xv, cy))
        p.setPen(Qt.NoPen)                             # the marker: a slim upright bar
        p.setBrush(QColor(ACCENT if self._drag else ACCENT2 if moved else COLORS["text"]))
        p.drawRect(QRectF(xv - 1, cy - 6, 2, 12))
        mono = QFont(MONO)
        mono.setPixelSize(11)
        p.setFont(mono)
        p.setPen(QColor(ACCENT2 if moved else COLORS["faint"]))
        if self.unipolar:
            text = f"{round(100 * (self._value - self.lo) / (self.hi - self.lo))}"
        else:
            n = round((self._value - self.origin) / self.step)
            text = f"{n:+d}" if n else "0"
        p.drawText(QRectF(self.width() - self.VALUE_W, 0, self.VALUE_W, self.height()), Qt.AlignVCenter | Qt.AlignRight, text)
        p.end()

    def _value_at(self, x: float) -> float:
        x0, x1 = self._track()
        return self.lo + (min(max(x, x0), x1) - x0) / (x1 - x0) * (self.hi - self.lo)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
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


HIST_SAMPLES = 250_000   # pixels counted for the histogram: plenty for its shape, cheap per render


def histogram(rgb: np.ndarray) -> np.ndarray:
    """Counts of each value 0..255 in a uint8 RGB array: rows R, G, B and luminance (Rec. 709).

    A big picture is sampled on an even grid, so a draft render while dragging costs a few ms.
    """
    h, w = rgb.shape[:2]
    step = max(1, int((h * w / HIST_SAMPLES) ** 0.5))
    px = rgb[::step, ::step, :3].reshape(-1, 3)
    lum = (px.astype(np.uint32) @ np.array([54, 183, 19], np.uint32)) >> 8   # weights sum to 256
    return np.stack([np.bincount(c, minlength=256)[:256] for c in (px[:, 0], px[:, 1], px[:, 2], lum)])


class Histogram(QWidget):
    """The shown picture's histogram: R, G and B laid over each other (where they overlap they add
    up towards white), or a single grey curve for brightness. Shadows left, highlights right."""

    MODES = ("rgb", "mono")

    def __init__(self, mode: str = "rgb"):
        super().__init__()
        self.mode = mode if mode in self.MODES else "rgb"
        self.counts: np.ndarray | None = None
        self.setFixedHeight(84)

    def set_image(self, rgb: np.ndarray | None):
        self.counts = None if rgb is None else histogram(rgb)
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        self.update()

    SOFTEN = 1.6          # the drawn curve is blurred by this many values (of 256): calm, still exact enough
    _KERNEL = np.exp(-0.5 * (np.arange(-5, 6) / SOFTEN) ** 2)
    _KERNEL /= _KERNEL.sum()

    @staticmethod
    def _smooth(row: np.ndarray) -> np.ndarray:
        out = np.convolve(row, (0.25, 0.5, 0.25), mode="same")      # tone curves leave combs of empty values
        out[0], out[-1] = row[0], row[-1]                           # but clipping at either end stays as it is
        return out

    @classmethod
    def soften(cls, row: np.ndarray) -> np.ndarray:
        """The curve as drawn: a gentle blur, mirrored at the ends so clipping keeps its full height."""
        padded = np.pad(row.astype(float), 5, mode="symmetric")
        return np.convolve(padded, cls._KERNEL, mode="valid")

    @staticmethod
    def peak(rows: np.ndarray) -> float:
        """The height the curves are scaled to. Clipping at either end, and a spike or two inside
        (a film border, a flat sky), would flatten everything else: each curve's 7th-tallest
        point counts, which a spike (3 points wide once smoothed) never reaches."""
        inner = np.sort(rows[:, 1:255], axis=1)[:, -7]
        return float(inner.max()) * 1.05 or float(rows.max()) or 1.0

    @staticmethod
    def _curve(row: np.ndarray, peak: float, area: QRectF) -> tuple[QPainterPath, QPainterPath]:
        """One curve: a smooth line through the values, and the area under it to fill."""
        ys = area.bottom() - np.minimum(row / peak, 1.0) * area.height()
        xs = area.left() + (np.arange(256) + 0.5) / 256 * area.width()
        pts = [QPointF(x, y) for x, y in zip(xs, ys)]
        line = QPainterPath(QPointF(area.left(), ys[0]))
        line.lineTo(pts[0])
        for i in range(255):                                        # Catmull-Rom, as cubic Béziers
            a, b, c, d = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, 255)]
            line.cubicTo(b + (c - a) / 6, c - (d - b) / 6, c)
        line.lineTo(QPointF(area.right(), ys[-1]))
        under = QPainterPath(line)
        under.lineTo(QPointF(area.right(), area.bottom()))
        under.lineTo(QPointF(area.left(), area.bottom()))
        under.closeSubpath()
        return line, under

    def _fill(self, area: QRectF, r: int, g: int, b: int, top: int, bottom: int) -> QLinearGradient:
        grad = QLinearGradient(0, area.top(), 0, area.bottom())    # fades towards the floor, like the faders
        grad.setColorAt(0, QColor(r, g, b, top))
        grad.setColorAt(1, QColor(r, g, b, bottom))
        return grad

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        box = QRectF(self.rect())
        frame = QPainterPath()
        frame.addRoundedRect(box, 6, 6)
        p.fillPath(frame, QColor(COLORS["field"]))
        p.setClipPath(frame)                                       # the curves follow the rounded corners
        p.setPen(QPen(QColor(COLORS["line"]), 1))              # quarter marks: shadows, mids, highlights
        for i in (1, 2, 3):
            x = round(box.left() + box.width() * i / 4) + 0.5
            p.drawLine(QPointF(x, box.top() + 4), QPointF(x, box.bottom() - 4))
        if self.counts is None or not self.counts[3].any():
            p.end()
            return
        area = box.adjusted(0, 8, 0, 0)
        counts = self.counts[:3] if self.mode == "rgb" else self.counts[3:]
        peak = self.peak(np.array([self._smooth(r) for r in counts]))
        rows = [self.soften(r) for r in counts]
        if self.mode == "rgb":
            p.setCompositionMode(QPainter.CompositionMode_Plus)    # overlaps add up towards white
            colors = CHANNELS
            fill, stroke = (120, 40), 200
        else:
            grey = QColor(COLORS["text"])
            colors = ((grey.red(), grey.green(), grey.blue()),)
            fill, stroke = (95, 25), 150
        for row, (r, g, b) in zip(rows, colors):
            line, under = self._curve(row, peak, area)
            p.fillPath(under, self._fill(area, r, g, b, *fill))
            pen = QPen(QColor(r, g, b, stroke), 1.2)
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            p.strokePath(line, pen)
        p.end()


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
            if self.edited(index.row()):          # adjusted by hand: a small violet mark under its corner
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(ACCENT2))
                p.drawRect(QRectF(target.right() - 10, target.bottom() + 3, 10, 2))
        p.restore()

    def sizeHint(self, option, index):
        return QSize(THUMB_SIZE + 22, THUMB_H + 18)


# Fader colours: the top (+) and bottom (−) end of each control
FADER_COLORS = {"plus": "#e6e6e6", "minus": "#6b6d73",
                "dens+": "#8a8c92", "dens-": "#f2f2f2", "con+": "#ffffff", "con-": "#8a8c92",
                "sat+": "#ff9f1c", "sat-": "#8a8c92",
                "warm+": "#ffa62b", "cool-": "#4f8cff", "mag+": "#ec1e79", "grn-": "#22b14c",
                "fx+": "#f2f2f2", "fx-": "#55575d",
                "plusC": "#29abe2", "minusR": "#ff3b30",
                "plusM": "#ec1e79", "minusG": "#22b14c",
                "plusY": "#ffe600", "minusB": "#3a63ff"}


def _tex_density(t: float, x: float) -> QColor:
    """Up = darker print, down = lighter."""
    g = round(25 + 215 * t)
    return QColor(g, g, g)


def _tex_contrast(t: float, x: float) -> QColor:
    """Up: black and white pull apart; down: they merge into flat grey."""
    a = 1 - t
    g = 128 - 118 * a if x < 0.5 else 128 + 118 * a
    return QColor(round(g), round(g), round(g))


def _tex_saturation(t: float, x: float) -> QColor:
    """Up: vivid colours; down: grey."""
    return QColor.fromHsvF(x * 0.9, 1 - t, 0.8)


def _tone_split(base: float, t: float, x: float) -> QColor:
    """A dark and a light half that pull apart going up (hard) and merge going down (soft)."""
    sep = 55 * (1 - t)
    g = base - sep if x < 0.5 else base + sep
    g = round(min(max(g, 0), 255))
    return QColor(g, g, g)



def _tex_glow(t: float, x: float) -> QColor:
    """A bright line whose halo widens going up."""
    import math
    d = abs(x - 0.5)
    halo = math.exp(-(d / (0.03 + 0.4 * (1 - t))) ** 2) * (1 - t)
    core = 1.0 if d < 0.08 else 0.0
    g = round(30 + 225 * max(core, halo))
    return QColor(g, g, g)


def _tex_sharpen(t: float, x: float) -> QColor:
    """A dark-to-light edge that gets crisper going up."""
    import math
    width = 0.03 + 0.45 * t
    g = round(40 + 200 / (1 + math.exp(-(x - 0.5) / (width / 4))))
    return QColor(g, g, g)


def _tex_noise(t: float, x: float) -> QColor:
    """Coloured speckle on grey that calms down going up."""
    h = (int(t * 997) * 7919 + int(x * 89) * 104729) % 1000003
    amp = 70 * t
    r, g, b = ((h >> k) % 255 / 255 - 0.5 for k in (0, 5, 11))
    return QColor(*(round(min(max(120 + amp * c, 0), 255)) for c in (r, g, b)))


TEXTURES = {"density": _tex_density, "contrast": _tex_contrast, "saturation": _tex_saturation,
            "highlight_tone": lambda t, x: _tone_split(200, t, x), "shadow_tone": lambda t, x: _tone_split(55, t, x),
            "glow": _tex_glow, "sharpen": _tex_sharpen, "color_noise": _tex_noise}


class FaderTrack(QWidget):
    """The painted part of a fader: a track from the + colour (top) to the − colour (bottom),
    a zero line in the middle, a bar growing from zero, and a knob at the current value.

    Drag the knob, scroll, or double-click to reset. It shows ±SPAN steps; beyond that the
    knob waits at the end and a small arrow shows the value goes further.
    """

    SPAN = 20                                   # steps from the middle to each end

    def __init__(self, fader: "Stepper", top: str, bottom: str, texture=None, unipolar: bool = False,
                 span: int = 20):
        super().__init__()
        self.SPAN = span                 # steps from the zero line to the end of the track
        self.fader, self.top, self.bottom = fader, QColor(top), QColor(bottom)
        self.unipolar = unipolar       # zero at the bottom: for controls that only go up (Glow)
        self.texture = texture         # fn(t 0=top..1=bottom, x 0..1) -> QColor: a picture of the effect
        self._tex_img: QImage | None = None
        self.active = False
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
        pad, h, cy, px = self._geometry()
        cx = self.width() / 2
        steps = self.fader.steps()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        shown = max(0 if self.unipolar else -self.SPAN, min(self.SPAN, steps))
        y = cy - shown * px
        if self.texture:
            track = QRectF(cx - 8, pad, 16, h)
            self._paint_texture(p, track, cy, y)
        else:
            track = QRectF(cx - 6, pad, 12, h)
            self._paint_gradient(p, track, pad, h, cy, y, steps)
        self._paint_ticks_knob(p, cx, cy, px, y, steps)
        p.end()

    def _paint_texture(self, p: QPainter, track: QRectF, cy: float, y: float):
        """The effect itself, dimmed; full strength between zero and the knob."""
        w, h = int(track.width()), int(track.height())
        if self._tex_img is None or self._tex_img.height() != h:
            img = QImage(w, h, QImage.Format_RGB32)
            for j in range(h):
                for i in range(w):
                    img.setPixelColor(i, j, self.texture(j / max(h - 1, 1), (i + 0.5) / w))
            self._tex_img = img
        path = QPainterPath()
        path.addRoundedRect(track, 6, 6)
        p.save()
        p.setClipPath(path)
        p.setOpacity(0.6)
        p.drawImage(track.topLeft(), self._tex_img)
        p.setOpacity(1.0)
        bar = QRectF(track.left(), min(y, cy), track.width(), abs(y - cy))
        p.drawImage(bar, self._tex_img, bar.translated(-track.left(), -track.top()))
        p.restore()
        p.setPen(QPen(QColor(ACCENT), 2) if self.active else QPen(QColor("#111214"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(track, 6, 6)

    def _paint_gradient(self, p: QPainter, track: QRectF, pad: float, h: float, cy: float, y: float, steps: int):
        cx = track.center().x()
        # track: faint + colour at the top, faint − colour at the bottom
        grad = QLinearGradient(0, pad, 0, pad + h)
        t, b = QColor(self.top), QColor(self.bottom)
        t.setAlpha(90)
        b.setAlpha(90)
        mid = QColor("#2a2c31")
        grad.setColorAt(0, t)
        grad.setColorAt(0.5, mid)
        grad.setColorAt(1, b)
        p.setPen(QPen(QColor(ACCENT), 2) if self.active else QPen(QColor("#111214"), 1))
        p.setBrush(grad)
        p.drawRoundedRect(track, 6, 6)
        # the bar from zero to the value, in the colour of the direction it went
        if steps:
            p.setPen(Qt.NoPen)
            p.setBrush(self.top if steps > 0 else self.bottom)
            p.drawRect(QRectF(cx - 3.5, min(y, cy), 7, abs(y - cy)))

    def _paint_ticks_knob(self, p: QPainter, cx: float, cy: float, px: float, y: float, steps: int):
        # ticks every 5 steps, a longer one at zero
        knob_y = y
        for k in range(0 if self.unipolar else -self.SPAN, self.SPAN + 1, 5):
            y = cy - k * px
            w = 7 if k == 0 else 4
            p.setPen(QPen(QColor("#c8c8c8") if k == 0 else QColor("#55575d"), 1.5 if k == 0 else 1))
            p.drawLine(QPointF(cx - 9 - w, y), QPointF(cx - 9, y))
            p.drawLine(QPointF(cx + 9, y), QPointF(cx + 9 + w, y))
        y = knob_y
        # knob
        p.setPen(QPen(QColor("#111214"), 1))
        p.setBrush(QColor("#f2f2f2"))
        p.drawRoundedRect(QRectF(cx - 13, y - 4, 26, 8), 1.5, 1.5)
        if abs(steps) > self.SPAN:                 # goes further than the track shows
            p.setPen(QPen(QColor("#f2f2f2"), 1.5))
            d = -1 if steps > 0 else 1
            p.drawLine(QPointF(cx - 4, y + d * 9), QPointF(cx, y + d * 13))
            p.drawLine(QPointF(cx + 4, y + d * 9), QPointF(cx, y + d * 13))

    def _steps_at(self, y: float) -> int:
        _, _, cy, px = self._geometry()
        return round((cy - y) / px)

    def mousePressEvent(self, e):
        self.fader.set_steps(self._steps_at(e.position().y()))

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
                 step: float = 0.05, lo: float = 0.0, hi: float = 3.0, default: float = 1.0, texture=None,
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
        self.track = FaderTrack(self, top, bottom, texture, unipolar, span)
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
        b.setStyleSheet(f"QToolButton {{ color: {color}; background: transparent; border: none; font-size: 12px; }}"
                        f"QToolButton:hover {{ background: {COLORS['button']}; }}")
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

    PATTERN = "dots"  # "dots" or "lines"
    SPACING = 16      # px between dots / lines
    DOT = 1.0         # dot size or line width in px (1 px = 2 device pixels on Retina)

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


class RollCard(QFrame):
    """One roll (or the single frames) in the left column: its name, then film stock and frame count.
    A roll's chevron expands the card to show its details (the details widget is moved in)."""

    clicked = Signal()
    toggled = Signal(bool)
    menu = Signal(object)                 # right-click: the global position for a menu

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
        self.title = QLabel()
        self.title.setObjectName("rollTitle")
        self.sub = QLabel()
        self.sub.setObjectName("rollSub")
        for w in (self.title, self.sub):
            w.setAttribute(Qt.WA_TransparentForMouseEvents)
            text.addWidget(w)
        head.addLayout(text, 1)
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

    def _on_toggle(self, on: bool):
        self.toggled.emit(on)

    def set_text(self, title: str, sub: str):
        self.title.setText(title)
        self.sub.setText(sub)

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
        bg = mix(mix(COLORS["panel"], COLORS["raise"], h), COLORS["select"], o)
        title = mix(ACCENT if self._selected else COLORS["text"], COLORS["on_select"], o)
        sub = mix(COLORS["muted"], COLORS["on_select_muted"], o)
        self.head.setStyleSheet(f"QFrame#rollHead {{ background: {bg}; border-radius: 9px; }} "
                                f"QLabel#rollTitle {{ color: {title}; }} "
                                f"QLabel#rollSub {{ color: {sub}; }}")
        self.chevron.on_light = o > 0.5
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
        self.slider = QSlider(Qt.Horizontal)
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
        self.center = QPointF(0.5, 0.5)
        self.zoom_100 = 0.0                  # the zoom at which one image pixel = one screen pixel (0 = unknown)
        self.hires: QPixmap | None = None
        self.loading = False
        self._press: QPointF | None = None
        self._panned = False
        self._hint_timer = QTimer(singleShot=True, interval=1200)
        self._hint_timer.timeout.connect(self.update)

    def set_image(self, pix: QPixmap | None, draft: bool = False):
        """A draft (a quicker, smaller render while a slider moves) keeps the frame of the last full
        render: their shapes differ by a hair, which would nudge the picture up and down."""
        self.source = pix
        if pix is not None and pix.height() > 0:
            aspect = pix.width() / pix.height()
            if not draft or not self._aspect or abs(aspect - self._aspect) / self._aspect > 0.01:
                old, self._aspect = self._aspect, aspect
                if self.crop_mode and self.ratio and self.crop and old and abs(aspect - old) / old > 0.001:
                    self._refit_crop()       # a format picked before the whole picture arrived
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

    def image_rect(self) -> QRectF:
        """Where the picture sits inside the label at fit: on whole pixels, from the last full render's shape."""
        if not self.source:
            return QRectF()
        aspect = self._aspect or self.source.width() / max(self.source.height(), 1)
        w, h = self.width(), self.height()
        if w / max(h, 1) > aspect:
            w = int(round(h * aspect))
        else:
            h = int(round(w / aspect))
        return QRectF((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def paintEvent(self, e):
        if not self.source:
            return super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = self.view_rect()
        if self.zoom > 1.0:
            pass                             # the window background draws it, across the whole window
        elif abs(self.angle) > 0.001:
            # Live preview: turn the last render, zoomed so no empty corners show
            # (the real render crops the corners the same way).
            import math
            a = math.radians(abs(self.angle))
            ratio = max(r.width() / max(r.height(), 1), r.height() / max(r.width(), 1))
            zoom = math.cos(a) + math.sin(a) * ratio
            p.save()
            p.setClipRect(r)
            p.translate(r.center())
            p.rotate(-self.angle)            # positive angle = counter-clockwise, like NegPy
            p.scale(zoom, zoom)
            p.translate(-r.center())
            p.drawPixmap(r.toRect(), self.source)
            p.restore()
        else:
            p.drawPixmap(r.toRect(), self.source)
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
            violet = QColor(ACCENT2)
            wash = QColor(ACCENT2)
            wash.setAlpha(26)
            for shape in self.area_marks:             # the areas already being cleaned
                poly = QPolygonF([QPointF(r.x() + q.x() * r.width(), r.y() + q.y() * r.height()) for q in shape])
                p.setPen(QPen(QColor(0, 0, 0, 130), 2.5))
                p.setBrush(Qt.NoBrush)
                p.drawPolygon(poly)
                p.setPen(QPen(violet, 1.2, Qt.DashLine))
                p.setBrush(wash)
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

    def _badge(self, p: QPainter, text: str, where):
        f = QFont()
        f.setPixelSize(11)
        p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(text) + 18
        x = 10 if where & Qt.AlignLeft else self.width() - w - 10
        y = 10 if where & Qt.AlignTop else self.height() - 32
        box = QRectF(x, y, w, 22)
        p.fillRect(box, QColor(0, 0, 0, 170))
        p.setPen(QColor("#e8e6e3"))
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
            self.brush_wheel.emit(1 if (dy or e.angleDelta().x()) > 0 else -1)
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

    def __init__(self, parent, opts: engine.ExportOptions, extra: dict, has_selection: bool, has_roll: bool):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(560)
        form = QFormLayout()
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)   # macOS keeps text fields narrow otherwise

        self.scope = exif_panel.ChevronCombo()
        self.scope.addItem("All images (every roll and single frame)", "all")
        self.scope.addItem("This group only", "group")
        if has_selection:
            self.scope.addItem("Selected frames", "selection")
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
                                    pattern=self.pattern.text(), overwrite=self.overwrite.currentData())
        extra = {"scope": self.scope.currentData(), "per_roll": self.per_roll.isChecked(),
                 "dest": self.dest.text().strip(), "reveal": self.reveal.isChecked()}
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
        "highlight_tone": "Contrast of the highlights, soft ← → hard: right makes skies and skin crisp, left keeps them soft",
        "shadow_tone": "Contrast of the shadows, soft ← → hard: right gives deep, inky blacks, left lifts them to a soft, matte look",
        "shadow_warmth": "Shadows only: cooler (blue) ← → warmer (amber)",
        "shadow_tint": "Shadows only: green ← → magenta. Right removes green fog",
        "highlight_warmth": "Highlights only: cooler (blue) ← → warmer (amber)",
        "highlight_tint": "Highlights only: green ← → magenta. Left removes a red-magenta cast",
        "glow": "A soft glow around bright areas, like the soft focus of a minilab scan",
        "sharpen": "How crisp fine detail is. Double-click to go back to the default",
        "color_noise": "Smooths coloured speckle in the grain. More can wash out fine colour detail",
    }
    STEPPERS = ["contrast", "density", "saturation", "cyan", "magenta", "yellow"]

    def __init__(self):
        super().__init__()
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

        # While a control is moving: quick draft renders, back to back. Once it rests for a
        # moment: one full-quality render (and the thumbnails of other selected frames).
        self.render_timer = QTimer(singleShot=True, interval=30)
        self.render_timer.timeout.connect(lambda: self.request_preview(draft=True))
        # sharp only once you really stop: switching on every hesitation of a drag looked like a strobe
        self._settle_timer = QTimer(singleShot=True, interval=450)
        self._settle_timer.timeout.connect(self._settled)
        self._pending_full = False
        self._dirty_thumbs: set = set()
        self._thumb_timer = QTimer(singleShot=True, interval=600)
        self._thumb_timer.timeout.connect(self._refresh_all_thumbs)
        self._reanalyze_timer = QTimer(singleShot=True, interval=900)
        self._reanalyze_timer.timeout.connect(self._reanalyze_rolls)
        # an edit becomes one undo step once the clicking/dragging pauses
        self._history_timer = QTimer(singleShot=True, interval=500)
        self._history_timer.timeout.connect(self._commit_history)
        self._sync_base: engine.Settings | None = None    # for applying edits to selected frames
        self._show_before = False
        self._adjusting: str | None = None        # Settings field changed by ↑ / ↓ while its key is held
        self._rendered_angle = 0.0                # fine rotation of the picture on screen
        self._request_angle: dict[int, float] = {}
        self._request_draft: dict[int, bool] = {}
        self._straighten_timer = QTimer(singleShot=True, interval=250)   # real render once dragging pauses
        self._straighten_timer.timeout.connect(self.request_preview)
        self._clipboard: engine.Settings | None = None
        self._fullscreen = False
        self._was_maximized = False
        self._recovery_on = True                      # off while another running window owns the file
        self._recovery_text = ""
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
                        ("Ctrl+Shift+Z", self.redo), ("Ctrl+E", self.open_export),
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
        self.export_btn.setVisible(bool(self.groups) and working)
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
        self.preview.reset_zoom()
        self.preview.setFocus()
        self.backdrop.update()

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
            self.toggle_pick_wb()
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
        self.grid.setObjectName("grid")
        self.grid.currentRowChanged.connect(self.on_select)
        self.grid.itemSelectionChanged.connect(lambda: self.refresh_exif())

        self.cards = QWidget()                        # one RollCard per roll (and one for single frames)
        self.cards_lay = QVBoxLayout(self.cards)
        self.cards_lay.setContentsMargins(0, 0, 0, 0)
        self.cards_lay.setSpacing(8)
        self.cards_lay.addStretch(1)
        cards_scroll = QScrollArea()
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
        left_lay.addWidget(hdr)
        add_row = QHBoxLayout()
        add_row.addWidget(button("Add Roll", self.open_roll))
        add_row.addWidget(button("Add Frames", self.open_frames))
        left_lay.addLayout(add_row)
        left_lay.addWidget(cards_scroll, 1)
        self.left = QWidget()
        self.left.setLayout(left_lay)
        self.left.setFixedWidth(280)

        # ---- Center: title, preview, Edit Crop
        self.title = QLabel("")
        self.title.setObjectName("title")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # long names never widen the window
        self.preview = PreviewLabel()
        self.preview.setFocusPolicy(Qt.ClickFocus)
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
        tb = FlowLayout(self.toolbar, spacing=6)  # one joined strip; wraps onto a second row in a narrow window
        copy_btn = button("Copy Settings", self.copy_settings, "bar")
        copy_btn.setToolTip("Copy this frame's adjustments (Cmd+C)")
        self.paste_btn = button("Paste to Selected", self.paste_settings, "bar")
        self.paste_btn.setToolTip("Applies the copied adjustments to the selected frames (crop and rotation stay)")
        self.heal_btn = button("Heal Dust", self.toggle_heal, "bar")
        self._tip(self.heal_btn, "Click each dust speck to remove it; Alt+scroll changes the brush size", "heal")
        reset_btn = button("Reset", self.reset_current, "bar")
        self._tip(reset_btn, "Back to the automatic conversion", "reset")
        self.area_btn = button("Clean Area", self.toggle_heal_area, "bar")
        self._tip(self.area_btn, "Draw around a part of the photo: dust and scratches found inside it are removed", "heal_area")
        self._bar_btns = (copy_btn, self.paste_btn, self.heal_btn, self.area_btn, reset_btn)
        for b in self._bar_btns:
            tb.addWidget(b)
        center = QVBoxLayout()
        center.addWidget(self.title, 0, Qt.AlignHCenter)
        center.addWidget(self.stack, 1)
        center.addWidget(self.toolbar)
        self.toolbar.setVisible(False)
        # While cropping, the strip under the photo offers film formats instead
        self.toolbar_crop = QWidget()
        tbc = FlowLayout(self.toolbar_crop, spacing=6)
        self.ratio_btns: dict[str, QPushButton] = {}
        for name, tip in self.CROP_FORMATS_TIPS.items():
            b = button(name, lambda n=name: self.set_crop_format(n), "bar")
            b.setCheckable(True)
            b.setToolTip(tip)
            self.ratio_btns[name] = b
            tbc.addWidget(b)
        self.ratio_btns["Free"].setChecked(True)
        for text, slot, tip in (("Turn", self.crop_turn, "Swap between upright and landscape"),
                                ("Whole Frame", self.crop_reset, "Back to the whole picture"),
                                ("Apply", self.toggle_crop, "Apply the crop (Return); Esc cancels")):
            b = button(text, slot, "barGo" if text == "Apply" else "bar")
            b.setToolTip(tip)
            tbc.addWidget(b)
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
            row.addWidget(IconButton(draw, arg, slot, tip), 1)
        lay.addLayout(row)
        crop_label = QLabel("Crop")
        crop_label.setObjectName("rowLabel")
        self.autocrop_cb = button("Auto", None, "toggle")  # a switch: neon while on
        self.autocrop_cb.setCheckable(True)
        self._tip(self.autocrop_cb, "Find the picture edges and crop the film border away", "auto_crop")
        self.autocrop_cb.toggled.connect(self.on_autocrop)
        self.crop_btn.setObjectName("toggle")
        self._tip(self.crop_btn, "Crop the photo", "crop")
        crop_row = QHBoxLayout()
        crop_row.setSpacing(6)
        crop_row.addWidget(crop_label)
        crop_row.addStretch(1)
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
        self.autocrop_cb.setFixedWidth(angle_width)
        self.straighten.value_input.setFixedSize(angle_width, self.autocrop_cb.sizeHint().height())
        self.straighten.top.setSpacing(6)
        self.straighten.top.addWidget(self.auto_straighten_btn)
        lay.addWidget(self.straighten)
        right.addWidget(p)

        self.steppers: dict[str, Stepper] = {}
        p, lay = panel()                                  # all six faders, like a mixing desk
        self.histogram = Histogram(pref("histogram"))     # above them, what they do to the picture
        hist_label = QLabel("Histogram")
        hist_label.setObjectName("rowLabel")
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(hist_label)
        row.addStretch(1)
        self.hist_mode_btn = ChannelButton(self.histogram.mode)
        self.hist_mode_btn.mode_changed.connect(self.set_histogram_mode)
        row.addWidget(self.hist_mode_btn)
        lay.addLayout(row)
        lay.addWidget(self.histogram)
        row = QHBoxLayout()
        row.setSpacing(0)
        for key, title, up, down in [("density", "D", "dens+", "dens-"), ("contrast", "Con", "con+", "con-"),
                                     ("saturation", "Sat", "sat+", "sat-")]:
            self.steppers[key] = Stepper(title, up, down, step=0.05, lo=0.0, hi=3.0, texture=TEXTURES[key])
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
        self.autowb_btn = button("Auto WB", self.auto_wb)
        self.autowb_btn.setToolTip("Make the average colour of the photo neutral")
        self.pick_btn = button("Pick Grey", self.toggle_pick_wb)
        self._tip(self.pick_btn, "Click something in the photo that should be grey or white; C/M/Y are set to make it neutral", "pick_wb")
        reset_wb = button("Reset WB", self.reset_wb)
        row.setSpacing(6)
        for b in (self.autowb_btn, self.pick_btn, reset_wb):
            row.addWidget(b)
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
        tabs.addWidget(plate, 0, Qt.AlignVCenter)
        tabs.addStretch(1)
        # Export: top right on every tab, as tall as the tabs; lights up once there is something to export
        self.export_btn = button("Export", self.open_export)
        self.export_btn.setObjectName("export")
        self.export_btn.setFixedSize(360, 36)          # as wide as the right column below it
        keep = self.export_btn.sizePolicy()
        keep.setRetainSizeWhenHidden(True)            # hidden, it still holds its space: the tab row never jumps
        self.export_btn.setSizePolicy(keep)
        self.export_btn.setVisible(False)             # appears with the first pictures
        tabs.addWidget(self.export_btn)
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
        self.stock_box = exif_panel.ChevronCombo()
        self.stock_box.setEditable(True)
        self.stock_box.setInsertPolicy(QComboBox.NoInsert)
        self.stock_box.addItem("Film stock not set", "")
        for d in engine.film_stocks():
            self.stock_box.addItem(d["displayName"], d["id"])
        comp = self.stock_box.completer()
        comp.setFilterMode(Qt.MatchContains)          # type "gold" to find Kodak Gold 200
        comp.setCompletionMode(QCompleter.PopupCompletion)
        self.stock_box.lineEdit().setPlaceholderText("Type to search a film stock")
        self.stock_box.activated.connect(self.on_stock_changed)
        self.stock_box.setToolTip("B&W film converts in B&W; colour film uses its own dye matrix when known")
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
        self.roll_status.setMinimumHeight(64)
        for c in (self.stock_box, self.push_box):        # the column's width, not the longest film name
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(6)
        for w in (self.stock_box, self.push_box, self.camera_edit, self.lens_edit, self.date_edit,
                  self.place_edit, self.roll_status):
            w.setMinimumWidth(0)
            lay.addWidget(w)
        for w in (self.camera_edit, self.lens_edit, self.place_edit):
            w.returnPressed.connect(self.escape)     # Return: done typing, keys drive the scanner again
        sheet = button("Contact Sheet", lambda: self.open_contact_sheet(), "toggle")
        sheet.setToolTip("Every frame of this roll on one sheet, like a darkroom contact print")
        lay.addWidget(sheet)
        remove = button("Remove Roll from List", lambda: self.group and self.remove_group(self.group), "toggle")
        remove.setToolTip("Takes the roll out of the app. The scans on disk are not touched")
        lay.addWidget(remove)
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

        # New frames
        lay = section("New frames")
        self.pref_autocrop = switch("auto_crop_new")
        row(lay, "Auto crop", "Crop the film border away on every frame you add (and after Reset).", self.pref_autocrop)

        # While working
        lay = section("While working")
        self.pref_warnings = switch("scan_warnings", lambda _on: self.update_nav())
        row(lay, "Scan warnings", "Say so next to the file name when a scan is very dark or blown out.", self.pref_warnings)

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
        self.more_btn = FadeButton("REFINE")              # the whole header row is the switch
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

        self.more_body = QWidget()
        body = QVBoxLayout(self.more_body)
        body.setContentsMargins(0, 0, 0, 0)

        body.setSpacing(2)
        self.sliders: dict[str, LineSlider] = {}
        defaults = engine.Settings()

        def line(key, title, lo, hi, unipolar=False, minus="", plus=""):
            f = LineSlider(title, lo, hi, default=getattr(defaults, key), unipolar=unipolar,
                           minus=FADER_COLORS.get(minus, ""), plus=FADER_COLORS.get(plus, ""))
            f.setToolTip(self.TOOLTIPS.get(key, ""))
            f.changed.connect(lambda _: self.on_setting_changed())
            self.sliders[key] = f
            body.addWidget(f)

        def head(text):
            h = QLabel(text.upper())
            h.setObjectName("header")
            body.addSpacing(12)
            body.addWidget(h)
            body.addSpacing(2)

        # Contrast of one end of the tonal range, like split-grade printing in a darkroom
        head("Split contrast")
        line("highlight_tone", "Highlights", -1.0, 1.0)
        line("shadow_tone", "Shadows", -1.0, 1.0)
        head("Finish")
        line("sharpen", "Sharpen", 0.0, 1.0, unipolar=True)
        line("color_noise", "Colour noise", 0.0, 1.0, unipolar=True)
        line("glow", "Glow", 0.0, 1.0, unipolar=True)
        # Casts that sit in one end of the tonal range (green shadows, red highlights)
        head("Shadow cast")
        line("shadow_warmth", "Warmth", -1.0, 1.0, minus="cool-", plus="warm+")
        line("shadow_tint", "Tint", -1.0, 1.0, minus="grn-", plus="mag+")
        head("Highlight cast")
        line("highlight_warmth", "Warmth", -1.0, 1.0, minus="cool-", plus="warm+")
        line("highlight_tint", "Tint", -1.0, 1.0, minus="grn-", plus="mag+")
        body.addSpacing(14)
        recalc = button("Re-measure Inside Crop", self.recalc)
        recalc.setToolTip("Measure the film again using only the area inside the crop, then Auto WB. "
                          "Helps when the film border or a light leak threw the colours off")
        body.addWidget(recalc)
        body.addSpacing(2)
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
        self.cards_lay.insertWidget(self.cards_lay.count() - 1, g.card)
        for f in imgs:
            self._add_photo(g, f)
        self._refresh_group_item(g)
        self._fit_left_column()
        self.right_stack.setVisible(True)
        self.grid.setVisible(True)
        self.export_btn.setVisible(self.main_stack.currentIndex() == 0)
        self.stack.setCurrentIndex(1)
        self._update_toolbars()

    def _add_photo(self, g: Group, f: Path):
        photo = Photo(path=f, group=g)
        photo.settings.flat_field = self.flat_field
        photo.settings.auto_crop = pref("auto_crop_new")
        engine.settings_for_stock(photo.settings, g.stock)
        photo.history, photo.hindex = [photo.settings.copy()], 0     # the automatic conversion: where undo ends
        g.photos.append(photo)
        self.render_thumb(photo)
        if photo.settings.auto_crop:
            self.auto_level(photo, automatic=True)

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
        for other in self.groups:
            other.card.set_selected(other is g)
            other.card.set_open(other is g and show)
            other.card.chevron.blockSignals(True)
            other.card.chevron.setChecked(other is g and show)
            other.card.chevron.update()
            other.card.chevron.blockSignals(False)

    # ----- Removing frames and rolls (from the list only: files on disk are never touched) ------
    def _frame_menu(self, pos):
        if not self.grid.itemAt(pos):
            return
        n = len(self.grid.selectedItems())
        m = QMenu(self)
        m.addAction(f"Remove {n} Frames from List" if n > 1 else "Remove Frame from List", self.remove_selected_frames)
        m.exec(self.grid.viewport().mapToGlobal(pos))

    def _group_menu(self, g: Group, pos):
        m = QMenu(self)
        if g.kind == "roll":
            m.addAction("Contact Sheet…", lambda: self.open_contact_sheet(g))
            m.addSeparator()
        m.addAction("Remove Roll from List" if g.kind == "roll" else "Remove All Single Frames from List",
                    lambda: self.remove_group(g))
        m.exec(pos)

    def _forget(self, photos: list[Photo]):
        """Drop every reference the window keeps to these frames."""
        gone = set(photos)
        if self._stale_thumb in gone:
            self._stale_thumb = None
        if self._uv and self._uv[0] in gone:
            self._uv = None
        self._dirty_thumbs -= gone
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
        self.histogram.set_image(None)
        self.preview.setText("")
        self.title.setText("")
        self.stack.setCurrentIndex(0)
        self.right_stack.setVisible(False)
        self.grid.setVisible(False)
        self.left.setVisible(False)
        self.export_btn.setVisible(False)
        self._update_toolbars()

    # ----- Safety net: recovery after a crash, a question before losing work -------------------
    # The app is for quick jobs: open, convert, close. So nothing is "saved" as a project. But the
    # work in progress is written to a small recovery file every few seconds, removed again on a
    # normal quit. If the file is still there at the next start, the app did not quit normally.
    @staticmethod
    def _recovery_path() -> Path:
        from PySide6.QtCore import QStandardPaths
        return Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or Path.home() / ".dumbconvert") / "recovery.json"

    def _session(self) -> dict:
        from dataclasses import asdict
        return {"pid": os.getpid(), "groups": [
            {"kind": g.kind, "name": g.name, "stock_id": g.stock_id, "camera": g.camera, "lens": g.lens, "date": g.date,
             "location": g.location, "location_label": g.location_label, "push_pull": g.push_pull,
             "photos": [{"path": str(ph.path), "settings": asdict(ph.settings), "exif": ph.exif} for ph in g.photos]}
            for g in self.groups]}

    def _write_recovery(self):
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
            for k in ("stock_id", "camera", "lens", "date", "location", "location_label", "push_pull"):
                setattr(g, k, gd.get(k, getattr(g, k)))
            self._add_group(g, [])
            for e in entries:
                photo = Photo(path=Path(e["path"]), group=g)
                try:
                    photo.settings = engine.Settings(**{k: tuples(v) for k, v in e.get("settings", {}).items() if k in fields_})
                except TypeError:
                    pass
                photo.settings.flat_field = self.flat_field
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

    def _confirm_quit(self, text: str, detail: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.NoIcon)
        box.setText(text)
        box.setInformativeText(detail)
        quit_btn = box.addButton("Quit Anyway", QMessageBox.DestructiveRole)
        box.addButton("Go Back", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])
        box.exec()
        return box.clickedButton() is quit_btn

    def closeEvent(self, e):
        self._commit_history()
        if self._sheets and not self._confirm_quit("A contact sheet is still being made.",
                                                   "Quitting now leaves it unfinished."):
            e.ignore()
            return
        if self._export_left > 0:
            done = self._export_total - self._export_left
            if not self._confirm_quit("An export is still running.",
                                      f"{done} of {self._export_total} files are written. Quitting now stops the rest."):
                e.ignore()
                return
        else:
            n = len(self.unexported())
            if n and not self._confirm_quit(
                    f"{n} adjusted frame{'s have' if n > 1 else ' has'} not been exported.",
                    "Your adjustments are not kept after you quit."):
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
        if g.photos:
            self.grid.setCurrentRow(0, QItemSelectionModel.ClearAndSelect)
        self.update_nav()
        if not self._typing():
            self.preview.setFocus()

    def _show_roll_details(self, g: Group):
        if g.kind != "roll":
            return
        self.stock_box.setCurrentIndex(max(self.stock_box.findData(g.stock_id), 0))
        self.camera_edit.set_value(g.camera)
        self.lens_edit.set_value(g.lens)
        self.date_edit.set_iso(g.date)
        self.place_edit.setText(g.location_label or g.location)
        self.push_box.setCurrentIndex(max(self.push_box.findData(g.push_pull), 0))
        self._update_roll_status(g)

    def _update_roll_status(self, g: Group):
        if g is not self.group or g.kind != "roll":
            return
        if g.analyzing:
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
        self._set_roll_stock(g, self.stock_box.currentData() or "")

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
        self.analyze_roll(g)   # a different dye matrix changes what the film measures

    # ----- Roll analysis ---------------------------------------------------
    def analyze_roll(self, g: Group):
        if g.kind != "roll" or not g.photos:
            return
        if g.analyzing:                 # run again with the new settings once this one ends
            g.analyzed = False
            self._reanalyze_timer.start()
            return
        members = [ph for ph in g.photos if not ph.failed]
        frames = [(ph.path, ph.settings.copy()) for ph in members]
        if not frames:
            return
        g.analyzing = True
        self._refresh_group_item(g)
        self._update_roll_status(g)
        self.pool.start(Job(lambda: engine.analyze_roll(frames) + (members,), lambda r: self.signals.roll.emit(g, r)))

    def on_roll_analyzed(self, g: Group, result):
        g.analyzing = False
        if g not in self.groups:
            return
        if isinstance(result, str):
            self.statusBar().showMessage(f"Roll analysis failed: {result}")
        else:
            floors, ceils, outliers, axis, members = result
            g.analyzed, g.outliers = True, sum(outliers)
            unusual = {ph for ph, out in zip(members, outliers) if out}
            for ph in g.photos:
                st = ph.settings
                if ph in unusual:      # far from the rest of the roll: its own measurement fits it better
                    st.roll_floors = st.roll_ceils = st.roll_axis = None
                else:
                    st.roll_floors, st.roll_ceils, st.roll_axis = floors, ceils, axis
                self.render_thumb(ph)
            if self.current in g.photos:
                self.request_preview()
            self.statusBar().showMessage(f"{g.name}: roll analysed.")
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
        st = p.settings.copy()
        self.pool.start(Job(lambda: engine.preview(p.path, st, THUMB_SIZE * 2),
                            lambda r: self.signals.thumb.emit(p, r)))

    def on_thumb(self, p: Photo, result):
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
            painter.setPen(QColor("#e8e6e3"))
            painter.drawText(tag, Qt.AlignCenter, text)
            painter.end()
        p.item.setIcon(QIcon(pix))

    # ----- Selecting -------------------------------------------------------
    def on_select(self, row: int):
        if self.preview.crop_mode:
            self.toggle_crop(apply=False)
        if row < 0 or row >= len(self.photos):
            return
        if self.photos[row] is not self.current:
            self._leave_frame()
        self.current = self.photos[row]
        self.preview.reset_zoom()                         # a new frame starts at fit
        self.grid.scrollToItem(self.current.item, QAbstractItemView.PositionAtCenter)
        if not self.current.history:
            self.current.history, self.current.hindex = [self.current.settings.copy()], 0
        self._sync_base = self.current.settings.copy()
        self.load_controls(self.current.settings)
        self.update_nav()
        self.request_preview()
        self.refresh_exif()
        if self.current.note is None:             # checked once per frame, quietly, off the UI thread
            p = self.current
            p.note = ""
            self.pool.start(Job(lambda: engine.scan_note(p.path), lambda r: self.signals.note.emit(p, r)))

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
            muted, text = COLORS["muted"], COLORS["text"]
            roll = (f"<span style='color:{text}; font-weight:600'>{self.group.name}</span>&nbsp;&nbsp;·&nbsp;&nbsp;"
                    if self.group and self.group.kind == "roll" else "")
            note = (f"&nbsp;&nbsp;·&nbsp;&nbsp;<span style='color:{COLORS['warn']}'>{self.current.note}</span>"
                    if self.current.note and self.pref_warnings.isChecked() else "")
            self.title.setText(f"{roll}<span style='color:{muted}'>{row + 1} / {n}&nbsp;&nbsp;·&nbsp;&nbsp;"
                               f"{self.current.path.name}</span>{note}")

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
        self.straighten.set_value(st.fine_rotation)
        for cb, val in ((self.autocrop_cb, st.auto_crop),):
            cb.blockSignals(True)
            cb.setChecked(val)
            cb.blockSignals(False)
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
            self._commit_history(ph)
            self._dirty_thumbs.add(ph)            # re-rendered once the slider rests
        self.statusBar().showMessage(f"Applied to {len(others) + 1} selected frames.")

    def edited(self):
        """Call after any edit of the current frame: a draft render now, a full one and one undo
        step after a pause.

        The draft timer is only started when idle, never restarted: restarting it on every
        slider movement would postpone the render until the slider stopped.
        """
        if not self.render_timer.isActive():
            self.render_timer.start()
        self._settle_timer.start()
        self._history_timer.start()
        self._stale_thumb = self.current       # until a full render of this edit has painted it

    def _settled(self):
        self.request_preview()
        for ph in self._dirty_thumbs:
            self.render_thumb(ph)
        self._dirty_thumbs.clear()

    def _commit_history(self, p: Photo | None = None):
        p = p or self.current
        if p is None:
            return
        if not p.history:
            p.history, p.hindex = [p.settings.copy()], 0
        elif not same_edits(p.history[p.hindex], p.settings):
            p.history = p.history[:p.hindex + 1] + [p.settings.copy()]
            p.hindex = len(p.history) - 1
        if p is self.current:
            self._update_undo_buttons()

    def _restore(self, idx: int):
        p = self.current
        snap = p.history[idx].copy()
        for f in NOT_EDITS:                       # keep what was measured since
            setattr(snap, f, getattr(p.settings, f))
        p.settings, p.hindex = snap, idx
        self._sync_base = snap.copy()
        self.load_controls(snap)
        self.request_preview()

    def undo(self):
        if not self.current:
            return
        if self._history_timer.isActive():        # finish the edit in progress first
            self._history_timer.stop()
            self._commit_history()
        if self.current.hindex > 0:
            self._restore(self.current.hindex - 1)

    def redo(self):
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
        for ph in targets:
            new = self._clipboard.copy()
            for f in NOT_EDITS + PER_FRAME:
                setattr(new, f, getattr(ph.settings, f))
            ph.settings = new
            self._commit_history(ph)
            self.render_thumb(ph)
        if self.current:
            self._sync_base = self.current.settings.copy()
            self.load_controls(self.current.settings)
            self.request_preview()
        self.statusBar().showMessage(f"Settings pasted to {len(targets)} frame(s).")

    def on_straighten_slider(self):
        if self.current:
            self.preview.reset_zoom()
            self.current.settings.fine_rotation = round(self.straighten.value(), 1)
            # instant: turn the picture already on screen; the real render follows after a pause
            self.preview.angle = self.current.settings.fine_rotation - self._rendered_angle
            self.preview.refresh()
            self._history_timer.start()
            self._straighten_timer.start()

    def _transform_shown(self, t: QTransform):
        """Turn/flip the picture on screen right away; the real render replaces it shortly."""
        if self.preview.source is not None and not self.preview.crop_mode:
            self.preview.set_image(self.preview.source.transformed(t, Qt.SmoothTransformation))
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
            self.current.settings.auto_crop = on
            if on:
                self.current.settings.crop = None   # auto replaces a manual crop
                self.auto_level(self.current)
            else:                                   # off: the frame as it was, the levelling included
                self._drop_levelling(self.current)
                self.straighten.set_value(self.current.settings.fine_rotation)
            self.edited()

    def auto_level(self, p: Photo, automatic: bool = False):
        """Auto crop also levels a tilted scan (unless you straightened it yourself): a crop of a
        tilted frame would keep slivers of the light around the film in its corners.
        `automatic`: the frame was just loaded, so the levelling is part of its automatic conversion."""
        if abs(p.settings.fine_rotation) > 0.05:
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
    # NegPy finds the film border on every render unless the rectangle it found is handed back
    # with the key of the framing it was found on. Found anew per render, the quick draft (while a
    # slider moves) and the full render each cropped a little differently, and the picture
    # twitched; the crop tool also had nothing to start from. A turn, a flip or straightening
    # changes the key: NegPy then looks again by itself until the new rectangle arrives.
    def _ensure_auto_crop(self, p: Photo):
        st = p.settings
        if not st.auto_crop or st.crop or p in self._levelling:
            return
        key = engine.auto_crop_key(st)
        if (st.auto_rect and st.auto_key == key) or self._autocrop_asked.get(p) == key:
            return
        self._autocrop_asked[p] = key
        snap = st.copy()
        self.pool.start(Job(lambda: engine.find_auto_crop(p.path, snap),
                            lambda r: self.signals.autocrop.emit(p, key, r)), URGENT)

    def _auto_crop_rect(self, p: Photo) -> QRectF | None:
        """The frame's auto crop as a crop-tool rectangle, when it is known for its framing now."""
        st = p.settings
        if st.auto_crop and not st.crop and st.auto_rect and st.auto_key == engine.auto_crop_key(st):
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
        st.auto_rect, st.auto_key = rect, found_key
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
        st = self.current.settings
        st.rotation = (st.rotation + quarter_turns) % 4

        def turn(r):
            if r is None:
                return None
            x0, y0, x1, y1 = r
            return (y0, 1 - x1, y1, 1 - x0) if quarter_turns > 0 else (1 - y1, x0, 1 - y0, x1)
        st.crop, st.analysis_rect = turn(st.crop), turn(st.analysis_rect)
        self._transform_shown(QTransform().rotate(-90 * quarter_turns))
        self.edited()

    def mirror(self):
        if not self.current:
            return
        self.preview.reset_zoom()
        st = self.current.settings
        st.mirror = not st.mirror
        flip = lambda r: None if r is None else (1 - r[2], r[1], 1 - r[0], r[3])
        st.crop, st.analysis_rect = flip(st.crop), flip(st.analysis_rect)
        self._transform_shown(QTransform().scale(-1, 1))
        self.edited()

    def reset_current(self):
        if self.current:
            old = self.current.settings
            new = engine.Settings(flat_field=old.flat_field, roll_floors=old.roll_floors,
                                  roll_ceils=old.roll_ceils, roll_axis=old.roll_axis)
            engine.settings_for_stock(new, self.current.group.stock if self.current.group else None)
            new.auto_crop = pref("auto_crop_new")
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

    def on_zoom(self, z: float):
        pv = self.preview
        over = z > 1.0
        if self.title.property("over") != over:          # a dark plate under the name, over the photo
            self.title.setProperty("over", over)
            self.title.style().unpolish(self.title)
            self.title.style().polish(self.title)
        self._update_zoom_100()
        if z <= 1.0:
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
        if p is not self.current or pv.zoom <= 1.0:
            pv.loading = False
            return
        if key != self._settings_key(p):                  # edited meanwhile: render again
            self._request_hires()
            return
        pv.loading = False
        if not isinstance(result, str):
            was_100 = abs(pv.zoom - pv.zoom_100) < 0.02 if pv.zoom_100 else False
            pv.hires, self._hires_key = to_pixmap(result), key
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
                self.toggle_pick_wb()
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
                                         f"{self.key_for('zoom') or 'a click'} or Cmd+scroll · {self.key_for('heal') or 'the button'} or Esc when done")
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
        self.statusBar().showMessage(f"{len(st.heal_spots)} spot(s) healed on this frame · Cmd+Z removes the last")

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
                self.toggle_pick_wb()
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
                                         f"Cmd+Z undoes the last · {self.key_for('heal_area') or 'the button'} or Esc when done")
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
        """W: the next click on the photo picks a spot that should be neutral."""
        if not self.current:
            return
        on = not self.preview.picking
        if on and self.preview.area_mode:
            self.toggle_heal_area()
        if on and self.preview.crop_mode:
            self.toggle_crop(apply=False)
        self.preview.picking = on
        self.preview.setCursor(Qt.CrossCursor)
        if not on:
            self.preview._update_cursor()
        self.pick_btn.setText("Click a Spot…" if on else "Pick Grey")
        self._mark(self.pick_btn, on)
        self.statusBar().showMessage("Click something that should be grey or white." if on else "", 0 if on else 1)

    def on_picked(self, pos: QPointF):
        self.toggle_pick_wb()                     # one click per pick
        if not self.current:
            return
        p, st, pt = self.current, self.current.settings.copy(), (pos.x(), pos.y())
        self.statusBar().showMessage("Balancing on the spot you clicked…")
        self.pool.start(Job(lambda: engine.auto_wb(p.path, st, point=pt), lambda r: self.signals.wb.emit(p, r)), URGENT)

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
            for p in self.selected_others():      # the selected frames too, like the sliders
                p.settings.cyan = p.settings.magenta = p.settings.yellow = 1.0
                self._commit_history(p)
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
    CROP_FORMATS = {"Free": None, "3:2": 3 / 2, "4:3": 4 / 3, "5:4": 5 / 4, "6:7": 7 / 6, "1:1": 1.0, "XPan": 65 / 24}
    CROP_FORMATS_TIPS = {"Free": "Any shape", "3:2": "35 mm (24 × 36) and 6 × 9", "4:3": "6 × 4.5 and half frame",
                         "5:4": "4 × 5 and 8 × 10 sheet film", "6:7": "6 × 7 medium format", "1:1": "6 × 6 square",
                         "XPan": "Panoramic 24 × 65"}

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

    def crop_reset(self):
        if self.preview.crop_mode:
            self.preview.crop = self.preview.format_area = QRectF(0, 0, 1, 1)   # the border included
            self.set_crop_format("Free")

    def _tool_lock(self, on: bool, keys_: tuple = (), keep: QPushButton | None = None):
        """While cropping, healing or cleaning an area, only the photo and that tool work: the rest
        of the window steps back (dimmed, and deaf to the mouse and the keys) until you are done.
        `keys_`: the actions that still answer; `keep`: the tool's own button under the photo."""
        dimmed = [self.left, self.right_stack, self.grid, self.tab_plate, self.export_btn, self.title]
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
            self.crop_btn.setToolTip("Pull the corners of the frame, drag inside to move it; Return applies, Esc cancels")
            self._update_toolbars()
            self.statusBar().showMessage("Crop: pull a corner · drag inside to move · pick a film format below · Return applies · Esc cancels")
            self.request_preview()
            return
        pv.crop_mode = False                      # finish editing
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
        size = PREVIEW_DRAFT if draft else PREVIEW_SIZE
        fn = lambda: engine.preview(p.path, st, size, uncropped=uncropped, keep_source=True)
        self._request_angle[rid] = st.fine_rotation
        self._request_draft[rid] = draft
        self.preview_pool.start(Job(fn, lambda r: self.signals.preview.emit(p, r, rid)))

    def on_preview(self, p: Photo, result, request_id: int):
        self.preview_busy = False
        if p is self.current and request_id == self.preview_request:
            if isinstance(result, str):
                self.preview.set_image(None)
                self.histogram.set_image(None)
                self.preview.setText(f"Could not convert {p.path.name}\n\n{result}")
            else:
                self.preview.setText("")
                self.preview.set_image(to_pixmap(result), draft=self._request_draft.get(request_id, False))
                self.histogram.set_image(result)
                # this render already contains its rotation; show only what changed since
                self._rendered_angle = self._request_angle.get(request_id, 0.0)
                self.preview.angle = p.settings.fine_rotation - self._rendered_angle
                if not self.preview.crop_mode and not self._show_before:
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
        self._request_angle.pop(request_id, None)
        self._request_draft.pop(request_id, None)
        if self.preview_pending:
            self.preview_pending = False
            full, self._pending_full = self._pending_full, False
            self.request_preview(draft=not full)

    def set_histogram_mode(self, mode: str):
        self.histogram.set_mode(mode)
        self.hist_mode_btn.set_mode(mode)           # from code: the dots follow (from a click they are there)
        set_pref("histogram", mode)

    # ----- Export ----------------------------------------------------------
    def open_export(self):
        if not self.all_photos():
            self.statusBar().showMessage("Nothing to export yet.")
            return
        sel = [self.photos[self.grid.row(i)] for i in self.grid.selectedItems()]
        extra = dict(self.export_extra)
        if not extra.get("dest"):
            first = self.all_photos()[0]
            extra["dest"] = str(first.path.parent / "converted")
        if self.current:
            extra["sample"] = self._tokens(self.current)
        dlg = ExportDialog(self, self.export_opts, extra, has_selection=len(sel) > 1,
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
        self._export(photos)

    def _tokens(self, p: Photo) -> dict:
        g = p.group
        roll = g and g.kind == "roll"
        return {"name": p.path.stem,
                "roll": g.name if roll else "",
                "film": (g.stock or {}).get("displayName", "") if roll else "",
                "frame": f"{g.photos.index(p) + 1:02d}" if g else "",
                "camera": g.camera if roll else ""}

    def _export(self, photos: list[Photo]):
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
            engine.new_export_batch()
        self._export_dir = str(out_dir)
        self.statusBar().showMessage(f"Exporting {self._export_total - self._export_left} of {self._export_total}…")
        for p in photos:
            st = p.settings.copy()
            p._exporting = edit_key(p.settings)
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


# The look: a grey window with black, borderless sections; neon light blue for anything chosen.
# Change a colour here and it changes everywhere (the stylesheet below and the painted widgets).
COLORS = {
    "bg": "#141518",          # window (under a dot grid, see Backdrop)
    "dots": "#3a3c43",        # the background grid (dots or lines)
    "panel": "#000000",       # sections: black, darker than the window
    "field": "#17181b",       # text fields inside sections (lighter than the black around them)
    "button": "#34353a",      # buttons: lighter than the section they sit on
    "button_hover": "#3f4046",
    "button_down": "#4a4b52",
    "text": "#e8e6e3",
    "muted": "#9a9aa0",
    "faint": "#5f6066",
    "soft": "rgba(232, 230, 227, 72%)",    # text, second level: labels, quiet buttons
    "dim": "rgba(232, 230, 227, 50%)",     # third level: hints, what a field would hold
    "ghost": "rgba(232, 230, 227, 34%)",   # captions and untouched values
    "warn": "#ffb454",        # a quiet warning (the scan itself is too dark / clipped)
    "line": "#2c2d32",        # hairlines inside sections
    "accent": "#2ee6ff",      # neon light blue: selection, focus, the main action
    "accent_hover": "#7af0ff",
    "accent2": "#b08cff",     # violet: the second accent, where blue already means something
                              # (an adjusted value, the mouse over something that is on, the field being typed in)
    "select": "#e8e6e3",      # anything chosen or pressed: a white plate...
    "on_select": "#000000",   # ...with its text reversed to black
    "on_select_muted": "#55575d",
    "raise": "#26272c",       # a quiet lift where a white plate would hide coloured text
    "on_accent": "#00181c",
}
ACCENT = COLORS["accent"]
ACCENT2 = COLORS["accent2"]
MONO = "Menlo"               # replaced by SF Mono at start-up when the Mac has it

# Type scale (px): one size per role; names and what is chosen are semibold, the rest regular.
#   title 20 (light)  ·  card title 15  ·  body 13  ·  small 12  ·  caption 10 (spaced capitals)
STYLE_TEMPLATE = """
QWidget { background: transparent; color: @text; font-size: 13px; }
QMainWindow, QDialog { background: @bg; }
QToolTip { background: @panel; color: @text; border: 1px solid @line; border-radius: 6px; padding: 5px 8px; font-size: 12px; }
QFrame#panel { background: @panel; border: none; border-radius: 9px; }

QLabel#header { color: @ghost; font-size: 10px; font-weight: 600; letter-spacing: 1.6px; }
QLabel#hint { color: @dim; font-size: 12px; padding-top: 4px; }
QLabel#welcome { font-size: 22px; font-weight: 300; color: @text; padding-bottom: 22px; letter-spacing: 0.3px; }
QLabel#pageTitle { font-size: 17px; font-weight: 600; color: @text; }
QLabel#title { color: @muted; font-size: 12px; padding: 3px 10px; }
QLabel#title[over="true"] { background: rgba(0, 0, 0, 170); color: @text; }
QLabel#faderTitle { color: @text; font-size: 12px; font-weight: 600; }
QLabel#rowLabel { color: @soft; font-size: 12px; font-weight: 500; }
QLabel#key { font-family: "@mono"; font-size: 12px; color: @accent; }
QLabel#faderValue { font-family: "@mono"; font-size: 11px; color: @ghost; }
QLabel#faderValue[edited="true"] { color: @accent2; font-weight: 600; }
QFrame#card { background: @panel; border: none; border-radius: 12px; }
QLabel#cardTitle { font-size: 16px; font-weight: 500; color: @text; border-bottom: 2px solid @accent; padding-bottom: 5px; }
QLabel#cardText { font-size: 12px; color: @dim; }
QLabel#exifTitle { font-size: 15px; font-weight: 600; color: @text; }
QToolButton#presetMenu { background: @bg; color: @muted; font-size: 13px; padding: 0; }
QToolButton#presetMenu:hover { color: @accent; }
QToolButton#presetMenu::menu-indicator { image: none; }
QMenu { background: @panel; color: @text; border: 1px solid @line; border-radius: 8px; padding: 5px; }
QMenu::item { padding: 6px 14px; border-radius: 5px; }
QMenu::item:selected { background: @select; color: @on_select; }
QMenu::item:disabled { color: @faint; }
QMenu::separator { height: 1px; background: @line; margin: 4px 6px; }

QPushButton { background: @panel; color: @text; border: none; border-radius: 6px; padding: 7px 14px; font-size: 12px; font-weight: 500; }
QFrame#panel QPushButton { background: @bg; }
QPushButton:hover, QFrame#panel QPushButton:hover { color: @accent; }
QPushButton:pressed, QFrame#panel QPushButton:pressed { background: @select; color: @on_select; }
QPushButton:disabled, QFrame#panel QPushButton:disabled { color: @faint; }
QPushButton#square { min-width: 36px; }
QFrame#panel QPushButton[on="true"] { color: @accent; }
QPushButton#tile, QFrame#panel QPushButton#tile { background: @bg; padding: 0; }
QPushButton#tile:pressed, QFrame#panel QPushButton#tile:pressed { background: @select; }
QPushButton#toggle, QFrame#panel QPushButton#toggle { background: @bg; color: @muted; padding: 6px 14px; min-width: 34px; }
QPushButton#toggle:hover, QFrame#panel QPushButton#toggle:hover { color: @accent; }
QPushButton#toggle:pressed, QFrame#panel QPushButton#toggle:pressed { color: @on_select; }
QPushButton#toggle:checked, QFrame#panel QPushButton#toggle:checked,
QPushButton#toggle[on="true"], QFrame#panel QPushButton#toggle[on="true"] { color: @accent; background: @raise; font-weight: 600; }
QPushButton#toggle[go="true"], QFrame#panel QPushButton#toggle[go="true"] { color: @accent2; background: @raise; font-weight: 600; }
QPushButton#toggle[go="true"]:hover, QFrame#panel QPushButton#toggle[go="true"]:hover { color: @text; }
QPushButton#toggle:checked:hover, QFrame#panel QPushButton#toggle:checked:hover,
QPushButton#toggle[on="true"]:hover, QFrame#panel QPushButton#toggle[on="true"]:hover { color: @accent2; }
QPushButton#bar { background: @panel; color: @muted; padding: 8px 18px; font-size: 12px; letter-spacing: 0.2px; }
QPushButton#bar:hover { color: @accent; }
QPushButton#bar:pressed { background: @select; color: @on_select; }
QPushButton#bar[on="true"], QPushButton#bar:checked { color: @accent; background: @raise; font-weight: 600; }
QPushButton#bar[on="true"]:hover, QPushButton#bar:checked:hover { color: @accent2; }
QPushButton#barGo { background: @raise; color: @accent2; padding: 8px 22px; font-size: 12px; font-weight: 600;
                    letter-spacing: 0.2px; }
QPushButton#barGo:hover { color: @text; }
QPushButton#barGo:pressed { background: @select; color: @on_select; }
QFrame#tabPlate { background: @panel; border-radius: 9px; }
QPushButton#primary, QFrame#panel QPushButton#primary { background: @accent; color: @on_accent; font-size: 13px;
                                                         font-weight: 700; padding: 9px 16px; }
QPushButton#primary:hover, QFrame#panel QPushButton#primary:hover { background: @accent_hover; color: @on_accent; }
QPushButton#primary:pressed, QFrame#panel QPushButton#primary:pressed { background: @accent; color: @on_accent; }
QPushButton#primary:disabled, QFrame#panel QPushButton#primary:disabled { background: @bg; color: @faint; }
QPushButton#export { background: @accent; color: @on_accent; font-size: 13px; font-weight: 700; padding: 0; border-radius: 9px;
                     letter-spacing: 0.3px; }
QPushButton#export:hover { background: @accent_hover; color: @on_accent; }
QPushButton#export:disabled { background: @panel; color: @faint; }
QPushButton#tab { background: transparent; border: none; border-radius: 6px;
                  color: @muted; padding: 6px 16px; font-size: 12px; letter-spacing: 0.3px; }
QPushButton#tab:checked { background: @raise; color: @text; font-weight: 600; }
QPushButton#tab:hover { color: @text; }
QPushButton#tab:checked:hover { color: @accent2; }
QToolButton { border: none; border-radius: 0; font-size: 14px; }
QPushButton#fold, QFrame#panel QPushButton#fold { background: transparent; color: @muted; font-size: 10px; font-weight: 600;
                                                  letter-spacing: 1.6px; text-align: left; padding: 4px 0; }
QPushButton#fold:hover, QFrame#panel QPushButton#fold:hover,
QPushButton#fold:checked, QFrame#panel QPushButton#fold:checked { background: transparent; color: @text; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: @field; border: 1px solid transparent;
                                                  border-radius: 6px; padding: 4px 8px;
                                                  font-size: 12px; selection-background-color: @accent;
                                                  selection-color: @on_accent; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateEdit:focus { background: #202126;
                                                                                         border-color: @accent2; }
QDoubleSpinBox#straightenAngle { padding: 4px 22px 4px 4px; }
QDoubleSpinBox#straightenAngle QLineEdit { background: transparent; border: none; padding: 0; }
QDoubleSpinBox#straightenAngle::up-button, QDoubleSpinBox#straightenAngle::down-button {
    subcontrol-origin: padding; width: 16px; border: none; background: transparent; margin-right: 2px; }
QDoubleSpinBox#straightenAngle::up-button { subcontrol-position: top right; border-top-right-radius: 4px; }
QDoubleSpinBox#straightenAngle::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 4px; }
QDoubleSpinBox#straightenAngle::up-button:hover, QDoubleSpinBox#straightenAngle::down-button:hover { background: @raise; }
QDoubleSpinBox#straightenAngle::up-button:pressed, QDoubleSpinBox#straightenAngle::down-button:pressed { background: @select; }
QDoubleSpinBox#straightenAngle::up-arrow, QDoubleSpinBox#straightenAngle::down-arrow { width: 7px; height: 5px; }
QDoubleSpinBox#straightenAngle::up-arrow { image: url("@icons/angle-up.svg"); }
QDoubleSpinBox#straightenAngle::down-arrow { image: url("@icons/angle-down.svg"); }
QDoubleSpinBox#straightenAngle::up-arrow:disabled, QDoubleSpinBox#straightenAngle::down-arrow:disabled { width: 0; height: 0; }
QDateEdit { background: @field; border: 1px solid transparent; border-radius: 6px;
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
QComboBox QAbstractItemView { background: @panel; border: 1px solid @line; selection-background-color: @select;
                              selection-color: @on_select; }
QCheckBox { spacing: 8px; font-size: 12px; color: @soft; }
QCheckBox::indicator { width: 12px; height: 12px; border-radius: 4px; background: @bg; border: 1px solid @faint; }
QCheckBox::indicator:hover { border-color: @text; }
QCheckBox::indicator:checked { background: @accent; border-color: @accent; }
QCheckBox::indicator:checked:hover { background: @accent2; border-color: @accent2; }

QFrame#rollCard { background: @panel; border: none; border-radius: 9px; }
QFrame#rollHead { background: transparent; }
QLabel#rollTitle { font-size: 13px; font-weight: 600; color: @text; }
QLabel#rollSub { font-size: 11px; color: @dim; }
QListWidget#groups { background: @panel; border: none; padding: 2px; font-size: 12px; }
QListWidget#groups::item { padding: 7px 8px; }
QListWidget#groups::item:selected { background: @select; color: @on_select; }
QListWidget#grid { background: @panel; border: none; border-radius: 9px; padding: 4px 8px;
                   font-size: 10px; font-family: "@mono"; color: @muted; }
QListWidget#grid::item, QListWidget#grid::item:selected { border: none; background: transparent; }

QSlider::groove:horizontal { height: 3px; background: @button; border-radius: 1px; }
QSlider::sub-page:horizontal { background: @button; }
QSlider::handle:horizontal { background: @text; width: 12px; margin: -5px 0; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: @accent; }
QSlider::handle:horizontal:pressed { background: @accent2; }
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


def make_style(mono: str = MONO) -> str:
    style = STYLE_TEMPLATE.replace("@mono", mono).replace("@icons", (Path(__file__).parent / "icons").as_posix())
    for name in sorted(COLORS, key=len, reverse=True):     # accent_hover before accent
        style = style.replace("@" + name, COLORS[name])
    return style


STYLE = make_style()


def self_test(path: str) -> int:
    """`DUMBCONVERT_SELFTEST=1 DumbConvert <scan>`: convert one scan without a window and report.
    Checks that a packaged app has everything it needs (NegPy, its Rust core, data files)."""
    import time
    t = time.time()
    img = engine.preview(Path(path), engine.Settings(auto_crop=True), 800)
    stocks = len(engine.film_stocks())
    print(f"SELFTEST OK: {img.shape[1]}x{img.shape[0]} preview, {stocks} film stocks, {time.time() - t:.1f}s", flush=True)
    import tempfile
    meta = engine.metadata_from_exif({"capture_date": "1997-09-30", "camera_model": "Test"})
    with tempfile.TemporaryDirectory() as d:
        for fmt in ("jpg", "heic", "tif", "dng"):
            out = engine.export(Path(path), Path(d), engine.Settings(auto_crop=True),
                                engine.ExportOptions(fmt=fmt, size="long_edge", long_edge=800), meta)
            print(f"SELFTEST export {fmt}: {out.stat().st_size // 1024} KB", flush=True)
    return 0


def _log_without_console():
    """Started without a console (Windows runs the app with pythonw), errors had nowhere to go and
    were lost. They go to dumbconvert.log next to the recovery file instead, so a tester can send it.
    The previous run's log is kept as dumbconvert.old.log."""
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
    # Memory: by default macOS keeps freed large blocks cached in the app. A converted scan frees
    # gigabytes of them, so after a few exports the app sat on 6 GB+ of memory it was not using.
    # MallocSpaceEfficient=1 hands them back at once (exports also got faster). The setting is
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
    mono = next((f for f in ("SF Mono", "Menlo") if f in QFontDatabase.families()),
                QFontDatabase.systemFont(QFontDatabase.FixedFont).family())
    app.setStyleSheet(make_style(mono))
    win = MainWindow()
    win.show()
    left_behind = win.recovery_found()               # only there if the last run did not quit normally
    if left_behind:
        n = sum(len(g.get("photos", [])) for g in left_behind.get("groups", []))
        box = QMessageBox(win)
        box.setIcon(QMessageBox.NoIcon)
        box.setText(f"{APP_NAME} did not quit normally last time.")
        box.setInformativeText(f"Restore the {n} frame{'s' if n != 1 else ''} you were working on, with their adjustments?")
        restore = box.addButton("Restore", QMessageBox.AcceptRole)
        box.addButton("Start Fresh", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is restore:
            win.restore_session(left_behind)
    win._recovery_timer.start()
    if len(sys.argv) > 1:          # allow: python -m simpleneg file1.tif folder/ ...
        win.add_rolls([Path(a) for a in sys.argv[1:]])
    sys.exit(app.exec())
