"""Key bindings: which key does what, a drawn keyboard to change them, and saving them.

How it fits together:
  * ACTIONS lists everything a key can do. The functions behind them live in
    MainWindow (app.py), which looks them up by id.
  * A binding is  key name -> action id, e.g. {"Q": "density_up"}. Key names are
    Qt's own (QKeySequence(...).toString()): "Q", "Space", "Left", "F1", ",".
    One key does one thing; one action can have several keys.
  * Bindings are saved with QSettings (on a Mac: ~/Library/Preferences/), so they
    survive restarts.
  * BindingsPage is the "Keys" tab: click a key on the drawing (or press it)
    and pick what it should do.
"""

import json
import sys
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

# The command key and the Return key as this system names them, for anything shown to the user.
MOD = "Cmd" if sys.platform == "darwin" else "Ctrl"
RETURN = "Return" if sys.platform == "darwin" else "Enter"


@dataclass(frozen=True)
class Action:
    id: str
    label: str          # in the action list
    short: str          # on the key cap (keep it to ~6 characters)
    group: str          # colour on the drawing
    repeat: bool = False    # holding the key repeats it
    shift: str = ""     # what Shift + the key does, if anything
    hold: bool = False  # active while the key is held down (watched separately)
    adjust: str = ""    # hold the key and press up / down to change this Settings field


ACTIONS = [
    Action("accept", "Next frame, then the next roll", "Next ⏎", "frames"),
    Action("next", "Next frame", "Next ▸", "frames", repeat=True),
    Action("prev", "Previous frame", "◂ Prev", "frames", repeat=True),
    Action("next_roll", "Next roll / group", "Roll ▾", "frames", repeat=True),
    Action("prev_roll", "Previous roll / group", "Roll ▴", "frames", repeat=True),
    Action("first", "First frame", "First", "frames"),
    Action("last", "Last frame", "Last", "frames"),
    Action("adjust_density", "Density  (↑ darker, ↓ lighter)", "D ↕", "density",
           hold=True, adjust="density"),
    Action("adjust_cyan", "Cyan ↔ Red  (↑ cyan, ↓ red)", "C ↕", "cyan",
           hold=True, adjust="cyan"),
    Action("adjust_magenta", "Magenta ↔ Green  (↑ magenta, ↓ green)", "M ↕", "magenta",
           hold=True, adjust="magenta"),
    Action("adjust_yellow", "Yellow ↔ Blue  (↑ yellow, ↓ blue)", "Y ↕", "yellow",
           hold=True, adjust="yellow"),
    Action("adjust_contrast", "Contrast  (↑ more, ↓ less)", "Con ↕", "tone", hold=True, adjust="contrast"),
    Action("adjust_saturation", "Saturation  (↑ more, ↓ less)", "Sat ↕", "tone", hold=True, adjust="saturation"),
    Action("density_up", "Density + (darker)", "D +", "density", True, "4× step"),
    Action("density_down", "Density − (lighter)", "D −", "density", True, "4× step"),
    Action("cyan_up", "Cyan +", "C +", "cyan", True, "4× step"),
    Action("cyan_down", "Cyan −", "C −", "cyan", True, "4× step"),
    Action("magenta_up", "Magenta +", "M +", "magenta", True, "4× step"),
    Action("magenta_down", "Magenta −", "M −", "magenta", True, "4× step"),
    Action("yellow_up", "Yellow +", "Y +", "yellow", True, "4× step"),
    Action("yellow_down", "Yellow −", "Y −", "yellow", True, "4× step"),
    Action("contrast_up", "Contrast +", "Con +", "tone", True, "4× step"),
    Action("contrast_down", "Contrast −", "Con −", "tone", True, "4× step"),
    Action("saturation_up", "Saturation +", "Sat +", "tone", True, "4× step"),
    Action("saturation_down", "Saturation −", "Sat −", "tone", True, "4× step"),
    Action("auto_wb", "Auto WB", "AWB", "tone"),
    Action("pick_wb", "White balance: click a neutral spot", "Pick", "tone"),
    Action("reset", "Reset this frame", "Reset", "tone"),
    Action("rotate_left", "Rotate left", "↺ 90°", "geometry"),
    Action("rotate_right", "Rotate right", "↻ 90°", "geometry"),
    Action("mirror", "Mirror", "Mirror", "geometry"),
    Action("tilt_left", "Straighten 0.1° counter-clockwise", "↺ 0.1°", "geometry", True, "1°"),
    Action("tilt_right", "Straighten 0.1° clockwise", "↻ 0.1°", "geometry", True, "1°"),
    Action("auto_straighten", "Auto straighten", "Level", "geometry"),
    Action("crop", "Edit crop / apply crop", "Crop", "geometry"),
    Action("heal", "Heal dust: click specks (H again to finish)", "Heal", "geometry"),
    Action("heal_area", "Clean area: draw around dust or scratches (A again to finish)", "Area", "geometry"),
    Action("zoom", f"Zoom: fit ↔ 100% (also click, pinch or {MOD}+scroll)", "Zoom", "view"),
    Action("fullscreen", "Full screen: only the photo (again, or Esc, to leave)", "Full", "view"),
    Action("auto_crop", "Auto crop on / off", "A.Crop", "geometry"),
    Action("auto_crop_refresh", "Auto crop: find the picture edges again", "Re-crop", "geometry"),
    Action("before", "Before (hold): the automatic conversion", "Before", "view", hold=True),
    Action("border", "White border on / off", "Border", "view"),
    Action("original", "Original scan (before conversion) on / off", "Orig.", "view"),
    Action("copy", "Copy settings", "Copy", "other"),
    Action("paste", "Paste settings to selected frames", "Paste", "other"),
    Action("sync_roll", "Sync Roll: send this frame's edits to the whole group", "Sync", "other"),
    Action("undo", "Undo", "Undo", "other", repeat=True),
    Action("redo", "Redo", "Redo", "other", repeat=True),
    Action("export", "Export…", "Export", "other"),
    Action("contact_sheet", "Contact sheet of this roll…", "Sheet", "other"),
    Action("keys", "Open the Keys tab", "Keys", "other"),
]
ACTION_BY_ID = {a.id: a for a in ACTIONS}

GROUP_NAMES = {"frames": "Frames", "density": "Density", "cyan": "Colour", "magenta": "Colour",
               "yellow": "Colour", "tone": "Tone", "geometry": "Geometry", "view": "View", "other": "Other"}
GROUP_COLORS = {"frames": "#1b8796", "density": "#77797f", "cyan": "#1c8fc0", "magenta": "#c4196a",
                "yellow": "#d9c400", "tone": "#5b5e66", "geometry": "#2f8f5b", "view": "#6b7d8f",
                "other": "#4a4c52"}

# The six corrections sit under the left hand and never make it move: hold one of them and press
# up or down with the right hand (Shift for a 4x step), like a lab scanner's colour keys.
#
#     Q  W  E        Density · Contrast · Saturation        (hold, then up / down)
#     A  S  D        Cyan · Magenta · Yellow                (hold, then up / down)
#     Z  X  C  V  B  N  M      zoom · crop · auto crop · level · before (hold) · pick neutral · mirror
#     R auto crop again · Y sync roll · O white border · T original scan · F full screen · J rotate left · K auto WB · L rotate right · U heal · I clean area · P sheet
#
# The one-step actions (cyan_up, ...) still exist; they are just not on a key by default.
DEFAULT_BINDINGS = {
    "Right": "next", "Left": "prev", "Space": "accept", "Return": "accept",
    "Down": "next_roll", "Up": "prev_roll", "Home": "first", "End": "last",
    "Q": "adjust_density", "W": "adjust_contrast", "E": "adjust_saturation",
    "A": "adjust_cyan", "S": "adjust_magenta", "D": "adjust_yellow",
    "Z": "zoom", "X": "crop", "C": "auto_crop", "V": "auto_straighten", "B": "before", "N": "pick_wb", "M": "mirror",
    "R": "auto_crop_refresh", "Y": "sync_roll", "O": "border", "T": "original", "F": "fullscreen", "J": "rotate_left", "K": "auto_wb", "L": "rotate_right",
    "U": "heal", "I": "heal_area", "P": "contact_sheet",
    ",": "tilt_left", ".": "tilt_right", "0": "auto_wb", "Backspace": "reset", "F1": "keys",
}

# Always the same, so they are not on the drawing's menu.
FIXED = [(f"{MOD}+Z / {MOD}+Shift+Z", "Undo / redo"), (f"{MOD}+C / {MOD}+V", "Copy settings / paste to selected"),
         (f"{MOD}+A", "Select every frame (then any correction changes them all)"),
         (f"{MOD}+E", "Export…"), (f"{MOD}+S", "Save this session (offered back at the next start)"),
         ("Esc", "Leave a text field, cancel the crop, leave full screen")]

# The drawing: rows of (label on the cap, key name or None if it can't be bound, width in key units)
_F = [("esc", None, 1.5)] + [(f"F{i}", f"F{i}", 1.125) for i in range(1, 13)]
KEYBOARD = [
    _F,
    [(c, c, 1) for c in "`1234567890-="] + [("delete", "Backspace", 2)],
    [("tab", None, 1.5)] + [(c, c, 1) for c in "QWERTYUIOP[]"] + [("\\", "\\", 1.5)],
    [("caps", None, 1.75)] + [(c, c, 1) for c in "ASDFGHJKL;'"] + [("return", "Return", 2.25)],
    [("shift", None, 2.25)] + [(c, c, 1) for c in "ZXCVBNM,./"] + [("shift", None, 2.75)],
    [("fn", None, 1), ("ctrl", None, 1), ("opt", None, 1), ("cmd", None, 1.25), ("", "Space", 5.5),
     ("cmd", None, 1.25), ("opt", None, 1), ("←", "Left", 1), ("↑↓", "UpDown", 1), ("→", "Right", 1)],
]
ON_DRAWING = {name for row in KEYBOARD for _, name, _ in row if name} | {"Up", "Down"}
MODIFIERS = {Qt.Key_Shift, Qt.Key_Control, Qt.Key_Meta, Qt.Key_Alt, Qt.Key_AltGr, Qt.Key_CapsLock,
             Qt.Key_Tab, Qt.Key_Escape, Qt.Key_unknown}


NEW_ACTIONS: set = {"auto_crop_refresh", "sync_roll", "border", "original"}   # actions added after v4 layouts were saved (see load)
SETTINGS_KEY = "key_bindings_v4"   # v4: the corrections on Q W E / A S D, held (older saved layouts are not reused)


def key_name(key: int) -> str | None:
    """Qt key code -> the name bindings use (None for modifiers and keys we keep fixed)."""
    if key in MODIFIERS or key == 0:
        return None
    if key == Qt.Key_Enter:            # the keypad's Enter counts as Return
        key = Qt.Key_Return
    return QKeySequence(key).toString() or None


def pretty(name: str) -> str:
    return {"Left": "←", "Right": "→", "Up": "↑", "Down": "↓", "Backspace": "delete",
            "Return": "return", "Space": "space"}.get(name, name)


def load() -> dict[str, str]:
    raw = QSettings("SimpleNeg", "SimpleNeg").value(SETTINGS_KEY, "")
    try:
        saved = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        saved = None
    if not isinstance(saved, dict):
        return dict(DEFAULT_BINDINGS)
    bindings = {k: v for k, v in saved.items() if v in ACTION_BY_ID}
    # An action added to the app after the layout was saved gets its default key, if that key is free
    for key, action in DEFAULT_BINDINGS.items():
        if action in NEW_ACTIONS and action not in bindings.values() and key not in bindings:
            bindings[key] = action
    return bindings


def save(bindings: dict[str, str]):
    QSettings("SimpleNeg", "SimpleNeg").setValue(SETTINGS_KEY, json.dumps(bindings))


class KeyboardView(QWidget):
    """Draws a Mac keyboard with each key's action on it. Click a key to select it."""

    clicked = Signal(str)

    def __init__(self):
        super().__init__()
        self.bindings: dict[str, str] = {}
        self.selected: str | None = None
        self.setMinimumSize(360, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._hover: str | None = None
        self._caps: list[tuple[QRectF, str | None, str]] = []   # (rect, key name, label), from the last paint

    def heightForWidth(self, w):
        return int(w * 0.4)

    def _layout(self):
        """Where every key cap goes, scaled to the widget's size."""
        unit = min(self.width() / 15.05, self.height() / 5.7)      # as large as the space allows
        x0 = (self.width() - unit * 15) / 2
        y, gap, caps = (self.height() - unit * 5.6) / 2, unit * 0.08, []
        for r, row in enumerate(KEYBOARD):
            h = unit * (0.6 if r == 0 else 1.0)
            x = x0
            for label, name, w in row:
                rect = QRectF(x + gap / 2, y + gap / 2, w * unit - gap, h - gap)
                if name == "UpDown":           # the half-height up/down pair
                    top, bottom = QRectF(rect), QRectF(rect)
                    top.setHeight(rect.height() / 2 - gap / 4)
                    bottom.setTop(rect.top() + rect.height() / 2 + gap / 4)
                    caps += [(top, "Up", "↑"), (bottom, "Down", "↓")]
                else:
                    caps.append((rect, name, label))
                x += w * unit
            y += h
        return caps, unit

    def paintEvent(self, e):
        """Black caps on the window. A key that does something carries its action in white and a
        thin line in its group's colour along the bottom; the rest only whisper their legend."""
        caps, unit = self._layout()
        self._caps = caps
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        small, big = QFont(), QFont()
        small.setPixelSize(max(8, int(unit * 0.17)))
        big.setPixelSize(max(9, int(unit * 0.22)))
        p.setPen(Qt.NoPen)
        for rect, name, label in caps:
            action = ACTION_BY_ID.get(self.bindings.get(name, "")) if name else None
            selected = bool(name) and name == self.selected
            hot = bool(name) and name == self._hover
            fill = "#e8e6e3" if selected else "#26272c" if hot else "#000000" if name else "#0b0c0e"
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(fill))
            p.drawRoundedRect(rect, 5, 5)
            if action:                             # the group's colour, as a line
                line = max(2.0, unit * 0.045)
                p.setBrush(QColor(GROUP_COLORS[action.group]).lighter(125))
                p.drawRoundedRect(QRectF(rect.left() + 6, rect.bottom() - line - 3, rect.width() - 12, line), line / 2, line / 2)
            p.setFont(small)
            p.setPen(QColor("#000000" if selected else "#6f7177" if name else "#3a3c43"))
            p.drawText(rect.adjusted(unit * 0.11, unit * 0.07, 0, 0), Qt.AlignLeft | Qt.AlignTop, label)
            if action and rect.height() > unit * 0.45:
                p.setFont(big)
                p.setPen(QColor("#000000" if selected else "#bdbdc1"))
                p.drawText(rect.adjusted(2, unit * 0.16, -2, 0), Qt.AlignCenter, action.short)
        p.end()

    def _key_at(self, pos) -> tuple[str | None, str]:
        for rect, name, label in self._caps:
            if rect.contains(pos):
                return name, label
        return None, ""

    def mousePressEvent(self, e):
        name, _ = self._key_at(e.position())
        if name:
            self.clicked.emit(name)

    def mouseMoveEvent(self, e):
        name, _ = self._key_at(e.position())
        a = ACTION_BY_ID.get(self.bindings.get(name, "")) if name else None
        self.setToolTip(f"{pretty(name)}: {a.label}" if a else (pretty(name) if name else ""))
        self.setCursor(Qt.PointingHandCursor if name else Qt.ArrowCursor)
        if name != self._hover:
            self._hover = name
            self.update()

    def leaveEvent(self, e):
        self._hover = None
        self.update()


class BindingsPage(QWidget):
    """The Keys tab: the keyboard drawing on the left, what the selected key does on the right."""

    changed = Signal(dict)

    def __init__(self, bindings: dict[str, str]):
        super().__init__()
        self.bindings = dict(bindings)
        self.setFocusPolicy(Qt.StrongFocus)

        self.keyboard = KeyboardView()
        self.keyboard.clicked.connect(self.select)
        title = QLabel("Keyboard")
        title.setObjectName("pageTitle")
        intro = QLabel("Click a key, or press it, then pick what it does. Saved as you go.")
        intro.setObjectName("rollSub")
        # what the coloured lines mean
        seen, chips = set(), []
        for a in ACTIONS:
            name = GROUP_NAMES[a.group]
            if name not in seen:
                seen.add(name)
                chips.append(f"<span style='color:{QColor(GROUP_COLORS[a.group]).lighter(125).name()}'>▬</span>&nbsp;&nbsp;{name}")
        groups = QLabel("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(chips))
        groups.setObjectName("rollSub")
        groups.setAlignment(Qt.AlignCenter)
        legend = QLabel("↕ keys: hold the key and press ↑ / ↓  (Shift for a 4× step)   ·   " +
                        "   ·   ".join(f"{k}  {v}" for k, v in FIXED))
        legend.setWordWrap(True)
        legend.setObjectName("rollSub")
        legend.setAlignment(Qt.AlignCenter)
        self.extra = QLabel()            # bindings on keys the drawing doesn't show (Home, End, non-US keys…)
        self.extra.setObjectName("rollSub")
        self.extra.setWordWrap(True)
        self.extra.setAlignment(Qt.AlignCenter)
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(title)
        left.addWidget(intro)
        left.addSpacing(8)
        left.addWidget(self.keyboard, 1)
        left.addSpacing(8)
        left.addWidget(groups)
        left.addSpacing(6)
        left.addWidget(legend)
        left.addWidget(self.extra)

        self.key_title = QLabel("No key selected")
        self.key_title.setObjectName("exifTitle")
        self.actions = QListWidget()
        self.actions.setObjectName("groups")
        self.actions.itemClicked.connect(self._assign)
        none = QListWidgetItem("Nothing")
        none.setData(Qt.UserRole, "")
        none.setForeground(QColor("#9a9aa0"))
        self.actions.addItem(none)
        group = None
        caps_font = QFont()
        caps_font.setPixelSize(10)
        caps_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.6)
        for a in ACTIONS:
            section = "Hold, then ↑ / ↓" if a.adjust else ("One step per press" if a.repeat and a.shift == "4× step"
                                                            else GROUP_NAMES[a.group])
            if section != group:
                group = section
                head = QListWidgetItem(group.upper())
                head.setFlags(Qt.NoItemFlags)
                head.setFont(caps_font)
                head.setForeground(QColor("#6f7177"))
                self.actions.addItem(head)
            item = QListWidgetItem(a.label + (f"   (Shift: {a.shift})" if a.shift else ""))
            item.setData(Qt.UserRole, a.id)
            self.actions.addItem(item)
        # Always live, so the list can be scrolled and read before a key is chosen (a disabled
        # list ignores the wheel). Without a key, a click on an action only asks for one.
        self.actions.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.actions.verticalScrollBar().setSingleStep(12)
        reset = QPushButton("Reset All to Defaults")
        reset.clicked.connect(self.reset_defaults)
        panel = QFrame()                 # the picker sits on a black panel, like the app's other sections
        panel.setObjectName("panel")
        right = QVBoxLayout(panel)
        right.setContentsMargins(14, 14, 14, 14)
        right.setSpacing(10)
        right.addWidget(self.key_title)
        right.addWidget(self.actions, 1)
        right.addWidget(reset)
        panel.setFixedWidth(300)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(22)
        lay.addLayout(left, 1)
        lay.addWidget(panel)
        self._refresh()

    def keyPressEvent(self, e):
        """Pressing a real key selects it, including keys that aren't on the drawing."""
        name = key_name(e.key())
        if name and not e.isAutoRepeat():
            self.select(name)
        else:
            super().keyPressEvent(e)

    def select(self, name: str):
        self.keyboard.selected = name
        current = self.bindings.get(name, "")
        for i in range(self.actions.count()):
            item = self.actions.item(i)
            if item.flags() & Qt.ItemIsSelectable and item.data(Qt.UserRole) == current:
                self.actions.setCurrentItem(item)
                self.actions.scrollToItem(item)
                break
        self._refresh()
        self.setFocus()          # keep listening for the next key press

    def _assign(self, item: QListWidgetItem):
        name = self.keyboard.selected
        if not item.flags() & Qt.ItemIsSelectable:
            return
        if not name:
            self.actions.clearSelection()
            self.key_title.setText("Choose a key first")
            return
        action = item.data(Qt.UserRole)
        self.actions.setCurrentItem(item)
        if action:
            self.bindings[name] = action
        else:
            self.bindings.pop(name, None)
        self._refresh()
        self.changed.emit(dict(self.bindings))
        self.setFocus()

    def reset_defaults(self):
        self.bindings = dict(DEFAULT_BINDINGS)
        self._refresh()
        if self.keyboard.selected:
            self.select(self.keyboard.selected)
        self.changed.emit(dict(self.bindings))

    def _refresh(self):
        self.keyboard.bindings = self.bindings
        self.keyboard.update()
        name = self.keyboard.selected
        if name:
            a = ACTION_BY_ID.get(self.bindings.get(name, ""))
            self.key_title.setText(f"{pretty(name)}  →  {a.short if a else 'nothing'}")
        others = sorted(k for k in self.bindings if k not in ON_DRAWING)
        self.extra.setText(("Keys not on the drawing: " + "  ·  ".join(
            f"<b>{pretty(k)}</b> {ACTION_BY_ID[self.bindings[k]].label}" for k in others)) if others else "")
