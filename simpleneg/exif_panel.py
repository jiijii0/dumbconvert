"""The Exif tab's right-hand panel, and the metadata fields it shares with the roll details.

How it fits together:
  * FIELDS lists the Exif tab's fields; the keys are what engine.metadata_from_exif reads.
  * A frame's value comes from the frame itself if set there, otherwise from its roll
    (the roll details on the left). MainWindow (app.py) resolves that and calls
    ExifPanel.show_values(); the panel displays, and reports edits through signals.
  * The field widgets are reused by the roll details:
      DateField      a date picker with a calendar (no invalid dates possible)
      LocationField  type a place, pick it from the list: its GPS coordinates are attached
      PresetField    camera / lens with presets (suggestions while typing, ▾ menu)
  * flash() briefly lights a field up after its value was saved.
"""

import json
import urllib.parse
import urllib.request

from PySide6.QtCore import (QDate, QEvent, QObject, QPoint, QPointF, QRunnable, QSettings, QStringListModel,
                            Qt, QThreadPool, QTimer, Signal)
from PySide6.QtGui import QAction, QColor, QIcon, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QCompleter, QDateEdit, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QPushButton, QVBoxLayout,
                               QWidget)

from . import engine

# (key, label, section, example shown when there is no value)
FIELDS = [
    ("capture_date", "Date", "Date & place", ""),
    ("gps", "Location", "Date & place", "Search a place, or paste coordinates"),
    ("camera_model", "Camera", "Camera & lens", "e.g. Leica M6"),
    ("lens_model", "Lens", "Camera & lens", "e.g. Summicron-M 50mm f/2"),
    ("film_stock_id", "Stock", "Film", ""),
]
KEYS = [f[0] for f in FIELDS]
EXAMPLES = {f[0]: f[3] for f in FIELDS}
MULTIPLE = "Multiple values"
FROM_ROLL = "  ·  from roll"
PRESET_KINDS = {"camera_model": "cameras", "lens_model": "lenses"}
PRESETS_KEY = "exif_presets_v2"     # {"cameras": ["Leica M6", ...], "lenses": [...]}

MUTED, ACCENT = "#85858a", "#7350c8"


def flash(widget: QWidget, ms: int = 900):
    """Light a field up for a moment: its value was saved."""
    def set_(on: bool):
        widget.setProperty("saved", on)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
    set_(True)
    QTimer.singleShot(ms, lambda: set_(False))


def _icon(draw) -> QIcon:
    pix = QPixmap(32, 32)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    draw(p)
    p.end()
    pix.setDevicePixelRatio(2)          # drawn at 2x: sharp on Retina, 16 px on screen
    return QIcon(pix)


def magnifier_icon() -> QIcon:
    def draw(p):
        p.setPen(QPen(QColor(MUTED), 2.6))
        p.drawEllipse(QPointF(13, 13), 7.5, 7.5)
        p.drawLine(QPointF(18.5, 18.5), QPointF(26, 26))
    return _icon(draw)


def chevron_icon() -> QIcon:
    def draw(p):
        p.setPen(QPen(QColor(MUTED), 2.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPolyline([QPointF(9, 13), QPointF(16, 20), QPointF(23, 13)])
    return _icon(draw)


def draw_chevron(widget: QWidget, p: QPainter = None):
    """The one dropdown mark used everywhere (combos, date, camera / lens), at the field's right edge."""
    own = p is None
    p = p or QPainter(widget)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(MUTED), 1.3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    x, y = widget.width() - 16, widget.height() / 2
    p.drawPolyline([QPointF(x - 3.5, y - 1.8), QPointF(x, y + 1.8), QPointF(x + 3.5, y - 1.8)])
    if own:
        p.end()


class ChevronCombo(QComboBox):
    """A dropdown with the same chevron as the date and camera / lens fields."""

    def paintEvent(self, e):
        super().paintEvent(e)
        draw_chevron(self)


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------
class DateField(QDateEdit):
    """A date picker: type it or pick it from the calendar. "No date" is the empty state, shown faint."""

    def _mark_empty(self, *_):
        empty = self.date() == self.NONE
        if self.property("empty") != empty:
            self.setProperty("empty", empty)
            self.style().unpolish(self)
            self.style().polish(self)

    NONE = QDate(1900, 1, 1)
    changed_iso = Signal(str)                 # "YYYY-MM-DD", or "" for no date

    def __init__(self):
        super().__init__()
        self.setCalendarPopup(True)
        self.setDisplayFormat("d MMM yyyy")
        self.setMinimumDate(self.NONE)
        self.setMaximumDate(QDate.currentDate())
        self.setSpecialValueText("No date")
        self.dateChanged.connect(self._mark_empty)
        self.setDate(self.NONE)
        self._mark_empty()
        self.setToolTip("When the film was shot. Delete clears it")
        self._loading = False
        # Saved at once: a picked date right away, a typed one when the typing ends (Return or
        # clicking elsewhere). A delayed save could land on the next frame if you moved on quickly.
        self.setKeyboardTracking(False)
        self.dateChanged.connect(lambda _: None if self._loading else self.changed_iso.emit(self.iso()))

    def iso(self) -> str:
        return "" if self.date() == self.NONE else self.date().toString("yyyy-MM-dd")

    def set_iso(self, text: str, empty_text: str = "No date", inherited: bool = False):
        self._loading = True
        self.setSpecialValueText(empty_text)
        t = (text or "")[:10]
        parts = [int(x) for x in t.split("-") if x.isdigit()] if t else []
        d = QDate(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1) if parts else self.NONE
        self.setDate(d if d.isValid() else self.NONE)
        self.setProperty("inherited", inherited)
        self.style().unpolish(self)
        self.style().polish(self)
        self._loading = False

    def mousePressEvent(self, e):
        """A click anywhere opens the calendar (typing still works with the keyboard)."""
        if self.date() == self.NONE:                     # open on this month, not on 1900 (after it opens)
            today = QDate.currentDate()
            QTimer.singleShot(0, lambda: self.calendarWidget().setCurrentPage(today.year(), today.month()))
        if e.button() == Qt.LeftButton:
            arrow = QPointF(self.width() - 10, self.height() / 2)    # where the popup button sits
            e = QMouseEvent(e.type(), arrow, self.mapToGlobal(arrow.toPoint()), e.button(), e.buttons(), e.modifiers())
        super().mousePressEvent(e)

    def paintEvent(self, e):
        super().paintEvent(e)
        draw_chevron(self)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.date() != self.NONE and \
                self.currentSection() == QDateEdit.NoSection:
            self.setDate(self.NONE)
            return
        super().keyPressEvent(e)


# ---------------------------------------------------------------------------
# Location: a place search with its own result list
# ---------------------------------------------------------------------------
# Photon: a free search over OpenStreetMap, made for as-you-type suggestions.
# Only the typed words are sent (no photos, no personal data); the app names itself generically.
SEARCH_URL = "https://photon.komoot.io/api/"


_place_cache: dict[str, list] = {}      # query -> results, for this run of the app


def search_places(query: str, limit: int = 8) -> list[tuple[str, str, float, float]]:
    """Places matching `query` -> [(name, "city, region, country", latitude, longitude)], English, no duplicates.

    Photon is a free, shared server, so the app asks it as little as it can: only after a pause
    in typing, never twice for the same words, and it says honestly who is asking (the
    User-Agent). Each person's requests come from their own internet address."""
    key = query.strip().lower()
    if key in _place_cache:
        return _place_cache[key]
    from . import __version__
    url = SEARCH_URL + "?" + urllib.parse.urlencode({"q": query, "limit": limit + 4, "lang": "en"})
    req = urllib.request.Request(url, headers={"User-Agent": f"DumbConvert/{__version__}"})
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.load(r)
    out, seen = [], set()
    for f in data.get("features", []):
        p = f.get("properties", {})
        lon, lat = f["geometry"]["coordinates"][:2]
        name = p.get("name") or p.get("city") or p.get("country") or ""
        where = ", ".join(dict.fromkeys(v for v in (p.get("city"), p.get("state"), p.get("country")) if v and v != name))
        if name and (name, where) not in seen:
            seen.add((name, where))
            out.append((name, where, float(lat), float(lon)))
    if len(_place_cache) > 300:
        _place_cache.clear()
    _place_cache[key] = out[:limit]
    return out[:limit]


class _SearchSignals(QObject):
    done = Signal(str, object)          # (query, results list, or None when the search failed)


class _SearchJob(QRunnable):
    def __init__(self, query: str, signals: _SearchSignals):
        super().__init__()
        self.query, self.signals = query, signals

    def run(self):
        try:
            results = search_places(self.query)
        except Exception:
            results = None
        self.signals.done.emit(self.query, results)


class _ResultList(QListWidget):
    """The suggestions under the field, drawn inside the main window (not a pop-up window, which
    can steal focus and flicker). It never takes the keyboard: typing stays in the field."""

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.NoFocus)
        self.hide()
        self.setObjectName("placeList")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


class LocationField(QLineEdit):
    """Type a place and pick it from the list (like the Weather app), or paste coordinates.

    `picked(label, lat, lon)` after choosing a place; `coords_typed(text)` after finishing
    with coordinates or an empty field.
    """

    picked = Signal(str, float, float)
    coords_typed = Signal(str)

    def __init__(self, placeholder: str = "Search a place, or paste coordinates"):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.addAction(magnifier_icon(), QLineEdit.LeadingPosition)
        self._signals = _SearchSignals()
        self._signals.done.connect(self._on_results)
        self._list = _ResultList()
        self._list.itemClicked.connect(self._choose_item)
        self._timer = QTimer(singleShot=True, interval=450)      # search once typing pauses
        self._timer.timeout.connect(self._search)
        self.textEdited.connect(lambda _: self._timer.start())
        self.editingFinished.connect(self._finished)

    # the list ------------------------------------------------------------------
    def _show_rows(self, rows: list[tuple[str, str, object]]):
        """rows: (title, subtitle, data); data None makes a grey, unselectable row."""
        lst = self._list
        lst.clear()
        for title, sub, data in rows:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, data)
            label = QLabel(f"<span style='color:{'#bdbdc1' if data else MUTED}'>{title}</span>"
                           + (f"<br><span style='color:{MUTED}; font-size:11px'>{sub}</span>" if sub else ""))
            label.setContentsMargins(10, 5, 10, 5)
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            if not data:
                item.setFlags(Qt.NoItemFlags)
            item.setSizeHint(label.sizeHint())
            lst.addItem(item)
            lst.setItemWidget(item, label)
        first = next((i for i in range(lst.count()) if lst.item(i).flags() & Qt.ItemIsSelectable), -1)
        lst.setCurrentRow(first)
        width = max(self.width(), 340)
        height = sum(lst.sizeHintForRow(i) for i in range(lst.count())) + 6
        win = self.window()
        if lst.parent() is not win:
            lst.setParent(win)                           # an overlay in the window, above everything
        pos = self.mapTo(win, QPoint(0, self.height() + 2))
        width = min(width, win.width() - pos.x() - 8)
        if pos.x() + width > win.width() - 8:            # keep it inside the window on the right
            pos.setX(max(8, win.width() - 8 - width))
        lst.setGeometry(pos.x(), pos.y(), width, min(height, 380, win.height() - pos.y() - 8))
        lst.show()
        lst.raise_()

    def hide_list(self):
        self._list.hide()

    # searching -----------------------------------------------------------------
    def _search(self):
        q = self.text().strip()
        if len(q) < 3 or engine.parse_gps(q) is not None:
            self.hide_list()
            return
        self._show_rows([("Searching…", "", None)])
        QThreadPool.globalInstance().start(_SearchJob(q, self._signals))

    def _on_results(self, query: str, results):
        if query != self.text().strip() or not (self.hasFocus() or self._list.isVisible()):
            return                                        # typing moved on, or the field was left
        if results is None:
            self._show_rows([("Place search is not available", "Offline? You can paste coordinates instead.", None)])
        elif not results:
            self._show_rows([("No places found", "Try another spelling, or paste coordinates.", None)])
        else:
            self._show_rows([(n, w, (f"{n}, {w}" if w else n, lat, lon)) for n, w, lat, lon in results])

    def _choose_item(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        if data:
            label, lat, lon = data
            self.hide_list()
            self.setText(label)
            self.setCursorPosition(0)                     # show the start of the name, not its end
            self.picked.emit(label, lat, lon)

    # keyboard ------------------------------------------------------------------
    def event(self, e):
        # While the list is open, ↑ ↓ Return Esc belong to it, not to the app's shortcuts
        if e.type() == QEvent.ShortcutOverride and self._list.isVisible() and \
                e.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            e.accept()
            return True
        return super().event(e)

    def keyPressEvent(self, e):
        lst = self._list
        if lst.isVisible() and e.key() in (Qt.Key_Up, Qt.Key_Down):
            rows = [i for i in range(lst.count()) if lst.item(i).flags() & Qt.ItemIsSelectable]
            if rows:
                cur = lst.currentRow() if lst.currentRow() in rows else rows[0]
                i = rows.index(cur) + (1 if e.key() == Qt.Key_Down else -1)
                lst.setCurrentRow(rows[max(0, min(i, len(rows) - 1))])
            return
        if lst.isVisible() and e.key() in (Qt.Key_Return, Qt.Key_Enter):
            item = lst.currentItem()
            if item and item.data(Qt.UserRole):
                self._choose_item(item)
                return
        if lst.isVisible() and e.key() == Qt.Key_Escape:
            self.hide_list()
            return
        super().keyPressEvent(e)

    def focusOutEvent(self, e):
        QTimer.singleShot(200, self.hide_list)            # after a click on the list has landed
        super().focusOutEvent(e)

    def _finished(self):
        t = self.text().strip()
        if not t or engine.parse_gps(t) is not None:
            self.coords_typed.emit(t)


# ---------------------------------------------------------------------------
# Camera / lens presets
# ---------------------------------------------------------------------------
class PresetStore(QObject):
    """The camera and lens presets, shared by every preset field and saved between sessions."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        raw = QSettings("SimpleNeg", "SimpleNeg").value(PRESETS_KEY, "")
        try:
            data = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            data = {}
        self.presets = {k: [str(v) for v in data.get(k, []) if str(v).strip()] for k in ("cameras", "lenses")}

    def add(self, kind: str, name: str):
        name = name.strip()
        if name and name not in self.presets[kind]:
            self.presets[kind].append(name)
            self._save()

    def set(self, kind: str, names: list[str]):
        self.presets[kind] = names
        self._save()

    def _save(self):
        QSettings("SimpleNeg", "SimpleNeg").setValue(PRESETS_KEY, json.dumps(self.presets))
        self.changed.emit()


_store: PresetStore | None = None


def presets() -> PresetStore:
    global _store
    if _store is None:
        _store = PresetStore()
    return _store


class PresetDialog(QDialog):
    """Edit one preset list: double-click a name to rename it."""

    def __init__(self, parent, title: str, names: list[str]):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(360, 380)
        self.list = QListWidget()
        for n in names:
            self._add(n)
        add, rem = QPushButton("Add"), QPushButton("Remove")
        add.clicked.connect(lambda: self.list.editItem(self._add("New preset")))
        rem.clicked.connect(lambda: self.list.takeItem(self.list.currentRow()))
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addWidget(self.list, 1)
        lay.addLayout(row)
        lay.addWidget(buttons)

    def _add(self, name: str) -> QListWidgetItem:
        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.list.addItem(item)
        return item

    def names(self) -> list[str]:
        return [self.list.item(i).text().strip() for i in range(self.list.count()) if self.list.item(i).text().strip()]


class PresetField(QLineEdit):
    """Camera or lens: suggestions from the presets while typing, and a ▾ menu inside the field
    to pick, save the current value, or edit the presets. `committed(text)` after a change."""

    committed = Signal(str)

    def __init__(self, kind: str, placeholder: str = ""):
        super().__init__()
        self.kind = kind
        self.setPlaceholderText(placeholder)
        self._last = ""
        self._model = QStringListModel(presets().presets[kind])
        comp = QCompleter(self._model, self)
        comp.setFilterMode(Qt.MatchContains)              # "m6" finds "Leica M6"
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.activated[str].connect(lambda t: self._commit(t))
        comp.popup().setObjectName("completer")           # styled as a solid list (app stylesheet)
        self.setCompleter(comp)
        presets().changed.connect(lambda: self._model.setStringList(presets().presets[kind]))
        menu_action = QAction(chevron_icon(), "Presets", self)
        menu_action.triggered.connect(self._open_menu)
        self.addAction(menu_action, QLineEdit.TrailingPosition)
        self.editingFinished.connect(lambda: self._commit(self.text()))

    def set_value(self, text: str):
        self.setText(text)
        self._last = text.strip()

    def _commit(self, text: str):
        text = text.strip()
        if text != self._last:
            self._last = text
            self.setText(text)
            self.committed.emit(text)

    def current(self) -> str:
        """The field's own text, else the roll value it shows behind it."""
        hint = self.placeholderText()
        return self.text().strip() or (hint[: -len(FROM_ROLL)] if hint.endswith(FROM_ROLL) else "")

    def _open_menu(self):
        store = presets()
        menu = QMenu(self)
        for name in store.presets[self.kind]:
            menu.addAction(name, lambda n=name: self._commit(n))
        if not store.presets[self.kind]:
            menu.addAction("No presets yet").setEnabled(False)
        menu.addSeparator()
        cur = self.current()
        save = menu.addAction(f"Save “{cur}” as Preset" if cur else "Save as Preset",
                              lambda: (store.add(self.kind, cur), flash(self)))
        save.setEnabled(bool(cur) and cur not in store.presets[self.kind])
        menu.addAction("Edit Presets…", self._edit)
        menu.exec(self.mapToGlobal(QPoint(self.width() - 20, self.height())))

    def _edit(self):
        store = presets()
        dlg = PresetDialog(self, "Camera Presets" if self.kind == "cameras" else "Lens Presets", store.presets[self.kind])
        if dlg.exec():
            store.set(self.kind, dlg.names())


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------
class ExifPanel(QWidget):
    """Date, location, camera, lens and film stock for the frames selected in the filmstrip."""

    edited = Signal(str, object)        # (field key, value; "" goes back to the roll's value)
    apply_fields = Signal(dict)         # several fields at once (a picked place)

    def __init__(self, panel_factory, button_factory):
        super().__init__()
        self._shown: dict[str, str] = {}       # what each field displays now (to spot real edits)
        self.widgets: dict[str, QWidget] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        p, play = panel_factory()                         # which frames the fields apply to + saved note
        top = QHBoxLayout()
        self.info = QLabel("No frame selected")
        self.info.setObjectName("exifTitle")
        self.saved_note = QLabel("")
        self.saved_note.setObjectName("savedNote")
        top.addWidget(self.info, 1)
        top.addWidget(self.saved_note)
        note = QLabel("Empty fields use the roll's details (left). Changes are saved as you go.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        play.addLayout(top)
        play.addWidget(note)
        lay.addWidget(p)
        self._note_timer = QTimer(singleShot=True, interval=2500)
        self._note_timer.timeout.connect(lambda: self.saved_note.setText(""))

        grids: dict[str, QGridLayout] = {}
        for key, label, section, example in FIELDS:
            if section not in grids:
                p, play = panel_factory()
                head = QLabel(section.upper())
                head.setObjectName("header")
                play.addWidget(head)
                g = QGridLayout()
                g.setHorizontalSpacing(10)
                g.setVerticalSpacing(8)
                g.setColumnStretch(1, 1)
                play.addLayout(g)
                grids[section] = g
                lay.addWidget(p)
            g = grids[section]
            r = g.rowCount()
            lab = QLabel(label)
            lab.setObjectName("hint")
            lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            w = self._make_field(key, example)
            g.addWidget(lab, r, 0)
            g.addWidget(w, r, 1)
            self.widgets[key] = w
        lay.addStretch(1)

    def _make_field(self, key: str, example: str) -> QWidget:
        if key == "capture_date":
            d = DateField()
            d.changed_iso.connect(lambda iso: self._emit(key, iso))
            return d
        if key == "gps":
            e = LocationField(example)
            e.picked.connect(self._emit_place)
            e.coords_typed.connect(lambda t: self._emit(key, t))
            return e
        if key in PRESET_KINDS:
            e = PresetField(PRESET_KINDS[key], example)
            e.committed.connect(lambda t, k=key: self._emit(k, t))
            return e
        c = ChevronCombo()                                # film stock
        c.setEditable(True)
        c.setInsertPolicy(QComboBox.NoInsert)
        c.addItem("", "")
        for d in engine.film_stocks():
            c.addItem(d["displayName"], d["id"])
        comp = c.completer()
        comp.setFilterMode(Qt.MatchContains)
        comp.setCompletionMode(QCompleter.PopupCompletion)
        c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(8)                     # the column's width, not the longest film name
        c.activated.connect(lambda i, c=c: self._emit(key, c.itemData(i) or ""))
        return c

    def _emit(self, key: str, value):
        if str(value) == self._shown.get(key, ""):
            return                                        # left the field without changing it
        self._shown[key] = str(value)
        self.edited.emit(key, value)

    def _emit_place(self, label: str, lat: float, lon: float):
        self._shown["gps"] = label
        self.apply_fields.emit({"gps": f"{lat:.6f}, {lon:.6f}", "gps_label": label})

    def saved(self, keys, frames: int):
        """Feedback after an edit landed: the fields light up and a note says where it went."""
        for k in keys:
            if k in self.widgets:
                flash(self.widgets[k])
        self.saved_note.setText(f"✓ Saved · {frames} frame{'s' if frames != 1 else ''}")
        self._note_timer.start()

    def show_values(self, info: str, values: dict):
        """values: key -> (own value or "", the roll's value, all selected frames agree?)."""
        self.info.setText(info)
        for key, w in self.widgets.items():
            own, fallback, same = values.get(key, ("", "", True))
            if isinstance(w, DateField):
                if not same:
                    w.set_iso("", MULTIPLE)
                else:
                    w.set_iso(own or fallback, "No date", inherited=not own and bool(fallback))
                self._shown[key] = w.iso() if same and own else ("" if same else "?")
            elif isinstance(w, QLineEdit):
                if isinstance(w, PresetField):
                    w.set_value(str(own) if same else "")
                else:
                    w.setText(str(own) if same else "")
                w.setPlaceholderText(MULTIPLE if not same else (f"{fallback}{FROM_ROLL}" if fallback else EXAMPLES[key]))
                self._shown[key] = w.text().strip()
            else:
                w.blockSignals(True)
                i = w.findData(own if own else fallback) if same else -1
                w.setCurrentIndex(max(i, 0))
                if not same:
                    w.setEditText("")
                    w.lineEdit().setPlaceholderText(MULTIPLE)
                w.blockSignals(False)
                self._shown[key] = str(w.currentData() if same else "")
        self.setEnabled(bool(values))
