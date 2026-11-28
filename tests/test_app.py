"""The window, driven like a person would: keys, clicks, the crop tool, several frames at once.

Each test names the bug report it guards against, so a change that brings one back fails here.
"""

import sys
import time
import unittest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QCompleter, QLabel

import support
from support import DUST, app, close, close_all, idle, pump, shown_picture, test_file, test_roll, window


class AppTestCase(unittest.TestCase):
    def tearDown(self):
        if getattr(self, "win", None) is not None:
            close(self.win)
            self.win = None
        close_all()

    def open_roll(self, frames=4):
        self.win = window()
        self.win.add_rolls([test_roll(frames)])
        idle(self.win)
        return self.win

    def open_frame(self, **kw):
        self.win = window()
        self.win.add_frames([test_file(**kw)])
        idle(self.win)
        return self.win


class TypingTest(AppTestCase):
    """Editing keys stay off while typing in the film stock box, so every letter can be typed."""

    def type_into(self, combo, word):
        # Without its suggestion list (which takes the keys while it is open): the case that failed
        completer = combo.completer()
        mode = completer.completionMode()
        completer.setCompletionMode(QCompleter.InlineCompletion)
        try:
            line = combo.lineEdit()
            line.clear()
            QTest.mouseClick(line, Qt.LeftButton)
            for ch in word:
                QTest.keyClick(QApplication.focusWidget(), ch)
            return line.text()
        finally:
            completer.setCompletionMode(mode)

    def test_letters_reach_the_film_stock_fields(self):
        win = self.open_roll()
        win._toggle_details(win.current.group, True)
        pump()
        before = win.current.settings.copy()
        self.assertEqual(self.type_into(win.stock_box, "qwe asd"), "qwe asd")
        win.set_tab(1)                                     # Exif tab
        pump()
        exif_stock = [c for c in win.exif_panel.findChildren(type(win.stock_box)) if c.isEditable()][0]
        self.assertEqual(self.type_into(exif_stock, "qwe asd"), "qwe asd")
        self.assertIsNone(win._adjusting)
        for f in ("density", "contrast", "saturation", "cyan", "magenta", "yellow"):
            self.assertEqual(getattr(win.current.settings, f), getattr(before, f), f"typing moved {f}")

    def test_film_stock_field_is_free_to_type_in(self):
        # The field starts empty (a placeholder only): typing searches at once
        win = self.open_roll()
        win._toggle_details(win.current.group, True)
        pump()
        line = win.stock_box.lineEdit()
        self.assertEqual(line.text(), "", "no stock: the field must be empty (a placeholder only)")
        self.assertTrue(line.placeholderText())
        line.setText("Kodak Gold 200")
        QTest.mouseClick(line, Qt.LeftButton)
        pump(0.05)
        self.assertEqual(line.selectedText(), "Kodak Gold 200", "clicking in must select the text, so typing replaces it")
        self.assertEqual(win.stock_box.completer().popup().objectName(), "completer")

    def test_adjust_keys_still_work_on_the_photo(self):
        win = self.open_frame()
        win.preview.setFocus()
        d = win.current.settings.density
        QTest.keyPress(win.preview, Qt.Key_Q)
        QTest.keyClick(win.preview, Qt.Key_Up)
        QTest.keyRelease(win.preview, Qt.Key_Q)
        idle(win)
        self.assertGreater(win.current.settings.density, d)


class CropTest(AppTestCase):
    def auto_cropped(self, **kw):
        win = self.open_frame(**kw)
        win.autocrop_cb.setChecked(True)
        idle(win)
        st = win.current.settings
        self.assertIsNotNone(st.auto_rect, "the auto crop was not kept")
        return win, st

    def test_edit_starts_from_the_auto_crop(self):
        # Edit starts from the auto crop (not the whole scan) and Apply keeps Auto on
        win, st = self.auto_cropped()
        win.toggle_crop()
        x0, y0, x1, y1 = st.auto_rect
        self.assertEqual(tuple(round(v, 4) for v in win.preview.crop.getCoords()),
                         tuple(round(v, 4) for v in (x0, y0, x1, y1)))
        win.toggle_crop(apply=True)
        idle(win)
        self.assertTrue(st.auto_crop, "applying an untouched crop turned Auto crop off")
        self.assertIsNone(st.crop)

    def searches(self, win):
        """Count the auto crop searches the window starts from here on."""
        count = []
        real = app.engine.find_auto_crop
        app.engine.find_auto_crop = lambda *a, **kw: count.append(1) or real(*a, **kw)
        self.addCleanup(setattr, app.engine, "find_auto_crop", real)
        return count

    def test_straightening_keeps_the_auto_crop(self):
        # Straightening a few degrees searched for the auto crop again
        win, st = self.auto_cropped()
        found = st.auto_rect
        count = self.searches(win)
        win.straighten.set_value(2.5)
        win.on_straighten_slider()
        win._straighten_released()
        idle(win)
        self.assertEqual(count, [], "straightening searched for the auto crop again")
        self.assertEqual(st.auto_rect, found)

    def test_turning_moves_the_auto_crop_without_searching(self):
        win, st = self.auto_cropped()
        x0, y0, x1, y1 = st.auto_rect
        count = self.searches(win)
        win.rotate(+1)
        idle(win)
        self.assertEqual(count, [])
        self.assertEqual(tuple(round(v, 6) for v in st.auto_rect), tuple(round(v, 6) for v in (y0, 1 - x1, y1, 1 - x0)))
        win.mirror()
        idle(win)
        self.assertEqual(count, [])

    def test_refresh_searches_again(self):
        win, st = self.auto_cropped()
        self.assertTrue(win.autocrop_refresh.isEnabled())
        win.straighten.set_value(2.5)
        win.on_straighten_slider()
        win._straighten_released()
        idle(win)
        count = self.searches(win)
        win.refresh_auto_crop()
        idle(win)
        self.assertEqual(len(count), 1, "Refresh did not search again")
        self.assertIsNotNone(st.auto_rect)
        self.assertAlmostEqual(st.fine_rotation, 2.5, msg="Refresh changed an angle you set")
        win.autocrop_cb.setChecked(False)
        self.assertFalse(win.autocrop_refresh.isEnabled())

    def test_straightening_a_cropped_frame_does_not_zoom(self):
        # The live turn zoomed in to hide corners, then the real render (crop kept) zoomed back out.
        # A crop (auto or yours) stays put while the picture turns; with no crop NegPy trims the
        # corners, so the live turn zooms to match.
        win, st = self.auto_cropped()
        win.straighten.set_value(3.0)
        win.on_straighten_slider()
        self.assertFalse(win.preview.turn_fills, "the live turn zooms although the crop stays put")
        win._straighten_timer.stop()
        win._straighten_released()
        idle(win)
        win.autocrop_cb.setChecked(False)
        idle(win)
        win.straighten.set_value(2.0)
        win.on_straighten_slider()
        self.assertTrue(win.preview.turn_fills, "with no crop the live turn must zoom like the render")

    def test_cropping_shows_the_picture_bigger(self):
        win = self.open_frame()
        normal = win.preview.view_rect().width()
        win.toggle_crop()
        idle(win)
        self.assertFalse(win.left.isVisible() or win.grid.isVisible() or win.right_stack.isVisible())
        self.assertGreater(win.preview.view_rect().width(), normal * 1.2, "the crop view is not bigger")
        win.toggle_crop(apply=False)
        idle(win)
        self.assertTrue(win.left.isVisible() and win.grid.isVisible() and win.right_stack.isVisible())

    def test_opening_the_crop_glides_and_the_toolbar_is_short(self):
        import time
        win = self.open_frame()
        pv = win.preview
        start = pv.view_rect().width()
        widths = []
        t = time.time()
        win.toggle_crop()
        while time.time() - t < 0.6:
            pump(0.02)
            widths.append(pv.view_rect().width())
        idle(win)
        end = pv.view_rect().width()
        between = [w for w in widths if start + 5 < w < end - 5]
        self.assertTrue(between, f"the picture jumped to its crop size: {widths}")
        names = [b.text() for b in win.toolbar_crop.findChildren(app.QPushButton)]
        self.assertNotIn("Whole Frame", names)
        self.assertIn("Reset", names)
        self.assertTrue(any(b.toolTip().startswith("Turn") for b in win.toolbar_crop.findChildren(app.QPushButton)))
        win.toggle_crop(apply=False)
        idle(win)

    def test_whole_frame_still_gives_the_whole_scan(self):
        win, st = self.auto_cropped()
        win.toggle_crop()
        win.crop_reset()
        win.toggle_crop(apply=True)
        idle(win)
        self.assertFalse(st.auto_crop)
        self.assertIsNone(st.crop)

    def test_formats_stay_inside_the_auto_crop(self):
        win, st = self.auto_cropped()
        win.toggle_crop()
        win.set_crop_format("1:1")
        area = win.preview.format_area
        c = win.preview.crop
        self.assertTrue(area.adjusted(-1e-6, -1e-6, 1e-6, 1e-6).contains(c), "the 1:1 frame went over the border")

    def test_square_then_3_2_keeps_a_portrait_upright(self):
        # A portrait crop stays portrait when the format changes
        win = self.open_frame(portrait=True)
        win.toggle_crop()
        idle(win)
        win.set_crop_format("1:1")
        win.set_crop_format("3:2")
        self.assertTrue(win.preview.crop_is_portrait())

    def test_format_picked_before_the_picture_arrives_gets_its_shape(self):
        win = self.open_frame()
        win.toggle_crop()
        win.set_crop_format("3:2")             # at once: the uncropped picture is still on its way
        idle(win)
        pv = win.preview
        self.assertAlmostEqual(pv.crop.width() * pv._aspect / pv.crop.height(), 1.5, places=3)

    def test_crop_edit_does_not_turn_the_frame(self):
        # Reported on a Mac: going into the crop, the frame sometimes turned 90 degrees. The format picked
        # in an earlier crop was still held, and the uncropped picture arriving refitted to it.
        win = self.open_frame()
        pv = win.preview
        win.toggle_crop()
        idle(win)
        win.set_crop_format("4:3")
        pv.crop = pv._largest(pv._k(), pv.crop.center(), app.QRectF(0.2, 0.2, 0.6, 0.6))
        win.toggle_crop(apply=True)
        idle(win)
        win.rotate(1)                          # the crop turns with the picture: now portrait
        idle(win)
        win.toggle_crop()
        idle(win)
        self.assertTrue(pv.crop_is_portrait(), "the portrait crop turned to landscape")
        x0, y0, x1, y1 = win.current.settings.crop
        self.assertAlmostEqual(pv.crop.width(), x1 - x0, places=3)
        self.assertAlmostEqual(pv.crop.height(), y1 - y0, places=3)
        win.toggle_crop(apply=False)
        idle(win)


class PreviewTest(AppTestCase):
    def test_no_strobe_while_dragging(self):
        # While a slider moves the preview stays on the draft; it never flips to the sharp render and back
        win = self.open_frame()
        shown = []
        original = win.preview.set_image
        # what is seen: a soft (smaller) picture or a sharp one; with a graphics card edits render sharp
        win.preview.set_image = lambda pix, draft=False, **kw: (
            shown.append("F" if pix.width() >= app.PREVIEW_SIZE - 2 or pix.height() >= app.PREVIEW_SIZE - 2 else "d"),
            original(pix, draft, **kw))
        fader = win.steppers["density"]
        for _ in range(4):                     # move for 0.4 s, hold still 0.3 s, four times
            t = time.time()
            while time.time() - t < 0.4:
                fader._set(fader.value() + 0.01, emit=True)
                pump(0.03)
            pump(0.3)
        idle(win)
        seq = "".join(shown)
        self.assertNotIn("Fd", seq, f"sharp then soft again while dragging: {seq}")
        self.assertTrue(seq.endswith("F"), "no sharp picture once the slider stopped")

    def test_colour_noise_hidden_for_bw(self):
        win = self.open_frame()
        st = win.current.settings
        st.grayscale = True
        win.load_controls(st)
        self.assertTrue(win.sliders["color_noise"].isHidden())
        st.grayscale = False
        win.load_controls(st)
        self.assertFalse(win.sliders["color_noise"].isHidden())


class HealInWindowTest(AppTestCase):
    def test_click_heals_the_speck(self):
        win = self.open_frame()
        before = shown_picture(win)
        win.toggle_heal()
        idle(win)
        r = win.preview.view_rect()
        x, y = DUST[0]
        QTest.mouseClick(win.preview, Qt.LeftButton, Qt.NoModifier,
                         QPoint(int(r.x() + x * r.width()), int(r.y() + y * r.height())))
        idle(win)
        self.assertEqual(len(win.current.settings.heal_spots), 1)
        after = shown_picture(win)
        h, w = before.shape[:2]
        self.assertGreater(abs(int(after[int(y * h), int(x * w)].max()) - int(before[int(y * h), int(x * w)].max())), 30)


class SelectionTest(AppTestCase):
    """GitHub #6: changes should reach every selected frame."""

    def select(self, rows):
        g = self.win.grid
        QTest.mouseClick(g.viewport(), Qt.LeftButton, Qt.NoModifier, g.visualItemRect(g.item(rows[0])).center())
        idle(self.win)
        for r in rows[1:]:
            QTest.mouseClick(g.viewport(), Qt.LeftButton, Qt.ControlModifier, g.visualItemRect(g.item(r)).center())
        idle(self.win)

    def values(self, field):
        return [round(getattr(p.settings, field), 3) for p in self.win.photos]

    def test_a_click_selects_only_that_frame(self):
        # The strip slid under a held button (centring on the press): a click with the slightest
        # hand movement selected the frames it passed, and edits then reached them too
        self.win = window()
        self.win.add_rolls([test_roll(16, name="Long roll")])
        idle(self.win)
        g, vp = self.win.grid, self.win.grid.viewport()
        for target in (7, 9):
            pos = g.visualItemRect(g.item(target)).center()
            QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, pos)
            pump(0.05)
            QTest.mouseMove(vp, pos + QPoint(2, 0))
            QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, pos + QPoint(2, 0))
            idle(self.win)
            with self.subTest(frame=target + 1):
                self.assertEqual(sorted(g.row(i) for i in g.selectedItems()), [target])
                self.assertIs(self.win.current, self.win.photos[target])
        self.assertTrue(g.visualItemRect(g.item(9)).intersects(vp.rect()), "the clicked frame was not kept in view")
        QTest.mouseClick(vp, Qt.LeftButton, Qt.ShiftModifier, g.visualItemRect(g.item(11)).center())
        idle(self.win)
        self.assertEqual(sorted(g.row(i) for i in g.selectedItems()), [9, 10, 11], "Shift-click no longer selects a run")

    def test_faders_reach_the_selection(self):
        win = self.open_roll()
        self.select([0, 1, 2])
        win.nudge("density", 2)
        idle(win)
        d = self.values("density")
        self.assertEqual(d[0], d[1])
        self.assertEqual(d[1], d[2])
        self.assertNotEqual(d[2], d[3], "a frame that was not selected changed")

    def test_white_balance_of_one_frame_does_not_travel(self):
        # A white balance picked on the current frame (Pick Grey) is that frame's alone: the next
        # slider move does not copy it to the other selected frames.
        win = self.open_roll()
        self.select([0, 1])
        win.on_wb(win.current, (1.2, 1.0, 0.9))        # what a Pick Grey result does
        idle(win)
        win.nudge("density", 1)
        idle(win)
        others = [p for p in win.photos[:2] if p is not win.current]
        self.assertEqual(win.current.settings.cyan, 1.2)
        self.assertEqual(others[0].settings.cyan, 1.0, "the picked white balance was copied")

    def test_reset_and_auto_white_balance_reach_the_selection(self):
        win = self.open_roll()
        self.select([0, 1, 2])
        win.auto_wb()
        idle(win)
        self.assertTrue(all(c != 1.0 for c in self.values("cyan")[:3]), "Auto WB skipped a selected frame")
        win.reset_wb()
        idle(win)
        self.assertEqual(self.values("cyan")[:3], [1.0, 1.0, 1.0])
        self.assertTrue(win.autowb_btn.isEnabled())


class RollListTest(AppTestCase):
    """A short screen with several rolls: the roll list scrolls."""

    def test_scrolling_over_a_dropdown_scrolls_the_list(self):
        # Scrolling past the film stock or development dropdown changed the roll's film
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QWheelEvent
        self.win = window(1280, 700)
        self.win.add_rolls([test_roll(2, name=f"Roll {i}") for i in range(5)])
        idle(self.win)
        st = self.win.current.settings
        before = repr(st)
        bar = self.win.cards_scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0, "the list does not need to scroll: make the window shorter")
        for combo in [c for c in self.win.roll_box.findChildren(QComboBox) if c.isVisible()]:
            bar.setValue(bar.maximum() // 2)
            y, text = bar.value(), combo.currentText()
            c = QPointF(combo.rect().center())
            QApplication.sendEvent(combo, QWheelEvent(c, combo.mapToGlobal(c), QPoint(), QPoint(0, -120), Qt.NoButton,
                                                      Qt.NoModifier, Qt.NoScrollPhase, False))
            with self.subTest(combo=text):
                self.assertEqual(combo.currentText(), text, "the wheel changed the dropdown")
                self.assertGreater(bar.value(), y, "the list did not scroll")
        self.assertEqual(repr(st), before, "the wheel changed the roll's settings")

    def test_an_opened_roll_shows_its_details(self):
        self.win = window(1280, 700)
        self.win.add_rolls([test_roll(2, name=f"Roll {i}") for i in range(5)])
        idle(self.win)
        g = self.win.groups[-1]
        self.win.cards_scroll.verticalScrollBar().setValue(0)
        self.win._toggle_details(g, True)
        idle(self.win)
        vp = self.win.cards_scroll.viewport()
        top = g.card.mapTo(vp, QPoint(0, 0)).y()
        self.assertGreaterEqual(top, 0)
        self.assertLess(top + 60, vp.height(), "the opened roll is below the bottom of the list")


class ExportScopeTest(AppTestCase):
    """Export all groups, the current group, or the selected frames ("Selected (n)")."""

    def shown(self, **kw):
        """Open the export window (without waiting on it) and return its choices and the one shown."""
        seen = {}

        def capture(dlg):
            seen["items"] = [dlg.scope.itemText(i) for i in range(dlg.scope.count())]
            seen["current"] = dlg.scope.currentText()
            return 0                               # cancelled: nothing is exported
        for g in self.win.groups:                  # export asks for every roll's film first
            if g.kind == "roll" and not g.stock_id:
                g.stock_id = "other-color"
        real = app.ExportDialog.exec
        app.ExportDialog.exec = capture
        try:
            self.win.open_export(**kw)
        finally:
            app.ExportDialog.exec = real
        return seen["items"], seen["current"]

    def select(self, rows):
        from PySide6.QtCore import QItemSelectionModel
        g = self.win.grid
        g.clearSelection()
        for r in rows:
            g.item(r).setSelected(True)
        g.setCurrentRow(rows[0], QItemSelectionModel.Select)

    def test_several_selected_frames_are_offered_and_chosen(self):
        self.open_roll(4)
        self.select([0, 1, 2])
        items, current = self.shown()
        self.assertEqual(items, ["All groups", "Current group", "Selected (3)"])
        self.assertEqual(current, "Selected (3)")

    def test_one_frame_is_not_a_selection(self):
        self.open_roll(4)
        self.select([1])
        items, current = self.shown()
        self.assertEqual(items, ["All groups", "Current group"])

    def test_export_selected_from_the_filmstrip(self):
        self.open_roll(4)
        self.select([2])
        items, current = self.shown(selected_only=True)
        self.assertEqual(current, "Selected (1)")


class BatchEditTest(AppTestCase):
    """Changes reach every selected frame with a notice on the photo; Sync Roll sends edits to the group."""

    select = ExportScopeTest.select

    def test_a_change_to_several_frames_says_so(self):
        win = self.open_roll(4)
        self.select([0, 1, 2])
        win.nudge("density", 2)
        self.assertTrue(win.toast.isVisible())
        self.assertEqual(win.toast.text(), "Updating 3 photos")

    def test_one_frame_has_no_notice(self):
        win = self.open_roll(4)
        self.select([1])
        win.nudge("density", 2)
        self.assertFalse(win.toast.isVisible())

    def test_turning_and_mirroring_reach_the_selection(self):
        win = self.open_roll(4)
        self.select([0, 1, 2])
        win.rotate(+1)
        win.mirror()
        idle(win)
        self.assertEqual([p.settings.rotation for p in win.photos], [1, 1, 1, 0])
        self.assertEqual([p.settings.mirror for p in win.photos], [True, True, True, False])

    def test_sync_roll_sends_the_chosen_edits(self):
        app.set_pref("sync", "")
        self.addCleanup(app.set_pref, "sync", "")
        win = self.open_roll(4)
        self.select([0])
        win.nudge("density", 3)
        win.nudge("cyan", 2)
        win.mirror()
        idle(win)
        src = win.current.settings
        win.sync_roll()                            # default: the look, not crop and rotation
        idle(win)
        for p in win.photos[1:]:
            self.assertAlmostEqual(p.settings.density, src.density)
            self.assertAlmostEqual(p.settings.cyan, src.cyan)
            self.assertFalse(p.settings.mirror, "crop and rotation were synced although not ticked")
        self.assertEqual(win.toast.text(), "Synced to 4 photos")
        app.set_pref("sync", '["framing"]')
        win.sync_roll()
        idle(win)
        self.assertTrue(all(p.settings.mirror for p in win.photos))

    def test_sync_and_its_undo_carry_every_refine_control(self):
        # every Refine control must be in a Sync group, reach the other frames, and come back with Cmd+Z
        app.set_pref("sync", "")
        self.addCleanup(app.set_pref, "sync", "")
        synced = {f for _label, fields in app.SYNC_GROUPS.values() for f in fields}
        refine = set(self.win_fields())
        self.assertEqual(refine - synced, set(), "Refine controls Sync Roll cannot send")
        win = self.open_roll(3)
        self.assertEqual(set(win.sliders) | set(win.hsl_sliders) | {"paper"}, refine,
                         "a Refine control was added: list it in win_fields so Sync and undo are tested for it")
        self.select([0])
        st = win.current.settings
        for key in refine:
            value = getattr(st, key)
            if isinstance(value, str):
                setattr(st, key, "kodak_endura")
                continue
            setattr(st, key, tuple(0.4 if i == 2 else v for i, v in enumerate(value)) if isinstance(value, tuple)
                    else (0.7 if value < 0.5 else 0.1))
        win.load_controls(st)
        win._history_timer.stop()
        win._commit_history()
        before = [p.settings.copy() for p in win.photos[1:]]
        win.sync_roll()
        idle(win)
        for p in win.photos[1:]:
            for key in refine:
                with self.subTest(frame=p.path.name, key=key):
                    self.assertEqual(getattr(p.settings, key), getattr(st, key))
        win.undo()
        idle(win)
        for p, old in zip(win.photos[1:], before):
            for key in refine:
                with self.subTest(undone=p.path.name, key=key):
                    self.assertEqual(getattr(p.settings, key), getattr(old, key))

    @staticmethod
    def win_fields():
        """Every setting a Refine control edits: the sliders and the colour mixer."""
        return ["exposure", "highlight_tone", "shadow_tone", "white_point", "black_point", "temp", "tint",
                "sharpen", "color_noise", "glow", "hsl_hue", "hsl_saturation", "hsl_luminance", "paper"]

    def test_undo_takes_back_a_whole_sync(self):
        # Cmd+Z after Sync Roll only undid the current frame's last edit
        app.set_pref("sync", "")
        self.addCleanup(app.set_pref, "sync", "")
        win = self.open_roll(4)
        self.select([0])
        before = [p.settings.density for p in win.photos]
        win.nudge("density", 3)
        win._history_timer.stop()
        win._commit_history()
        win.sync_roll()
        idle(win)
        synced = win.current.settings.density
        win.undo()
        idle(win)
        self.assertEqual([p.settings.density for p in win.photos[1:]], before[1:], "the sync was not undone")
        self.assertAlmostEqual(win.current.settings.density, synced, msg="undo also took this frame's own edit")
        win.redo()
        idle(win)
        self.assertTrue(all(abs(p.settings.density - synced) < 1e-9 for p in win.photos))
        win.undo()                                 # the sync again, then this frame's own edit
        win.undo()
        idle(win)
        self.assertAlmostEqual(win.current.settings.density, before[0])

    def test_undo_takes_back_an_edit_of_the_whole_selection(self):
        win = self.open_roll(4)
        self.select([0, 1, 2])
        before = [p.settings.density for p in win.photos]
        win.nudge("density", 2)
        win._history_timer.stop()
        win._commit_history()
        self.assertNotEqual([p.settings.density for p in win.photos[:3]], before[:3])
        win.undo()
        idle(win)
        self.assertEqual([p.settings.density for p in win.photos], before, "undo left the other frames changed")
        win.rotate(+1)
        win._history_timer.stop()
        win._commit_history()
        win.undo()
        idle(win)
        self.assertEqual([p.settings.rotation for p in win.photos], [0, 0, 0, 0])

    def test_reset_reaches_the_selection_and_undoes_in_one_step(self):
        win = self.open_roll(4)
        self.select([0, 1, 2])
        win.nudge("density", 3)
        win._history_timer.stop()
        win._commit_history()
        edited = [p.settings.density for p in win.photos]
        win.reset_current()
        win._history_timer.stop()
        win._commit_history()
        idle(win)
        self.assertEqual([p.settings.density for p in win.photos[:3]], [1.0, 1.0, 1.0])
        self.assertEqual(win.toast.text(), "Updating 3 photos")
        win.undo()
        idle(win)
        self.assertEqual([p.settings.density for p in win.photos], edited)

    def test_the_gear_remembers_what_to_sync(self):
        import json
        app.set_pref("sync", "")
        self.addCleanup(app.set_pref, "sync", "")
        win = self.open_roll(2)
        win._sync_choices()
        boxes = {cb.text(): cb for cb in win._sync_pop.findChildren(app.QCheckBox)}
        self.assertFalse(boxes["Crop and rotation"].isChecked())
        boxes["Colour (C / M / Y, temp, tint)"].setChecked(False)
        win._sync_pop.close()
        self.assertEqual(json.loads(app.pref("sync")), ["exposure", "saturation", "tone", "refine", "paper", "points", "mixer"])


class TrashAfterExportTest(AppTestCase):
    """Export can move the original scans to the Trash: only scans whose own export worked."""

    def setUp(self):
        import shutil
        self.folder = support.TMP / f"trash-roll-{time.time_ns()}"
        self.folder.mkdir()
        for i, src in enumerate(sorted(test_roll(3, name="Trash source").glob("*.tif"))):
            shutil.copy(src, self.folder / f"frame{i + 1:02d}.tif")
        self.bin = support.TMP / f"bin-{time.time_ns()}"
        self.bin.mkdir()
        self._real = app.move_to_trash

        def fake_trash(path):              # never the real Trash in a test
            path.rename(self.bin / path.name)
            return True
        app.move_to_trash = fake_trash

    def tearDown(self):
        app.move_to_trash = self._real
        super().tearDown()

    def export(self, photos, trash):
        win = self.win
        win.export_extra = {"dest": str(support.TMP / f"out-{time.time_ns()}"), "per_roll": False, "reveal": False}
        win._export(photos, trash=trash)
        end = time.time() + 120
        while win._export_left and time.time() < end:
            pump(0.05)
        idle(win)

    def test_exported_scans_go_to_the_trash_and_leave_the_list(self):
        self.win = window()
        self.win.add_rolls([self.folder])
        idle(self.win)
        g = self.win.groups[0]
        first, second, third = g.photos
        self.export([first, second], trash=True)
        self.assertEqual(sorted(p.name for p in self.bin.iterdir()), ["frame01.tif", "frame02.tif"])
        self.assertTrue(third.path.exists(), "a scan that was not exported was moved")
        self.assertEqual(g.photos, [third])

    def test_nothing_moves_without_the_option(self):
        self.win = window()
        self.win.add_rolls([self.folder])
        idle(self.win)
        self.export(list(self.win.groups[0].photos), trash=False)
        self.assertEqual(list(self.bin.iterdir()), [])
        self.assertEqual(len(self.win.groups[0].photos), 3)

    def test_a_failed_export_keeps_its_scan(self):
        self.win = window()
        self.win.add_rolls([self.folder])
        idle(self.win)
        g = self.win.groups[0]
        real_export = app.engine.export

        def broken(path, *a, **kw):
            if path.name == "frame02.tif":
                raise OSError("disk full")
            return real_export(path, *a, **kw)
        app.engine.export = broken
        try:
            self.export(list(g.photos), trash=True)
        finally:
            app.engine.export = real_export
        self.win._export_failed = []
        self.assertEqual(sorted(p.name for p in self.bin.iterdir()), ["frame01.tif", "frame03.tif"])
        self.assertTrue((self.folder / "frame02.tif").exists())
        self.assertEqual([p.path.name for p in g.photos], ["frame02.tif"])

    def test_the_option_is_off_each_time(self):
        dlg = app.ExportDialog(None, app.engine.ExportOptions(), {"trash": True}, False, True)
        self.assertFalse(dlg.trash.isChecked())


class RollNameTest(AppTestCase):
    """Roll names come from folder names: a long one is cut short, and the pencil renames the roll."""

    def test_a_long_name_is_cut_short(self):
        self.win = window()
        long_name = "2026-11-14 Kyoto and Nara with Portra 400 pushed one stop, second half of the trip"
        self.win.add_rolls([test_roll(2, name=long_name)])
        idle(self.win)
        card = self.win.groups[0].card
        self.assertLessEqual(card.width(), self.win.left.width(), "a long name widened the roll card")
        self.assertEqual(card.title.text(), long_name)
        self.assertTrue(QLabel.text(card.title).endswith("…"))
        self.assertEqual(card.head.toolTip(), long_name)

    def test_the_trash_asks_before_removing(self):
        win = self.open_roll(2)
        g = win.groups[0]
        answers = []

        def answer(box, which):
            def exec_(self_box):
                answers.append(self_box.text())
                [b for b in self_box.buttons() if b.text() == which][0].click()
            return exec_
        real = app.QMessageBox.exec
        try:
            app.QMessageBox.exec = answer(None, "Cancel")
            g.card.trash.click()
            self.assertIn(g, win.groups, "Cancel removed the roll")
            app.QMessageBox.exec = answer(None, "Remove")
            g.card.trash.click()
            self.assertNotIn(g, win.groups)
        finally:
            app.QMessageBox.exec = real
        self.assertEqual(len(answers), 2, "no warning was shown")

    def test_the_pencil_renames_the_roll(self):
        win = self.open_roll(2)
        g, card = win.groups[0], win.groups[0].card
        self.assertTrue(card.pencil.isVisibleTo(card))
        card.start_rename()
        card.name_edit.setText("Kyoto, day 1")
        card.name_edit.returnPressed.emit()
        self.assertEqual(g.name, "Kyoto, day 1")
        self.assertEqual(card.title.text(), "Kyoto, day 1")
        self.assertEqual(win._tokens(win.current)["roll"], "Kyoto, day 1")
        card.start_rename()
        card.name_edit.setText("Something else")
        QTest.keyClick(card.name_edit, Qt.Key_Escape)
        self.assertEqual(g.name, "Kyoto, day 1", "Esc did not cancel the rename")


class BorderKeyTest(AppTestCase):
    """O shows or hides the white border; Settings and the export window follow."""

    def test_o_toggles_the_border(self):
        app.set_pref("border", True)
        self.addCleanup(app.set_pref, "border", True)
        win = self.open_frame()
        self.assertGreater(win.preview.border, 0)
        win.preview.setFocus()
        QTest.keyClick(win.preview, Qt.Key_O)
        idle(win)
        self.assertFalse(app.pref("border"))
        self.assertEqual(win.preview.border, 0)
        self.assertFalse(win.pref_border.isChecked())
        dlg = app.ExportDialog(None, app.engine.ExportOptions(), {}, 0, False)
        self.assertFalse(dlg.border.isChecked(), "export starts with the border as shown")
        dlg.border.setChecked(True)
        self.assertEqual(dlg.options()[0].border_px, app.pref("border_px"))


class SmoothEditingTest(AppTestCase):
    """The basics must feel solid: turning, undo, sliders, crop, switching frames."""

    def open_roll(self, frames=4):
        self.win = window()
        self.win.add_rolls([test_roll(frames, name=f"Smooth {frames}")])
        idle(self.win)
        return self.win

    def orientations(self, win):
        """The four ways the current frame can point, as small grey pictures to compare with."""
        import numpy as np
        from PySide6.QtGui import QImage

        refs = []
        for k in range(4):
            arr = app.engine.preview(win.current.path, app.engine.Settings(rotation=k), 400)
            q = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888)
            refs.append(self.grey(q))

        def which(pix):
            a = self.grey(pix.toImage())
            return int(np.argmin([np.abs(a - r).mean() for r in refs]))
        return which

    @staticmethod
    def grey(img):
        import numpy as np
        from PySide6.QtGui import QImage
        q = img.scaled(48, 48).convertToFormat(QImage.Format_Grayscale8)
        return np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, :q.width()].astype(float)

    def test_three_quick_turns_never_step_back(self):
        # A render started before the second and third press arrived late and turned the picture back once
        win = self.open_roll(2)
        which = self.orientations(win)
        pv = win.preview
        for trial in range(3):
            shown, real = [], pv.set_image
            pv.set_image = lambda pix, draft=False, real=real, shown=shown, **kw: (
                real(pix, draft=draft, **kw), shown.append(which(pix)) if pix else None)
            start = win.current.settings.rotation
            try:
                for _ in range(3):
                    win.rotate(+1)
                    pump(0.04)
                idle(win)
            finally:
                pv.set_image = real
            want = (start + 3) % 4
            with self.subTest(trial=trial):
                back = [b for a, b in zip([start] + shown, shown) if (b - a) % 4 == 3]
                self.assertEqual(back, [], f"the picture turned back: {shown}")
                self.assertEqual(shown[-1], want)
                self.assertEqual(win.current.settings.rotation, want)

    def test_undo_walks_back_through_turns_of_the_whole_selection(self):
        # Turned twice with every frame selected: the second Cmd+Z undid the current frame alone
        win = self.open_roll(4)
        for r in range(4):
            win.grid.item(r).setSelected(True)
        turns = lambda: [p.settings.rotation for p in win.photos]   # noqa: E731
        win.rotate(+1)
        win.rotate(+1)
        idle(win)
        self.assertEqual(turns(), [2, 2, 2, 2])
        for want in ([1, 1, 1, 1], [0, 0, 0, 0]):
            win.undo()
            idle(win)
            self.assertEqual(turns(), want)
        for want in ([1, 1, 1, 1], [2, 2, 2, 2]):
            win.redo()
            idle(win)
            self.assertEqual(turns(), want)

    def test_editing_never_moves_or_resizes_the_picture(self):
        win = self.open_roll(2)
        pv = win.preview

        def steady(name, edit):
            rects = []
            real = pv.set_image
            pv.set_image = lambda pix, draft=False, **kw: (real(pix, draft=draft, **kw), rects.append(pv.view_rect().getRect()))
            before = pv.view_rect().getRect()
            try:
                edit()
                pump(0.6)
                idle(win)
            finally:
                pv.set_image = real
            with self.subTest(name):
                self.assertTrue(rects, "nothing was rendered")
                self.assertEqual(set(rects), {before}, "the picture moved or changed size")
        steady("density", lambda: win.nudge("density", 1))
        steady("refine", lambda: [win.sliders["sharpen"].set_value(0.3 + 0.05 * i) or win.on_setting_changed()
                                  or pump(0.01) for i in range(5)])
        win.autocrop_cb.setChecked(True)
        idle(win)
        steady("auto crop on", lambda: win.nudge("cyan", 2))
        win.straighten.set_value(1.8)
        win.on_straighten_slider()
        win._straighten_released()
        idle(win)
        steady("straightened", lambda: win.nudge("saturation", 1))

    def test_crop_formats_and_reset(self):
        win = self.open_frame()
        self.assertIn("65:24", win.ratio_btns)
        self.assertNotIn("XPan", win.ratio_btns)
        win.current.settings.crop = (0.2, 0.2, 0.8, 0.8)
        for auto in (False, True):
            app.set_pref("auto_crop_new", auto)
            try:
                win.current.settings.crop = (0.2, 0.2, 0.8, 0.8)
                win.crop_restore()
                idle(win)
                with self.subTest(auto_crop_for_new_frames=auto):
                    self.assertIsNone(win.current.settings.crop)
                    self.assertEqual(win.current.settings.auto_crop, auto)
                    if auto:
                        self.assertIsNotNone(win.current.settings.auto_rect, "auto crop was not found again")
            finally:
                app.set_pref("auto_crop_new", False)

    def test_original_scan_on_its_key_and_button(self):
        app.set_pref("border", True)
        win = self.open_frame()
        thumb = win.current.thumb
        win.preview.setFocus()
        QTest.keyClick(win.preview, Qt.Key_T)
        idle(win)
        self.assertTrue(win._show_original)
        self.assertTrue(win.original_btn.property("on"))
        self.assertEqual(win.preview.border, 0, "the white border belongs to the conversion, not the negative")
        self.assertIs(win.current.thumb, thumb, "the filmstrip must keep the conversion")
        win.original_btn.click()
        idle(win)
        self.assertFalse(win._show_original)
        self.assertGreater(win.preview.border, 0)

    def test_title_shows_the_frame_and_its_exposure(self):
        real = app.engine.scan_exposure
        app.engine.scan_exposure = lambda path: ("1/125", "f/5.6", "ISO 100")
        try:
            self.win = window()
            self.win.add_rolls([test_roll(2, name="Folder name")])
            idle(self.win)
            text = self.win.title.text()
            info = self.win.title_info.info_html
        finally:
            app.engine.scan_exposure = real
        self.assertIn(self.win.current.path.name, text)
        self.assertNotIn("Folder name", text, "the roll's name is on the left already")
        self.assertNotIn("1/125", text, "the title keeps only the file name")
        self.assertTrue(self.win.title_info.isVisible())
        for part in ("1 of 2", "Film", "1/125", "f/5.6", "ISO 100", "exp-shutter.svg", "exp-aperture.svg", "exp-iso.svg",
                     "PICTURE", "SCAN", "File", "TIF", "px", "CONVERSION", "Lab print", "white light"):
            self.assertIn(part, info, "the info dot does not show the frame's details")

    def test_switching_frames_shows_the_picture_at_once(self):
        win = self.open_roll(4)
        pv = win.preview
        shown, real = [], pv.set_image
        pv.set_image = lambda pix, draft=False, **kw: (real(pix, draft=draft, **kw), shown.append(draft))
        try:
            win.grid.setCurrentRow(1)
            self.assertTrue(shown, "nothing was on screen until the render finished")
            idle(win)
        finally:
            pv.set_image = real
        self.assertFalse(shown[-1], "the sharp render did not follow")
        self.assertIn(win.photos[2].path, win._prefetched, "the next frame was not prepared")


class GpuViewerTest(AppTestCase):
    """With the graphics card: edits render sharp once, straightening renders as you drag, and the
    frames either side are prepared so a switch is quick."""

    def setUp(self):
        if not app.engine.gpu_active() or app.engine._gpu_processor() is None:
            self.skipTest("no usable graphics card here")

    def count_renders(self, win, action, wait=0.8):
        shown, real = [], win.preview.set_image
        win.preview.set_image = lambda pix, draft=False, **kw: (real(pix, draft=draft, **kw), shown.append(pix.width()))
        try:
            action()
            pump(wait)
            idle(win)
        finally:
            win.preview.set_image = real
        return shown

    def test_one_sharp_render_per_edit(self):
        win = self.open_roll(2)
        shown = self.count_renders(win, lambda: win.nudge("density", 1))
        self.assertEqual(len(shown), 1, "the sharp render was repeated after the edit came to rest")
        self.assertGreaterEqual(shown[0], app.PREVIEW_SIZE - 2, "the edit rendered a soft draft")

    def test_straightening_renders_while_dragging(self):
        win = self.open_roll(2)

        def drag():
            for i in range(12):
                win.straighten.set_value(0.2 * (i + 1))
                win.on_straighten_slider()
                pump(0.05)
        shown = self.count_renders(win, drag, wait=0.0)
        self.assertGreaterEqual(len(shown), 3, "no real renders while dragging")
        win._straighten_released()
        idle(win)
        self.assertAlmostEqual(win._rendered_angle, win.current.settings.fine_rotation)

    def test_neighbours_are_prepared(self):
        win = self.open_roll(4)
        win.grid.setCurrentRow(1)
        idle(win)
        self.assertIn(win.photos[0].path, win._prefetched)
        self.assertIn(win.photos[2].path, win._prefetched)


class GpuSwitchTest(AppTestCase):
    def test_settings_switch_turns_the_card_off_and_on(self):
        self.addCleanup(app.set_pref, "gpu", True)
        self.addCleanup(setattr, app.engine, "USE_GPU", True)
        win = self.open_frame()
        self.assertEqual(app.engine.USE_GPU, app.pref("gpu"))
        win.pref_gpu.setChecked(False)
        idle(win)
        self.assertFalse(app.engine.USE_GPU)
        self.assertFalse(app.pref("gpu"))
        self.assertIsNotNone(win.preview.source, "the preview must still render on the CPU")
        win.pref_gpu.setChecked(True)
        idle(win)
        self.assertTrue(app.engine.USE_GPU)


class RefineTabsTest(AppTestCase):
    """Refine has Light · Color · Finish; Exposure is the D fader; Whites / Blacks; the colour mixer."""

    def open_refine(self, frames=2):
        self.win = window(1440, 1000)
        self.win.add_rolls([test_roll(frames, name=f"Refine {frames}")])
        idle(self.win)
        self.win.more_btn.setChecked(True)
        pump(0.4)
        return self.win

    def test_tabs_show_their_controls(self):
        win = self.open_refine()
        for i, visible in ((0, win.exposure_slider), (1, win.hsl_sliders["hsl_hue"]), (2, win.sliders["sharpen"])):
            win._refine_tab(i)
            pump(0.4)
            with self.subTest(tab=win.refine_bar.labels[i]):
                self.assertTrue(visible.isVisible())
                self.assertEqual(win.refine_bar.current, i)
                self.assertEqual(win.refine_pages.currentIndex(), i)

    def test_the_section_fits_each_tab_and_the_highlight_glides(self):
        from PySide6.QtTest import QTest
        win = self.open_refine()
        heights = []
        for i in range(3):
            win._refine_tab(i)
            pump(0.5)
            heights.append(win.more_body.height())
            with self.subTest(tab=i):
                self.assertLessEqual(abs(win.more_body.height() - win.more_body.sizeHint().height()), 2,
                                     "the section is not the height of its tab")
        self.assertLess(heights[2], heights[1] - 60, "Finish should be much shorter than Color")
        win._refine_tab(1)                          # on the way the section passes between the two heights
        import time
        seen_h, seen_pos, end = [], [], time.time() + 1.5
        while time.time() < end and (win.more_body.height() != heights[1] or win.refine_bar._pos != 1.0):
            pump(0.01)
            seen_h.append(win.more_body.height())
            seen_pos.append(win.refine_bar._pos)
        self.assertTrue(any(heights[2] < h < heights[1] for h in seen_h), "the height jumped instead of gliding")
        self.assertTrue(any(2.0 > x > 1.0 for x in seen_pos), "the highlight jumped instead of gliding")
        pump(0.5)
        self.assertEqual(win.refine_bar._pos, 1.0)
        bar = win.refine_bar                        # a click on a name chooses it
        QTest.mouseClick(bar, Qt.LeftButton, pos=QPoint(int(bar.width() * 5 / 6), bar.height() // 2))
        pump(0.5)
        self.assertEqual(win.refine_pages.currentIndex(), 2)

    def test_exposure_is_its_own_control(self):
        # Exposure moved the D fader; it is the positive's exposure in stops, D stays where it is
        import numpy as np
        win = self.open_refine()
        st = win.current.settings
        d0 = st.density
        mid = lambda: float(np.median(shown_picture(win).astype(float)))   # noqa: E731
        before = mid()
        win.exposure_slider.set_value(0.5)
        win.on_setting_changed()
        pump(0.6)
        idle(win)
        self.assertAlmostEqual(win.current.settings.exposure, 0.5)
        self.assertEqual(win.current.settings.density, d0, "Exposure moved the density")
        self.assertAlmostEqual(win.steppers["density"].value(), d0)
        self.assertGreater(mid(), before + 10, "Exposure did not brighten the picture")

    def test_whites_and_blacks_move_their_end_of_the_tones(self):
        import numpy as np
        win = self.open_refine()

        def ends():                                    # inside the picture, not the scan's black border
            a = shown_picture(win).astype(float).mean(-1)
            h, w = a.shape
            a = a[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
            return np.percentile(a, 1), np.percentile(a, 97)
        lo0, hi0 = ends()
        win.sliders["white_point"].set_value(0.8)
        win.on_setting_changed()
        pump(0.6)
        idle(win)
        self.assertGreater(ends()[1], hi0 + 3, "Whites did not brighten the light end")
        win.sliders["black_point"].set_value(0.8)
        win.on_setting_changed()
        pump(0.6)
        idle(win)
        self.assertGreater(ends()[0], lo0 + 3, "Blacks did not lift the dark end")

    def test_the_mixer_changes_one_colour_and_remembers_it(self):
        win = self.open_refine()
        st = win.current.settings
        win._refine_tab(1)
        before = shown_picture(win).astype(float)
        win.hsl_dots[5].click()                       # blue
        win.hsl_sliders["hsl_saturation"].set_value(-1.0)
        win._on_hsl("hsl_saturation")
        pump(0.6)
        idle(win)
        self.assertEqual(st.hsl_saturation, (0.0,) * 5 + (-1.0,) + (0.0,) * 2)
        self.assertTrue(win.hsl_dots[5].edited and not win.hsl_dots[0].edited)
        after = shown_picture(win).astype(float)
        spread_before = (before.max(-1) - before.min(-1))
        spread_after = (after.max(-1) - after.min(-1))
        blue = (before[..., 2] > before[..., 0] + 80) & (before[..., 2] > before[..., 1] + 60)
        self.assertGreater(blue.sum(), 50, "no blue patch found to test with")
        self.assertLess(spread_after[blue].mean(), spread_before[blue].mean() * 0.4, "blue was not taken out")
        red = (before[..., 0] > before[..., 1] + 120) & (before[..., 0] > before[..., 2] + 120)
        self.assertLess(abs(spread_after[red].mean() - spread_before[red].mean()), 6, "red changed too")
        win.hsl_dots[0].click()                       # another colour shows its own (untouched) values
        self.assertEqual(win.hsl_sliders["hsl_saturation"].value(), 0.0)
        win.hsl_dots[5].click()
        self.assertEqual(win.hsl_sliders["hsl_saturation"].value(), -1.0)
        win._history_timer.stop()
        win._commit_history()
        win.undo()
        idle(win)
        self.assertEqual(win.current.settings.hsl_saturation, (0.0,) * 8, "undo did not take the mixer back")
        self.assertEqual(win.hsl_sliders["hsl_saturation"].value(), 0.0)

    def test_the_temp_dropper_neutralises_a_clicked_spot(self):
        from PySide6.QtCore import QPointF
        import numpy as np
        win = self.open_refine()
        win._refine_tab(1)
        st = win.current.settings
        win.sliders["temp"].set_value(0.7)                # a warm cast to take out
        win.on_setting_changed()
        pump(0.6)
        idle(win)
        win.temp_picker.click()
        self.assertTrue(win.preview.picking and win.temp_picker.property("on"))
        win.toggle_pick_wb()                               # W switches to the grey picker: one waits
        self.assertTrue(win.pick_btn.property("on"))
        self.assertFalse(win.temp_picker.property("on"))
        win.temp_picker.click()
        a = shown_picture(win)
        grey = np.argwhere((np.abs(a.astype(int) - a.mean(-1, keepdims=True)).max(-1) < 40) & (a.mean(-1) > 120))
        y, x = grey[len(grey) // 2]
        win.on_picked(QPointF(x / (a.shape[1] - 1), y / (a.shape[0] - 1)))
        idle(win)
        self.assertFalse(win.preview.picking, "one click per pick")
        self.assertLess(win.current.settings.temp, 0.5, "the dropper did not take the warm cast out")
        win.temp_picker.click()
        win.stop_pick()
        self.assertFalse(win.preview.picking or win.temp_picker.property("on"))
        st.grayscale = True
        win.load_controls(st)
        self.assertTrue(win.wb_box.isHidden(), "Temp and Tint show on B&W")
        st.grayscale = False
        win.load_controls(st)

    def test_the_mixer_dropper_chooses_the_clicked_colour(self):
        from PySide6.QtCore import QPointF
        import numpy as np
        win = self.open_refine()
        win._refine_tab(1)
        a = shown_picture(win)
        for name, band, mask in (("green", 3, (a[..., 1].astype(int) - a[..., 0] > 120) & (a[..., 1].astype(int) - a[..., 2] > 120)),
                                 ("red", 0, (a[..., 0].astype(int) - a[..., 1] > 150) & (a[..., 0].astype(int) - a[..., 2] > 150))):
            y, x = np.argwhere(mask)[mask.sum() // 2]
            win.mixer_picker.click()
            self.assertTrue(win.preview.picking and win.mixer_picker.property("on"))
            win.on_picked(QPointF(x / (a.shape[1] - 1), y / (a.shape[0] - 1)))
            with self.subTest(name):
                self.assertFalse(win.preview.picking, "one click per pick")
                self.assertEqual(win._hsl_band, band)
                self.assertTrue(win.hsl_dots[band].isChecked())
        win.hsl_sliders["hsl_saturation"].set_value(-0.5)   # the sliders now edit the picked colour
        win._on_hsl("hsl_saturation")
        self.assertEqual(win.current.settings.hsl_saturation[0], -0.5)

    def test_sync_options_sit_inside_the_sync_button(self):
        win = self.open_refine()
        split = win.sync_btn.parentWidget()
        self.assertIs(win.sync_gear.parentWidget(), split, "the options arrow is not joined to Sync")
        self.assertEqual(split.objectName(), "syncSplit")
        self.assertLess(win.sync_gear.x() - (win.sync_btn.x() + win.sync_btn.width()), 3)

    def _drag(self, win, slider, value):
        """A real drag on a slider's track to `value`, then the pause that makes it one undo step."""
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        slider.parentWidget().parentWidget().show()
        x = round(slider._x(value))
        y = slider.height() // 2
        QTest.mousePress(slider, Qt.LeftButton, pos=QPoint(round(slider._x(slider.value())), y))
        QTest.mouseMove(slider, QPoint(x, y))
        QTest.mouseRelease(slider, Qt.LeftButton, pos=QPoint(x, y))
        win._history_timer.stop()
        win._commit_history()

    def test_undo_and_redo_take_back_every_refine_slider(self):
        win = self.open_refine()
        for key, slider in win.sliders.items():
            win._refine_tab({"exposure": 0, "highlight_tone": 0, "shadow_tone": 0, "white_point": 0, "black_point": 0,
                             "temp": 1, "tint": 1}.get(key, 2))
            pump(0.05)
            before = getattr(win.current.settings, key)
            target = 0.6 if before < 0.5 else 0.1
            self._drag(win, slider, target)
            with self.subTest(slider=key):
                after = getattr(win.current.settings, key)
                self.assertNotAlmostEqual(after, before, msg="the drag did not change it")
                win.undo()
                self.assertAlmostEqual(getattr(win.current.settings, key), before, msg="Cmd+Z did not take it back")
                self.assertAlmostEqual(slider.value(), before, msg="the slider did not follow the undo")
                win.redo()
                self.assertAlmostEqual(getattr(win.current.settings, key), after, msg="Cmd+Shift+Z did not bring it back")
                self.assertAlmostEqual(slider.value(), after)
                win.undo()

    def test_undo_and_redo_take_back_the_mixer_per_colour(self):
        win = self.open_refine()
        win._refine_tab(1)
        st = lambda: win.current.settings       # noqa: E731
        for band in (0, 5):                      # red, then blue: two steps
            win._hsl_band_chosen(band)
            for key in ("hsl_hue", "hsl_saturation", "hsl_luminance"):
                self._drag(win, win.hsl_sliders[key], 0.5)
        self.assertEqual(st().hsl_saturation[0], 0.5)
        self.assertEqual(st().hsl_saturation[5], 0.5)
        for _ in range(3):                       # back over blue's three edits
            win.undo()
        self.assertEqual((st().hsl_hue[5], st().hsl_saturation[5], st().hsl_luminance[5]), (0.0, 0.0, 0.0))
        self.assertEqual(st().hsl_luminance[0], 0.5, "undo went too far")
        self.assertEqual(win.hsl_sliders["hsl_hue"].value(), 0.0, "the sliders show the colour chosen, undone")
        self.assertFalse(win.hsl_dots[5].edited, "blue still marked as edited after undo")
        win.redo()
        self.assertEqual(st().hsl_hue[5], 0.5)
        self.assertTrue(win.hsl_dots[5].edited)

    def test_undo_takes_back_the_droppers(self):
        from PySide6.QtCore import QPointF
        win = self.open_refine()
        win._refine_tab(1)
        win.temp_picker.click()
        win.on_picked(QPointF(0.5, 0.5))
        idle(win)
        win._history_timer.stop()
        win._commit_history()
        picked = (win.current.settings.temp, win.current.settings.tint)
        self.assertNotEqual(picked, (0.0, 0.0))
        win.undo()
        self.assertEqual((win.current.settings.temp, win.current.settings.tint), (0.0, 0.0))
        self.assertEqual((win.sliders["temp"].value(), win.sliders["tint"].value()), (0.0, 0.0))

    def test_the_print_look_picker_follows_the_film_and_undoes(self):
        win = self.open_refine()
        win._refine_tab(2)
        box = win.paper_box
        names = [box.itemText(i) for i in range(box.count())]
        self.assertEqual(names[:2], ["None", "Lab print"])
        self.assertIn("Frontier", names)
        self.assertEqual(box.currentText(), "Lab print", "a new photo does not show the default look")
        i = box.findData("kodak_endura")
        box.setCurrentIndex(i)
        box.activated.emit(i)
        idle(win)
        win._history_timer.stop()
        win._commit_history()
        self.assertEqual(win.current.settings.paper, "kodak_endura")
        win.undo()
        self.assertEqual(win.current.settings.paper, "lab")
        self.assertEqual(box.currentData(), "lab", "the picker did not follow the undo")
        win.redo()
        self.assertEqual(box.currentData(), "kodak_endura")
        st = win.current.settings                    # B&W film: the B&W papers
        st.grayscale = True
        win.load_controls(st)
        bw = [box.itemData(i) for i in range(box.count())]
        self.assertIn("ilford_mg_rc", bw)
        self.assertNotIn("kodak_endura", bw)
        self.assertEqual(box.currentData(), "neutral", "a colour paper shown on B&W film")
        st.grayscale = False
        win.load_controls(st)

    def test_a_click_on_a_slider_name_or_value_does_nothing(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        win = self.open_refine()
        for key in ("exposure", "highlight_tone"):
            s = win.sliders[key]
            start = s.value()
            QTest.mouseClick(s, Qt.LeftButton, pos=QPoint(10, s.height() // 2))            # on the name
            QTest.mouseClick(s, Qt.LeftButton, pos=QPoint(s.width() - 6, s.height() // 2))  # on the number
            with self.subTest(key):
                self.assertEqual(s.value(), start, "a click off the line moved the slider")
                self.assertEqual(getattr(win.current.settings, key), start)
        s = win.sliders["exposure"]
        self._drag(win, s, 0.5)
        QTest.mouseDClick(s, Qt.LeftButton, pos=QPoint(10, s.height() // 2))               # double-click still resets
        self.assertEqual(s.value(), 0.0)

    def test_the_mixer_hides_for_bw(self):
        win = self.open_refine()
        win._refine_tab(1)
        st = win.current.settings
        st.grayscale = True
        win.load_controls(st)
        self.assertTrue(win.mixer.isHidden())
        self.assertFalse(win.bw_joke.isHidden(), "B&W film should get the joke instead")
        first = win.bw_joke.joke()
        win._refine_tab(0)
        win._refine_tab(1)
        self.assertNotEqual(win.bw_joke.joke(), first, "the joke should change each time")
        st.grayscale = False
        win.load_controls(st)
        self.assertFalse(win.mixer.isHidden())
        self.assertTrue(win.bw_joke.isHidden())


class SharpFullViewTest(AppTestCase):
    """F: in full view the whole picture shows at full resolution, not the preview enlarged."""

    def wait_for(self, cond, seconds=20.0):
        import time
        end = time.time() + seconds
        while time.time() < end and not cond():
            pump(0.05)
        return cond()

    def test_full_view_shows_the_full_resolution_picture(self):
        import support
        folder = support.TMP / "Big scans"
        folder.mkdir(exist_ok=True)
        for i in range(2):
            support.negative(folder / f"big{i}.tif", size=(2400, 3600), exposure=1.0 + 0.2 * i, dust=False)
        win = window(1440, 900)
        win.add_rolls([folder])
        idle(win)
        pv = win.preview
        pv.devicePixelRatioF = lambda: 4.0            # a screen with far more pixels than the preview
        win.toggle_fullscreen()
        pump(0.3)
        win._check_sharp_fit()
        if pv.fit_hires is None:                      # on its way: the loader shows in the middle
            self.assertTrue(pv.centre_loading(), "no loader while the full-resolution picture loads")
            pv.grab()                                 # paints it (and starts it turning)
            self.assertTrue(pv._spin_timer.isActive())
        self.assertTrue(self.wait_for(lambda: pv.fit_hires is not None), "full view kept the low-res preview")
        self.assertFalse(pv.centre_loading(), "the loader stayed over the sharp picture")
        self.assertTrue(self.wait_for(lambda: not pv._spin_timer.isActive(), 2), "the loader kept ticking")
        self.assertGreater(max(pv.fit_hires.width(), pv.fit_hires.height()),
                           max(pv.source.width(), pv.source.height()) * 1.5, "the picture is not full resolution")
        first = win.current
        win.nudge("density", 2)                       # an edit: the stale sharp picture must not show
        self.assertTrue(self.wait_for(lambda: pv.fit_hires is None or win._hires_key == win._settings_key(first)))
        self.assertTrue(self.wait_for(lambda: pv.fit_hires is not None and win._hires_key == win._settings_key(first)),
                        "the edit was not rendered sharp again")
        win.grid.setCurrentRow(1)                     # the next frame
        self.assertTrue(self.wait_for(lambda: win._hires_key is None or win._hires_key[0] == id(win.current)
                                      or pv.fit_hires is None))
        self.assertTrue(self.wait_for(lambda: pv.fit_hires is not None and win._hires_key[0] == id(win.photos[1])),
                        "the next frame did not get its own full-resolution picture")
        win.toggle_fullscreen()
        self.assertIsNone(pv.fit_hires, "leaving full view kept the big picture")
        pv.loading = True                             # a detail render for a zoom, outside full view
        self.assertFalse(pv.centre_loading(), "the full-view loader showed in the normal view")
        pv.loading = False
        del pv.devicePixelRatioF


class RollWhiteBalanceTest(AppTestCase):
    def test_every_frame_of_a_roll_decodes_on_the_rolls_white_balance(self):
        # a camera in auto white balance gave each frame its own: the roll's baseline turned some blue
        from unittest import mock
        with mock.patch.object(app.engine, "roll_white_balance", return_value=(1.78, 1.0, 2.11)) as rwb:
            win = window(1440, 900)
            win.add_rolls([test_roll(3, name="AWB roll")])
            idle(win)
            win._set_roll_stock(win.group, "other-color")     # the analysis waits for the film
            idle(win, 300)
        self.assertTrue(rwb.called)
        self.assertTrue(all(p.settings.raw_wb == (1.78, 1.0, 2.11) for p in win.photos), "a frame kept its own white balance")
        self.assertEqual(win._reset_settings(win.current).raw_wb, (1.78, 1.0, 2.11), "Reset dropped the roll's white balance")


class TopRowAndAccentTest(AppTestCase):
    """The top row lines up with the columns under it; the film stock asks to be chosen; the accent
    is a setting that applies at once."""

    def test_the_top_row_lines_up_with_the_columns(self):
        win = self.open_roll(2)
        x = lambda w: w.mapTo(win, w.rect().topLeft()).x()        # noqa: E731
        self.assertEqual(x(win.tab_plate) + win.tab_plate.width(), x(win.left) + win.left.width(),
                         "the tab bar is not as wide as the rolls column")
        self.assertIs(win.save_btn.parent(), win.tab_plate, "the save icon is not part of the tab bar")
        self.assertEqual(win.export_wrap.width(), win.right_stack.width())
        self.assertEqual(x(win.export_btn), x(win.right_stack) + 11, "Export does not start at the panels' edge")
        self.assertEqual(x(win.export_more) + win.export_more.width(), x(win.right_stack) + win.right_stack.width() - 21)
        centre = x(win.title_box) + win.title_box.width() / 2
        photo = x(win.stack) + win.stack.width() / 2
        self.assertLess(abs(centre - photo), 2, "the file name is not over the photo")
        win.set_tab(3)
        self.assertFalse(win.title_box.isVisible(), "the file name shows on the Settings page")
        win.set_tab(0)
        self.assertTrue(win.title_box.isVisible())

    def test_the_film_stock_asks_to_be_chosen(self):
        win = self.open_roll(2)
        g = win.group
        self.assertTrue(win.stock_box.property("needed"))
        self.assertTrue(g.card._needs_film)
        self.assertIn(app.FILM_NEEDED, g.label()[1])
        gold = next(d["id"] for d in app.engine.film_stocks() if d["displayName"] == "Kodak Gold 200")
        win.stock_box.setCurrentIndex(win.stock_box.findData(gold))
        win.on_stock_changed(0)
        idle(win)
        self.assertFalse(win.stock_box.property("needed"), "the field still asks once a film is chosen")
        self.assertFalse(g.card._needs_film)

    def test_the_accent_applies_at_once_and_is_remembered(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        self.addCleanup(lambda: (app.set_pref("accent", "purple"), app.apply_accent("purple")))
        win = self.open_roll(2)
        win.set_tab(3)
        picker = win.accent_picker
        rect = dict(picker._rects())["gold"]
        QTest.mouseClick(picker, Qt.LeftButton, pos=QPoint(int(rect.center().x()), int(rect.center().y())))
        self.assertEqual(app.pref("accent"), "gold")
        self.assertEqual(app.ACCENT, app.ACCENTS["gold"][1])
        self.assertEqual(app.COLORS["on_accent"], app.ACCENTS["gold"][3], "text on gold must stay readable")
        self.assertIn(app.ACCENTS["gold"][1], app.QApplication.instance().styleSheet())
        dots = app._accent_dots()
        self.assertIn(app.ACCENTS["gold"][1], (app.Path(dots) / "check-dot.svg").read_text())
        self.assertEqual(set(app.ACCENTS), {"purple", "blue", "gold", "teal"})


class FilmFirstTest(AppTestCase):
    """The film stock (or "No film stock") is chosen before anything is measured: the roll waits for
    it, says so, and cannot be exported; changing it while an analysis runs starts a new one at once."""

    def test_the_roll_waits_for_its_film(self):
        win = self.open_roll(3)
        g = win.group
        self.assertFalse(g.analyzing or g.analyzed, "the roll was analysed before its film was chosen")
        self.assertTrue(win.stock_box.property("needed"), "the film stock field does not ask to be chosen")
        bar = win.roll_progress
        self.assertTrue(bar.isVisible() and bar.waiting)
        self.assertIn("choose the film stock", bar.label.text())
        win.open_export()
        self.assertIn("Choose the film stock first", win.statusBar().currentMessage())
        win._set_roll_stock(g, "other-color")
        idle(win, 300)
        self.assertTrue(g.analyzed)
        self.assertFalse(bar.waiting)
        self.assertEqual(bar.label.palette().color(bar.label.foregroundRole()).name(), app.COLORS["text"],
                         "the analysis line is not in the normal text grey")
        win._set_roll_stock(g, "")                    # taken back: waits again
        self.assertFalse(g.analyzed or g.analyzing)
        self.assertTrue(bar.waiting)

    def test_no_film_stock_is_a_choice_too(self):
        win = self.open_roll(3)
        g = win.group
        box = win.stock_box
        box.setCurrentIndex(box.findData(""))          # "No film stock", picked on purpose
        win.on_stock_changed(box.currentIndex())
        idle(win, 300)
        self.assertTrue(g.no_stock and g.film_chosen)
        self.assertTrue(g.analyzed, "choosing No film stock did not let the roll be analysed")
        self.assertFalse(box.property("needed"))
        self.assertIn("No film stock", g.label()[1])
        self.assertEqual(box.lineEdit().text(), "No film stock")
        session = win._session()["groups"][0]
        self.assertTrue(session["no_stock"], "the choice is not kept with the session")
        gold = next(d["id"] for d in app.engine.film_stocks() if d["displayName"] == "Kodak Gold 200")
        box.setCurrentIndex(box.findData(gold))         # a film after all
        win.on_stock_changed(box.currentIndex())
        idle(win, 300)
        self.assertFalse(g.no_stock)
        self.assertEqual(g.stock_id, gold)

    def test_each_roll_keeps_its_own_film(self):
        self.win = win = window()
        win.add_rolls([test_roll(2, name="Roll A"), test_roll(2, name="Roll B")])
        idle(win)
        a, b = win.groups[0], win.groups[1]
        win.show_group(a)
        win._set_roll_stock(a, "other-color")
        idle(win, 300)
        win.show_group(b)
        self.assertTrue(win.stock_box.property("needed"), "the second roll took the first roll's film")
        self.assertIn("choose the film stock", win.roll_progress.label.text())
        self.assertFalse(b.analyzed)
        win.show_group(a)
        self.assertFalse(win.stock_box.property("needed"))
        self.assertEqual(win.stock_box.currentData(), "other-color")
        self.assertTrue(a.analyzed)

    def test_a_new_film_interrupts_the_running_analysis(self):
        win = self.open_roll(4)
        g = win.group
        done = []
        real = win.roll_progress.show_done
        win.roll_progress.show_done = lambda name: (done.append(name), real(name))
        payloads = []
        win.signals.roll.connect(lambda grp, r: payloads.append(r[0]))
        gold = next(d["id"] for d in app.engine.film_stocks() if d["displayName"] == "Kodak Gold 200")
        win._set_roll_stock(g, "other-color")
        first = g.analysis_id
        win._set_roll_stock(g, gold)                  # while the first one runs
        self.assertEqual(g.analysis_id, first + 1)
        self.assertTrue(g.analyzing)
        idle(win, 300)
        self.assertTrue(g.analyzed)
        self.assertEqual(done, [g.name], "the interrupted analysis also reported")
        self.assertIn(first + 1, payloads)
        self.assertTrue(all(p.settings.film == "Kodak Gold 200 (approx)" for p in g.photos))
        before = [p.settings.roll_floors for p in g.photos]
        win.on_roll_analyzed(g, (first, ((0, 0, 0),) * 5 + ([], None)))     # a late, stale result
        self.assertEqual([p.settings.roll_floors for p in g.photos], before, "a stale analysis was applied")

    def test_other_stocks_let_an_unknown_film_through(self):
        ids = [d["id"] for d in app.engine.film_stocks()]
        self.assertEqual(ids[-2:], ["other-color", "other-bw"])
        self.assertEqual(app.engine.crosstalk_for_stock(app.engine.stock_by_id("other-color")), "Generic C41")


class RollProgressTest(AppTestCase):
    """The foot of the window shows a roll's analysis: "analysing… N %" over a bar that fills, then
    "roll analysed"."""

    def test_the_progress_line_follows_the_analysis(self):
        win = window(1440, 900)
        seen = []
        win.signals.roll_progress.connect(lambda g, f: seen.append(f))
        win.add_rolls([test_roll(4, name="Progress roll")])
        idle(win)
        g = win.group
        win._set_roll_stock(g, "other-color")
        idle(win, 300)
        self.assertTrue(g.analyzed)
        self.assertTrue(seen, "the analysis said nothing on the way")
        self.assertTrue(all(0.0 <= f <= 1.0 for f in seen))
        self.assertGreaterEqual(max(seen), 0.75, "the progress never got near the end")
        bar = win.roll_progress
        self.assertTrue(bar.isVisible())
        self.assertEqual(bar.label.text(), "Progress roll: roll analysed")
        pump(0.5)
        self.assertAlmostEqual(bar.value(), 1.0, places=2, msg="the bar did not fill when done")
        self.assertTrue(bar._hide.isActive(), "the finished line never goes away")
        g.analyzing, g.progress = True, 0.42          # while one runs
        win._show_roll_progress()
        self.assertEqual(bar.label.text(), "Progress roll: analysing…  42 %")
        self.assertFalse(bar._hide.isActive(), "a running analysis would hide")
        self.assertLess(bar.value(), 0.5, "a new analysis did not start the bar from empty")
        g.analyzing = False


class SessionSaveTest(AppTestCase):
    """The save icon keeps the session; quitting asks to save; the next start offers it back.
    Only a crash leaves work behind without a save."""

    def setUp(self):
        super().setUp()
        import support
        for name in ("session.json", "recovery.json"):
            (support.TMP / name).unlink(missing_ok=True)
        self.addCleanup(lambda: [(support.TMP / n).unlink(missing_ok=True) for n in ("session.json", "recovery.json")])

    def opened(self, name):
        win = window(1440, 900)
        win.add_rolls([test_roll(2, name=name)])
        idle(win)
        return win

    def quit(self, win, answer):
        from PySide6.QtGui import QCloseEvent
        asked = []
        win._ask_quit = lambda text, detail, can_save: asked.append((text, can_save)) or answer
        ev = QCloseEvent()
        win.closeEvent(ev)
        return ev.isAccepted(), asked

    def test_the_save_icon_saves_and_marks_unsaved_changes(self):
        win = self.opened("Save me")
        self.assertTrue(win.session_dirty(), "a new roll counts as unsaved")
        win._commit_history()
        self.assertTrue(win.save_btn.dirty)
        win.save_btn.click()
        self.assertTrue(win._session_path().exists())
        self.assertFalse(win.session_dirty())
        self.assertFalse(win.save_btn.dirty, "the dot stayed after saving")
        win.nudge("density", 2)
        win._history_timer.stop()
        win._commit_history()
        self.assertTrue(win.save_btn.dirty, "an edit did not mark the session unsaved")

    def test_quitting_asks_to_save_and_each_answer_does_what_it_says(self):
        win = self.opened("Quit roll")
        accepted, asked = self.quit(win, "cancel")
        self.assertFalse(accepted, "Cancel quit anyway")
        self.assertEqual(asked, [("Save your session before quitting?", True)])
        self.assertFalse(win._session_path().exists())
        accepted, _ = self.quit(win, "quit")
        self.assertTrue(accepted)
        self.assertFalse(win._session_path().exists(), "Quit Without Saving saved")
        win._recovery_path().write_text("{}")
        accepted, _ = self.quit(win, "save")
        self.assertTrue(accepted)
        self.assertTrue(win._session_path().exists(), "Save and Quit did not save")
        self.assertFalse(win._recovery_path().exists(), "a normal quit left a recovery file")
        accepted, asked = self.quit(win, "cancel")    # saved, nothing running: no question at all
        self.assertTrue(accepted)
        self.assertEqual(asked, [])

    def test_the_next_start_offers_the_saved_session(self):
        win = self.opened("Next start")
        win.nudge("density", 3)
        win._history_timer.stop()
        win._commit_history()
        density = win.current.settings.density
        win.save_session()
        paths = [p.path for p in win.all_photos()]
        asked = []
        fresh = window(1440, 900)
        fresh.offer_restore(ask=lambda text, detail: asked.append(text) or True)
        idle(fresh)
        self.assertEqual(asked, ["Restore your saved session?"])
        self.assertEqual([p.path for p in fresh.all_photos()], paths)
        self.assertAlmostEqual(fresh.photos[0].settings.density, density)
        self.assertFalse(fresh.session_dirty(), "a restored session counts as unsaved")
        other = window(1440, 900)                     # Start Fresh: nothing opens
        other.offer_restore(ask=lambda text, detail: False)
        self.assertEqual(other.all_photos(), [])

    def test_nothing_saved_means_no_question(self):
        win = window(1440, 900)
        win.offer_restore(ask=lambda text, detail: self.fail("asked with nothing saved"))
        self.assertEqual(win.all_photos(), [])

    def test_a_crash_offers_its_own_work_first(self):
        import json
        win = self.opened("Crash roll")
        win.save_session()
        win.nudge("density", 4)                       # work after the save, then a crash
        win._history_timer.stop()
        win._commit_history()
        data = win._session()
        data["pid"] = 0
        win._recovery_path().write_text(json.dumps(data))
        crashed_density = win.current.settings.density
        asked = []
        fresh = window(1440, 900)
        fresh.offer_restore(ask=lambda text, detail: asked.append(text) or True)
        idle(fresh)
        self.assertEqual(len(asked), 1, "after restoring the crash it should not offer the save too")
        self.assertIn("did not quit normally", asked[0])
        self.assertAlmostEqual(fresh.photos[0].settings.density, crashed_density)
        self.assertTrue(fresh.session_dirty(), "the work since the save is not saved yet")


class LightSettingTest(AppTestCase):
    """Settings › Scanning light: one switch for the whole setup, never undone with an edit."""

    def test_the_light_switch_reaches_every_frame_and_stays(self):
        app.set_pref("light", "white")
        self.addCleanup(app.set_pref, "light", "white")
        win = window(1440, 900)
        win.add_rolls([test_roll(3, name="Light roll")])
        idle(win)
        self.assertTrue(all(p.settings.light == "white" for p in win.photos))
        win.nudge("density", 2)                      # an edit before the switch, to undo afterwards
        win._history_timer.stop()
        win._commit_history()
        win.light_tabs.select(1)
        idle(win)
        self.assertEqual(app.pref("light"), "rgb")
        self.assertTrue(all(p.settings.light == "rgb" for p in win.photos), "a frame kept the old light")
        win.undo()                                   # undo takes back the edit, not the light
        self.assertEqual(win.current.settings.light, "rgb")
        win.reset_current()
        idle(win)
        self.assertEqual(win.current.settings.light, "rgb", "Reset forgot the light")
        win.add_rolls([test_roll(1, name="Light roll 2")])
        idle(win)
        self.assertTrue(all(p.settings.light == "rgb" for p in win.photos), "a new frame ignored the setting")
        win.set_light("white")
        self.assertEqual(win.light_tabs.current, 0)
        self.assertTrue(all(p.settings.light == "white" for p in win.photos))


class LogTest(unittest.TestCase):
    def test_errors_go_to_a_log_without_console(self):
        # Windows runs the app with pythonw: no console, so errors go to a log file
        out, err = sys.stdout, sys.stderr
        try:
            sys.stdout = sys.stderr = None
            app._log_without_console()
            print("an error from a job", file=sys.stderr)
            log = sys.stderr
            log.flush()
        finally:
            if sys.stderr not in (out, err):
                sys.stderr.close()
            sys.stdout, sys.stderr = out, err
        text = (support.TMP / "dumbconvert.log").read_text()
        self.assertIn("an error from a job", text)
        self.assertIn(app.APP_NAME, text)


if __name__ == "__main__":
    unittest.main()
