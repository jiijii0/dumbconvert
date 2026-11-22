"""The window, driven like a person would: keys, clicks, the crop tool, several frames at once.

Each test names the bug report it guards against, so a change that brings one back fails here.
"""

import sys
import time
import unittest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCompleter

import support
from support import DUST, app, close, idle, pump, shown_picture, test_file, test_roll, window


class AppTestCase(unittest.TestCase):
    def tearDown(self):
        if getattr(self, "win", None) is not None:
            close(self.win)
            self.win = None

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
    """Reddit #14: the editing keys were active in the film stock box, so "D" could not be typed."""

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

    def test_adjust_keys_still_work_on_the_photo(self):
        win = self.open_frame()
        win.preview.setFocus()
        d = win.current.settings.density
        QTest.keyPress(win.preview, Qt.Key_Q)
        QTest.keyClick(win.preview, Qt.Key_Up)
        QTest.keyRelease(win.preview, Qt.Key_Q)
        idle(win)
        self.assertGreater(win.current.settings.density, d)


class MouseAdjustTest(AppTestCase):
    """Hold an adjust key (Q W E / A S D) and move the mouse up or down: the same as ↑ / ↓."""

    def setUp(self):
        self.y = 500
        test = self

        class FakeCursor:                    # the offscreen platform has no pointer to move
            @staticmethod
            def pos():
                return QPoint(300, test.y)
        self._cursor = app.QCursor
        app.QCursor = FakeCursor

    def tearDown(self):
        app.QCursor = self._cursor
        super().tearDown()

    def move(self, dy):
        self.y += dy
        pump(0.1)                            # the window polls the pointer while the key is held

    def test_moving_the_mouse_steps_the_held_correction(self):
        win = self.open_frame()
        win.preview.setFocus()
        fader = win.steppers["contrast"]
        start = fader.steps()
        QTest.keyPress(win.preview, Qt.Key_W)
        self.move(-3 * app.MainWindow.MOUSE_ADJUST_PX)                   # up: more contrast
        self.assertEqual(fader.steps(), start + 3)
        self.move(app.MainWindow.MOUSE_ADJUST_PX - 1)                    # less than a step: nothing yet
        self.assertEqual(fader.steps(), start + 3)
        self.move(1 + 4 * app.MainWindow.MOUSE_ADJUST_PX)                # down: less contrast
        self.assertEqual(fader.steps(), start - 2)
        QTest.keyRelease(win.preview, Qt.Key_W)
        self.move(-10 * app.MainWindow.MOUSE_ADJUST_PX)                  # key let go: the mouse is just a mouse
        self.assertEqual(fader.steps(), start - 2)
        self.assertEqual(win.steppers["density"].steps(), 0, "another correction moved")
        idle(win)
        self.assertEqual(win.current.settings.contrast, fader.value())

    def test_mouse_alone_changes_nothing(self):
        win = self.open_frame()
        before = win.current.settings.copy()
        self.move(-20 * app.MainWindow.MOUSE_ADJUST_PX)
        idle(win)
        for f in ("density", "contrast", "saturation", "cyan", "magenta", "yellow"):
            self.assertEqual(getattr(win.current.settings, f), getattr(before, f), f"the mouse moved {f}")


class CropTest(AppTestCase):
    def auto_cropped(self, **kw):
        win = self.open_frame(**kw)
        win.autocrop_cb.setChecked(True)
        idle(win)
        st = win.current.settings
        self.assertIsNotNone(st.auto_rect, "the auto crop was not kept")
        return win, st

    def test_edit_starts_from_the_auto_crop(self):
        # Reddit #8: Edit showed the whole scan with its border, and Apply turned Auto off
        win, st = self.auto_cropped()
        win.toggle_crop()
        x0, y0, x1, y1 = st.auto_rect
        self.assertEqual(tuple(round(v, 4) for v in win.preview.crop.getCoords()),
                         tuple(round(v, 4) for v in (x0, y0, x1, y1)))
        win.toggle_crop(apply=True)
        idle(win)
        self.assertTrue(st.auto_crop, "applying an untouched crop turned Auto crop off")
        self.assertIsNone(st.crop)

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
        # Reddit #9: the crop switched from portrait to landscape
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


class PreviewTest(AppTestCase):
    def test_no_strobe_while_dragging(self):
        # Reddit #4: the preview jumped between the soft draft and the sharp render while a slider moved
        win = self.open_frame()
        shown = []
        original = win.preview.set_image
        win.preview.set_image = lambda pix, draft=False: (shown.append("d" if draft else "F"), original(pix, draft))
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
        # A white balance picked on the current frame (Pick Grey) is that frame's alone; it used to be
        # copied to the other selected frames with the next slider move.
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


class LogTest(unittest.TestCase):
    def test_errors_go_to_a_log_without_console(self):
        # Windows runs the app with pythonw: no console, and errors were lost
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
