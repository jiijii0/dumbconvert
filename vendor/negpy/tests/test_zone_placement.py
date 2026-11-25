import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.features.exposure.analysis import encoded_of_zone, zone_of_encoded
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.placement import (
    DENSITY_RANGE,
    KNEE_CANDIDATES,
    ZonePin,
    predicted_zone,
    solve_placement,
)

METRICS = {"norm_density_range": 1.4}


def _pin(val: float, target: float) -> ZonePin:
    return ZonePin(nx=0.5, ny=0.5, val_rgb=(val, val, val), val_luma=val, target_zone=target)


class TestZoneRuler(unittest.TestCase):
    def test_encoded_zone_roundtrip(self):
        enc = np.linspace(0.0, 1.0, 21)
        for e in enc:
            self.assertAlmostEqual(encoded_of_zone(float(zone_of_encoded(e))), float(e), places=6)
        for z in np.linspace(0.0, 10.0, 31):
            self.assertAlmostEqual(float(zone_of_encoded(encoded_of_zone(float(z)))), float(z), places=6)


class TestPlacementSolver(unittest.TestCase):
    def setUp(self):
        self.exposure = ExposureConfig()

    def test_one_pin_lands_on_target(self):
        pin = _pin(0.46, 5.0)
        sol = solve_placement(self.exposure, None, METRICS, [pin])
        self.assertIsNotNone(sol)
        self.assertFalse(sol.clamped)
        self.assertGreaterEqual(sol.fields["density"], DENSITY_RANGE[0])
        self.assertLessEqual(sol.fields["density"], DENSITY_RANGE[1])
        self.assertAlmostEqual(sol.achieved[0], 5.0, delta=0.05)

    def test_one_pin_fields(self):
        sol = solve_placement(self.exposure, None, METRICS, [_pin(0.46, 5.0)])
        self.assertEqual(set(sol.fields), {"density", "auto_exposure"})
        self.assertFalse(sol.fields["auto_exposure"])

    def test_two_pins_land_on_both_targets(self):
        pins = [_pin(0.65, 3.0), _pin(0.30, 8.0)]
        sol = solve_placement(self.exposure, None, METRICS, pins)
        self.assertIsNotNone(sol)
        self.assertEqual(set(sol.fields), {"density", "grade", "auto_exposure", "auto_normalize_contrast"})
        self.assertFalse(sol.fields["auto_exposure"])
        self.assertFalse(sol.fields["auto_normalize_contrast"])
        self.assertEqual(sol.fields["grade"], round(sol.fields["grade"]))
        self.assertGreaterEqual(sol.fields["grade"], EXPOSURE_CONSTANTS["iso_r_min"])
        self.assertLessEqual(sol.fields["grade"], EXPOSURE_CONSTANTS["iso_r_max"])
        self.assertAlmostEqual(sol.achieved[0], 3.0, delta=0.15)
        self.assertAlmostEqual(sol.achieved[1], 8.0, delta=0.15)

    def test_unreachable_target_clamps(self):
        # Zone 0 needs paper density beyond d_max: unreachable at any exposure.
        sol = solve_placement(self.exposure, None, METRICS, [_pin(0.2, 0.0)])
        self.assertTrue(sol.clamped)
        self.assertEqual(sol.fields["density"], DENSITY_RANGE[1])
        self.assertGreater(sol.achieved[0], 0.3)

    def test_split_grade_still_converges(self):
        exposure = replace(self.exposure, shadow_grade=20.0, highlight_grade=-15.0)
        pins = [_pin(0.65, 3.0), _pin(0.30, 8.0)]
        sol = solve_placement(exposure, None, METRICS, pins)
        self.assertIsNotNone(sol)
        self.assertAlmostEqual(sol.achieved[0], 3.0, delta=0.15)
        self.assertAlmostEqual(sol.achieved[1], 8.0, delta=0.15)

    def test_degenerate_two_pins_returns_none(self):
        sol = solve_placement(self.exposure, None, METRICS, [_pin(0.5, 3.0), _pin(0.5, 8.0)])
        self.assertIsNone(sol)

    def test_achieved_recomputed_at_rounded_values(self):
        pins = [_pin(0.65, 3.0), _pin(0.30, 8.0)]
        sol = solve_placement(self.exposure, None, METRICS, pins)
        self.assertEqual(sol.fields["density"], round(sol.fields["density"], 2))
        applied = replace(self.exposure, **sol.fields)
        for i, pin in enumerate(pins):
            self.assertAlmostEqual(sol.achieved[i], predicted_zone(applied, None, METRICS, pin.val_luma), places=6)


class TestKneeSolve(unittest.TestCase):
    """Third pin: Density + Grade from the extremes, plus one knee control for the middle."""

    def setUp(self):
        self.exposure = ExposureConfig()
        self.extremes = [_pin(0.65, 3.0), _pin(0.30, 8.0)]

    def _two_pin_zone_of(self, val: float) -> float:
        """Where `val` lands once the two extremes are placed — the zone a third pin
        starts from, so a test can ask for something a known distance away."""
        two = solve_placement(self.exposure, None, METRICS, self.extremes)
        return predicted_zone(replace(self.exposure, **two.fields), None, METRICS, val)

    def test_three_pins_solve_one_knee_control(self):
        mid = _pin(0.48, self._two_pin_zone_of(0.48) + 0.4)
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, mid])
        self.assertIsNotNone(sol)
        self.assertIn(sol.knee, [c.field for c in KNEE_CANDIDATES])
        self.assertEqual(
            set(sol.fields),
            {"density", "grade", sol.knee, "auto_exposure", "auto_normalize_contrast"},
        )

    def test_solved_knee_stays_inside_its_slider(self):
        mid = _pin(0.48, self._two_pin_zone_of(0.48) + 0.4)
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, mid])
        candidate = next(c for c in KNEE_CANDIDATES if c.field == sol.knee)
        self.assertGreaterEqual(sol.fields[sol.knee], candidate.lo)
        self.assertLessEqual(sol.fields[sol.knee], candidate.hi)

    def test_third_pin_gets_closer_than_a_two_pin_solve_leaves_it(self):
        where = self._two_pin_zone_of(0.48)
        ask = where + 0.7
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, _pin(0.48, ask)])
        self.assertLess(abs(sol.achieved[2] - ask), abs(where - ask))

    def test_the_extremes_still_land_with_a_third_pin_in_play(self):
        mid = _pin(0.48, self._two_pin_zone_of(0.48) + 0.4)
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, mid])
        self.assertAlmostEqual(sol.achieved[0], 3.0, delta=0.2)
        self.assertAlmostEqual(sol.achieved[1], 8.0, delta=0.2)

    def test_a_knee_pin_either_side_of_its_centre_both_converge(self):
        for val in (0.72, 0.36):
            with self.subTest(val=val):
                ask = self._two_pin_zone_of(val) + 0.3
                sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, _pin(val, ask)])
                self.assertIsNotNone(sol)
                self.assertAlmostEqual(sol.achieved[2], ask, delta=0.35)

    def test_a_third_pin_already_on_target_leaves_the_knee_alone(self):
        mid = _pin(0.48, self._two_pin_zone_of(0.48))
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, mid])
        self.assertEqual(sol.knee, "")
        self.assertEqual(set(sol.fields), {"density", "grade", "auto_exposure", "auto_normalize_contrast"})
        self.assertEqual(len(sol.achieved), 3)

    def test_an_unreachable_third_target_reads_as_clamped(self):
        sol = solve_placement(self.exposure, None, METRICS, [*self.extremes, _pin(0.48, 0.0)])
        self.assertTrue(sol.clamped)
        self.assertNotAlmostEqual(sol.achieved[2], 0.0, delta=0.2)

    def test_three_pins_sharing_one_tone_are_unsolvable(self):
        pins = [_pin(0.5, 3.0), _pin(0.5, 5.0), _pin(0.5, 8.0)]
        self.assertIsNone(solve_placement(self.exposure, None, METRICS, pins))

    def test_the_solve_is_bounded_by_the_pass_cap(self):
        import negpy.features.exposure.placement as placement

        calls = 0
        real = placement.predicted_zone

        def counted(*args):
            nonlocal calls
            calls += 1
            return real(*args)

        mid = _pin(0.48, self._two_pin_zone_of(0.48) + 0.4)
        with patch.object(placement, "predicted_zone", counted):
            solve_placement(self.exposure, None, METRICS, [*self.extremes, mid])
        # A 2-pin solve is ~1.5k evaluations; the cap keeps three pins within a small
        # multiple of that rather than a third nested bisection's blow-up.
        self.assertLess(calls, 20_000)


class TestZonePlacementLifecycle(unittest.TestCase):
    def setUp(self):
        from negpy.desktop.controller import AppController
        from negpy.desktop.session import AppState, DesktopSessionManager
        from negpy.services.rendering.preview_manager import PreviewManager

        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)
        self.controller.state.preview_raw = np.empty((8, 8, 3), dtype=np.float32)
        self.controller.state.current_file_hash = "f1"
        self.controller.canvas = MagicMock()
        self.controller.canvas.display_size.return_value = (8, 8)
        # Vertical ramp: a click's ny picks its normalized-log value, so two pins
        # at different heights read two different tones.
        ramp = np.tile(np.linspace(0.15, 0.85, 8, dtype=np.float32).reshape(8, 1, 1), (1, 8, 3))
        self.controller.state.last_metrics["normalized_log"] = ramp
        self.render_tasks: list = []
        self.controller.render_requested.connect(self.render_tasks.append)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.prefetch_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _arm(self, zone: float = 5.0) -> None:
        self.controller.arm_zone_target(zone)

    def _pin(self, nx: float, ny: float) -> None:
        self.controller.handle_canvas_clicked(nx, ny)

    def _place(self, nx: float, ny: float, zone: float = 5.0) -> None:
        self._arm(zone)
        self._pin(nx, ny)


class TestZoneStripArming(unittest.TestCase):
    """The strip is the control: a cell click says which zone the next pin prints as."""

    def _strip(self):
        from negpy.desktop.view.widgets.charts import ZoneStripWidget

        strip = ZoneStripWidget()
        strip.resize(110, 24)
        strip.update_data(np.full(11, 0.1))
        return strip

    def _click(self, strip, x: float):
        from PyQt6.QtCore import QEvent, QPointF, Qt
        from PyQt6.QtGui import QMouseEvent

        strip.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(x, 12.0),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    def _settle(self, strip):
        # A click reports once the double-click interval has passed with no second click.
        strip._click_timer.stop()
        strip._fire_click()

    def _double_click(self, strip, x: float):
        from PyQt6.QtCore import QEvent, QPointF, Qt
        from PyQt6.QtGui import QMouseEvent

        strip.mouseDoubleClickEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonDblClick,
                QPointF(x, 12.0),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )


class TestZonePlacementRows(unittest.TestCase):
    def _widget(self):
        from negpy.desktop.view.widgets.stats import ZonePlacementRows

        return ZonePlacementRows()


class TestZonePlacementConfirmKey(unittest.TestCase):
    """Enter accepts the placement from the canvas, with one pin or two."""

    def _overlay(self, pins: int):
        from negpy.desktop.session import AppState, ToolMode
        from negpy.desktop.view.canvas.overlay import CanvasOverlay
        from negpy.features.exposure.placement import ZonePin

        state = AppState()
        state.zone_pins = [ZonePin(nx=0.5, ny=0.5, val_rgb=(0.5,) * 3, val_luma=0.5, target_zone=5.0) for _ in range(pins)]
        overlay = CanvasOverlay(state)
        overlay.set_tool_mode(ToolMode.ZONE_PLACE)
        return overlay


if __name__ == "__main__":
    unittest.main()
