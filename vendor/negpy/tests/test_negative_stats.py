import unittest

import numpy as np

from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.stats import clipping_row, negative_statistics


def _by_name(rows, name):
    return next(r for r in rows if r.name == name)


class TestNegativeStatistics(unittest.TestCase):
    def _rows(self, dr=1.3, anchor=0.46, lo=0.0, hi=0.0):
        return negative_statistics(dr, anchor, lo, hi)

    def test_density_value_in_negative_row(self):
        self.assertEqual(_by_name(self._rows(dr=1.82), "Negative").value, "Δ1.82")

    def test_exposure_ev_number(self):
        a = EXPOSURE_CONSTANTS["assumed_anchor"]
        # +0.1 normalized at dr 1.3 → +0.1*1.3/0.30103 ≈ +0.43 EV, brighter = +.
        row = _by_name(self._rows(anchor=a + 0.1, dr=1.3), "Exposure")
        self.assertIn("EV", row.value)
        self.assertTrue(row.value.startswith("+"))
        # No density range → no EV number to show.
        self.assertEqual(_by_name(self._rows(anchor=a + 0.1, dr=None), "Exposure").value, "—")

    def test_clipping_warn(self):
        clean = _by_name(self._rows(lo=0.001, hi=0.002), "Clipping")
        self.assertFalse(clean.warn)
        self.assertIn("%", clean.value)
        blown = _by_name(self._rows(lo=0.0, hi=0.05), "Clipping")
        self.assertTrue(blown.warn)
        self.assertEqual((blown.value, blown.detail), ("0.0 · 5.0%", "Shadows 0.0% · highlights 5.0%"))
        # The Metering card has the room for the names.
        self.assertEqual(clipping_row(0.0, 0.05).value, "Sh 0.0% · Hi 5.0%")

    def test_missing_inputs_blank(self):
        rows = negative_statistics(None, None, None, None)
        self.assertTrue(all(r.value == "—" for r in rows))

    def test_every_row_always_present(self):
        self.assertEqual([r.name for r in self._rows()], ["Negative", "Exposure", "Clipping", "Scan clip", "Repair", "Gamut"])

    def test_scan_clip_warns_only_above_threshold(self):
        clean = _by_name(negative_statistics(1.3, 0.46, 0.0, 0.0, scan_clip=(0.001, 0.0, 0.0)), "Scan clip")
        self.assertFalse(clean.warn)
        self.assertEqual(clean.value, "0.1 · 0.0 · 0.0%")
        self.assertTrue(clean.per_channel and clean.detail == "Red 0.1% · green 0.0% · blue 0.0%")
        blown = _by_name(negative_statistics(1.3, 0.46, 0.0, 0.0, scan_clip=(0.031, 0.002, 0.0)), "Scan clip")
        self.assertTrue(blown.warn)
        self.assertEqual(blown.value, "3.1 · 0.2 · 0.0%")

    def test_negative_row_character(self):
        from negpy.features.exposure.logic import default_grade_range

        nominal = default_grade_range()
        # The value is the range alone; the hover names the character.
        self.assertTrue(_by_name(self._rows(dr=nominal), "Negative").detail.startswith("Normal negative. "))
        self.assertTrue(_by_name(self._rows(dr=nominal * 0.5), "Negative").detail.startswith("Flat negative. "))
        self.assertTrue(_by_name(self._rows(dr=nominal * 1.5), "Negative").detail.startswith("Contrasty negative. "))
        with_base = negative_statistics(1.82, None, None, None, density_above_base=(0.12, 0.85, 2.1), base_source="picked")
        self.assertEqual(_by_name(with_base, "Negative").value, "Δ1.82")
        self.assertEqual(_by_name(negative_statistics(None, None, None, None), "Negative").value, "—")

    def test_scan_clip_fraction_measurement(self):
        from negpy.features.exposure.normalization import measure_clip_fractions

        img = np.full((64, 64, 3), 0.5, dtype=np.float32)
        img[:16, :, 0] = 1.0  # top quarter of R at sensor white
        r, g, b = measure_clip_fractions(img)
        self.assertAlmostEqual(r, 0.25, delta=0.01)
        self.assertEqual(g, 0.0)
        self.assertEqual(b, 0.0)


def test_clip_fractions_from_bin_array():
    from negpy.features.exposure.analysis import output_clip_fractions

    # (4, 256) bins (R, G, B, L): 10% of R in the black bin, 20% of G in white.
    buf = np.zeros((4, 256), dtype=np.float32)
    buf[0, 0] = 10.0
    buf[0, 128] = 90.0  # R: 10% shadows clipped
    buf[1, 255] = 20.0
    buf[1, 128] = 80.0  # G: 20% highlights clipped
    buf[2, 128] = 100.0
    buf[3, 128] = 100.0
    lo, hi = output_clip_fractions(buf)
    assert abs(lo - 0.10) < 1e-4
    assert abs(hi - 0.20) < 1e-4


if __name__ == "__main__":
    unittest.main()
