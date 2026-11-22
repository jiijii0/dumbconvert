"""The conversion engine (simpleneg/engine.py) on generated test negatives, without the window."""

import unittest

import numpy as np
from PIL import Image

from support import DUST, TMP, engine, test_file


def render(path, st, size=1200, **kw):
    return engine.preview(path, st, size, **kw).astype(int)


def shift(a, b) -> tuple[int, int]:
    """How far picture b sits from picture a, in pixels (phase correlation)."""
    fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = np.fft.ifft2(fa * fb.conj() / (np.abs(fa * fb.conj()) + 1e-9)).real
    y, x = np.unravel_index(r.argmax(), r.shape)
    h, w = r.shape
    return (x if x < w // 2 else x - w, y if y < h // 2 else y - h)


class HealTest(unittest.TestCase):
    """Healing repairs the dust and leaves the rest of the picture visibly as it was. (NegPy measures the
    cleaned frame again, so the rest may move by a level or so: invisible, and allowed here.)"""

    @staticmethod
    def _visible_change(before, after, keep_out) -> tuple[float, float]:
        d = np.abs(before - after).max(-1)
        mask = np.ones(d.shape, bool)
        h, w = d.shape
        for y0, y1, x0, x1 in keep_out:
            mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
        return float(d[mask].mean()), float(np.percentile(d[mask], 99.9))

    def test_heal_spot_repairs_only_the_dust(self):
        f = test_file()
        st = engine.Settings()
        before = render(f, st)
        st.heal_spots = tuple((x, y, 40.0) for x, y in DUST)
        after = render(f, st)
        h, w = before.shape[:2]
        for x, y in DUST:
            yy, xx = int(y * h), int(x * w)
            ring = before[yy, xx + 20]
            self.assertLess(np.abs(after[yy, xx] - ring).max(), np.abs(before[yy, xx] - ring).max() / 3,
                            f"the speck at {(x, y)} was not repaired")
        mean, p999 = self._visible_change(before, after, [(y - .03, y + .03, x - .03, x + .03) for x, y in DUST])
        self.assertLess(mean, 1.5, "healing changed the rest of the picture")
        self.assertLessEqual(p999, 5, "healing changed the rest of the picture")

    def test_clean_area_only_inside_the_shape(self):
        f = test_file()
        st = engine.Settings()
        before = render(f, st)
        st.heal_areas = (((0.35, 0.45), (0.50, 0.45), (0.50, 0.65), (0.35, 0.65)),)   # around the first speck
        after = render(f, st)
        h, w = before.shape[:2]
        x, y = DUST[0]
        self.assertGreater(np.abs(after[int(y * h), int(x * w)] - before[int(y * h), int(x * w)]).max(), 30,
                           "the speck inside the area was not cleaned")
        mean, p999 = self._visible_change(before, after, [(0.42, 0.68, 0.32, 0.53)])
        self.assertLess(mean, 1.5, "cleaning changed the picture outside the shape")
        self.assertLessEqual(p999, 5, "cleaning changed the picture outside the shape")


class AutoCropTest(unittest.TestCase):
    def setUp(self):
        self.f = test_file()
        self.st = engine.Settings(auto_crop=True)
        found = engine.find_auto_crop(self.f, self.st)
        self.assertIsNotNone(found, "auto crop found no border on the test negative")
        self.st.auto_rect, self.st.auto_key = found

    def test_found_rect_crops_the_light_around_the_film(self):
        x0, y0, x1, y1 = self.st.auto_rect
        self.assertGreater(x0, 0.03)
        self.assertGreater(y0, 0.03)
        self.assertLess(x1, 0.97)
        self.assertLess(y1, 0.97)

    def test_draft_and_full_render_frame_the_same(self):
        # Found anew per render they differed by a pixel or two, and the picture twitched (Reddit #6)
        draft, full = render(self.f, self.st, 800), render(self.f, self.st, 1600)
        small = np.asarray(Image.fromarray(full.astype(np.uint8)).resize(draft.shape[1::-1], Image.LANCZOS))
        self.assertLessEqual(max(abs(v) for v in shift(small.mean(-1), draft.mean(-1))), 1)

    def test_key_follows_the_framing(self):
        self.assertEqual(engine.auto_crop_key(self.st), self.st.auto_key)
        for change in ({"rotation": 1}, {"mirror": True}, {"fine_rotation": 1.5}):
            turned = self.st.copy()
            for k, v in change.items():
                setattr(turned, k, v)
            with self.subTest(change=change):
                self.assertNotEqual(engine.auto_crop_key(turned), self.st.auto_key)

    def test_export_matches_the_preview(self):
        out = engine.export(self.f, TMP / "export", self.st, engine.ExportOptions())
        w, h = Image.open(out).size
        shown = render(self.f, self.st, 1600)
        self.assertAlmostEqual(w / h, shown.shape[1] / shown.shape[0], delta=0.004)


class StraightenTest(unittest.TestCase):
    """Straightening a frame with no crop changed its colours at the first 0.1°: NegPy then measured
    the light around the film too, not only the film."""

    @staticmethod
    def _measured(path, st):
        """The film floors and ceils NegPy measures for this render."""
        cfg = engine.to_config(st, path=path)
        ip = engine._processor()
        src, _, _ = ip._load_source_f32(str(path), cfg, fast_decode=True)
        ip.release_source_cache()
        _, metrics = ip.run_pipeline(np.ascontiguousarray(src[::2, ::2]), cfg, engine.source_hash(path) + "|test",
                                     max(src.shape[:2]) // 2, prefer_gpu=False, readback_metrics=True)
        b = metrics["log_bounds"]
        return np.array([*b.floors, *b.ceils])

    def test_straightening_keeps_the_colours(self):
        def middle(img):                 # stays where it is: the frame turns about its centre
            h, w = img.shape[:2]
            k = h // 10
            return img[h // 2 - k:h // 2 + k, w // 2 - k:w // 2 + k].reshape(-1, 3).mean(0)

        f = test_file()
        level = middle(render(f, engine.Settings()))
        for angle in (0.1, 2.0):
            with self.subTest(angle=angle):
                diff = np.abs(middle(render(f, engine.Settings(fine_rotation=angle))) - level)
                self.assertLess(diff.max(), 12.0, f"straightening changed the colours: {diff.round(1)}")

    def test_a_straightened_frame_measures_the_film_only(self):
        f = test_file()
        for turn in ({}, {"rotation": 1, "mirror": True}):
            level = self._measured(f, engine.Settings(**turn))
            for angle in (0.1, 2.0, -7.0):
                with self.subTest(turn=turn, angle=angle):
                    tilted = self._measured(f, engine.Settings(fine_rotation=angle, **turn))
                    np.testing.assert_allclose(tilted, level, atol=0.15)

    def test_the_measured_area_lies_inside_the_turned_film(self):
        # on a picture NegPy turned itself: white where the film (off centre, on purpose) is
        from negpy.features.geometry.logic import apply_fine_rotation

        h, w = 400, 600
        film = (0.1, 0.2, 0.7, 0.9)
        picture = np.zeros((h, w, 3), np.float32)
        picture[int(film[1] * h):int(film[3] * h), int(film[0] * w):int(film[2] * w)] = 1.0
        for angle in (-10.0, 3.0, 10.0, 40.0):
            with self.subTest(angle=angle):
                x0, y0, x1, y1 = engine._inside_turned(film, w / h, angle)
                inside = apply_fine_rotation(picture, angle)[int(np.ceil(y0 * h)) + 1:int(y1 * h) - 1,
                                                             int(np.ceil(x0 * w)) + 1:int(x1 * w) - 1]
                self.assertGreater(inside.size, 0)
                self.assertGreater(inside.min(), 0.9, "the measured area reaches outside the film")
        x0, y0, x1, y1 = engine._inside_turned(film, w / h, 3.0)
        self.assertGreater((x1 - x0) * (y1 - y0), 0.8 * 0.6 * 0.7, "far smaller than the film at 3°")

    def test_only_when_nothing_else_says_where_to_measure(self):
        area = (0.3, 0.3, 0.6, 0.6)
        for name, f, st in (("film fills the scan", test_file(border=False), engine.Settings(fine_rotation=2.0)),
                            ("a crop", test_file(), engine.Settings(fine_rotation=2.0, crop=(0.2, 0.2, 0.8, 0.8))),
                            ("not straightened", test_file(), engine.Settings())):
            with self.subTest(name):
                self.assertIsNone(engine.to_config(st, path=f).process.analysis_rect)
        own = engine.to_config(engine.Settings(fine_rotation=2.0, analysis_rect=area), path=test_file())
        self.assertEqual(tuple(own.process.analysis_rect), area)


class RefineTest(unittest.TestCase):
    """Every Refine slider must change the picture (a slider doing nothing was reported)."""

    SLIDERS = (("highlight_tone", 1.0), ("shadow_tone", 1.0), ("sharpen", 1.0), ("glow", 1.0))

    def _check(self, grayscale: bool, sliders):
        f = test_file(dust=False)
        st = engine.Settings(grayscale=grayscale)
        base = render(f, st, 800)
        for field, value in sliders:
            with self.subTest(field=field, bw=grayscale):
                changed = st.copy()
                setattr(changed, field, value)
                self.assertGreater(np.abs(render(f, changed, 800) - base).mean(), 0.3, f"{field} did nothing")

    def test_colour(self):
        self._check(False, self.SLIDERS + (("color_noise", 1.0),))

    def test_bw(self):
        self._check(True, self.SLIDERS)


if __name__ == "__main__":
    unittest.main()
