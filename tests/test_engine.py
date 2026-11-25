"""The conversion engine (simpleneg/engine.py) on generated test negatives, without the window."""

import unittest

import numpy as np
from PIL import Image

from support import DUST, TMP, engine, negative, test_file


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


    def test_the_graphics_card_heals_as_well_as_the_cpu(self):
        # The viewer renders on the card at ~1600 px: spots and areas must clean there as on the CPU.
        f = test_file()
        x, y = DUST[0]
        area = (((0.35, 0.45), (0.50, 0.45), (0.50, 0.65), (0.35, 0.65)),)
        for name, st in (("spot", engine.Settings(heal_spots=((x, y, 40.0),))), ("area", engine.Settings(heal_areas=area))):
            left = {}
            for gpu in (False, True):
                plain = engine.preview(f, engine.Settings(), 1600, gpu=gpu).astype(float)
                healed = engine.preview(f, st, 1600, gpu=gpu).astype(float)
                h, w = plain.shape[:2]
                yy, xx = int(y * h), int(x * w)
                ring = plain[yy - 12:yy + 13, xx + 20:xx + 30].reshape(-1, 3).mean(0)
                left[gpu] = float(np.abs(healed[yy - 2:yy + 3, xx - 2:xx + 3].reshape(-1, 3).mean(0) - ring).max())
            with self.subTest(heal=name):
                self.assertLess(left[False], 12, f"the {name} left the speck on the CPU")
                self.assertLess(left[True], 12, f"the {name} left the speck on the graphics card")


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

    def test_steps_in_past_the_rebate(self):
        # Auto crop steps past the film's edge on a dense base, colour and B&W (no orange or bright
        # band left). The picture starts 7 % in from the left, 8 % from the top.
        for name, f, st in (("colour", self.f, engine.Settings()),
                            ("B&W", test_file(bw=True), engine.Settings(grayscale=True)),
                            ("turned", self.f, engine.Settings(rotation=1)),
                            ("straightened", self.f, engine.Settings(fine_rotation=1.5))):
            with self.subTest(name):
                x0, y0, x1, y1 = engine.find_auto_crop(f, st)[0]
                self.assertGreaterEqual(min(x0, y0), 0.065, f"the rebate is still in the crop: {x0:.3f}, {y0:.3f}")
                self.assertLessEqual(max(x1, y1), 0.935, f"the rebate is still in the crop: {x1:.3f}, {y1:.3f}")
                self.assertGreater(min(x1 - x0, y1 - y0), 0.8, "cut into the picture")

    def test_a_film_with_no_rebate_is_not_cut_into(self):
        x0, y0, x1, y1 = engine.find_auto_crop(test_file(border=False), engine.Settings())[0]
        self.assertGreater(min(x1 - x0, y1 - y0), 0.9)

    def test_draft_and_full_render_frame_the_same(self):
        # The draft and the full render frame the picture identically, so it does not twitch
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


class RollTest(unittest.TestCase):
    """With the roll's colour, a bracket's under- and overexposed frames stay grey, and none of them
    is taken for an outlier."""

    STOPS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)

    @staticmethod
    def grey(img):
        # the mid grey patch: (0.30, 0.75) of the picture, inside the scan's 7 % / 8 % border
        h, w = img.shape[:2]
        x, y = int((0.07 + 0.86 * 0.30) * w), int((0.08 + 0.84 * 0.75) * h)
        p = img[y - 6:y + 6, x - 6:x + 6].reshape(-1, 3).astype(float).mean(0)
        return p - p.mean()

    def test_a_bracket_shares_one_colour(self):
        files = [negative(TMP / "bracket" / f"f{i}.tif", exposure=2 ** s, dust=False) for i, s in enumerate(self.STOPS)]
        # The test negatives have no dye crosstalk, so unmixing them would add an error of its own
        # that grows with exposure: this test is about the roll's colour, so it is left off.
        floors, ceils, slope, outliers, axis = engine.analyze_roll([(f, engine.Settings(crosstalk=0.0)) for f in files])
        self.assertFalse(any(outliers), "frames of a bracket were taken for outliers")
        for f, stop in zip(files, self.STOPS):
            st = engine.Settings(crosstalk=0.0, roll_floors=floors, roll_ceils=ceils, roll_slope=slope, roll_axis=axis)
            with self.subTest(stop=stop):
                cast = self.grey(render(f, st, 900))
                self.assertLess(np.abs(cast).max(), 4.0, f"the grey has a cast at {stop:+} EV: {cast.round(1)}")

    def test_an_odd_frame_is_still_an_outlier(self):
        f = [negative(TMP / "roll-odd" / f"f{i}.tif", exposure=0.9 + 0.1 * i, dust=False) for i in range(4)]
        floors = [engine.frame_bounds(p, engine.Settings())[0] for p in f]
        ceils = [engine.frame_bounds(p, engine.Settings())[1] for p in f]
        ceils[2] = tuple(np.array(ceils[2]) + (0.35, 0.0, -0.35))      # one frame under another light
        from negpy.features.exposure.normalization import pool_dye_slopes
        _, _, outliers = pool_dye_slopes(np.array(floors), np.array(ceils))
        self.assertEqual(list(outliers), [False, False, True, False])


class ThumbnailTest(unittest.TestCase):
    """Thumbnails keep the shrunk decode (NegPy keep_small): same picture, and still follow every edit."""

    def test_same_picture_as_a_preview_and_edits_still_show(self):
        f = test_file()
        st = engine.Settings()
        first = engine.thumbnail(f, st, 240)
        np.testing.assert_array_equal(first, engine.preview(f, st, 240))
        st.density = 1.3
        darker = engine.thumbnail(f, st, 240)          # from the kept decode
        np.testing.assert_array_equal(darker, engine.preview(f, st, 240))
        self.assertLess(darker.mean(), first.mean() - 3, "the thumbnail did not follow the edit")


class BorderTest(unittest.TestCase):
    """Export with a white border: exactly that many pixels, white, around the picture."""

    def test_full_resolution_adds_the_border_around(self):
        f = test_file()
        plain = engine.export(f, TMP / "border0", engine.Settings(), engine.ExportOptions(fmt="png", overwrite=True))
        framed = engine.export(f, TMP / "border40", engine.Settings(),
                               engine.ExportOptions(fmt="png", border_px=40, overwrite=True))
        a, b = np.asarray(Image.open(plain)), np.asarray(Image.open(framed))
        self.assertEqual(b.shape[:2], (a.shape[0] + 80, a.shape[1] + 80))
        ring = np.concatenate([b[:40].reshape(-1, b.shape[2]), b[:, -40:].reshape(-1, b.shape[2])])
        self.assertTrue((ring == ring.max()).all(), "the border is not plain white")
        np.testing.assert_array_equal(b[40:-40, 40:-40], a)

    def test_a_set_size_keeps_its_size(self):
        framed = engine.export(test_file(), TMP / "border-le", engine.Settings(),
                               engine.ExportOptions(fmt="png", size="long_edge", long_edge=1200, border_px=30,
                                                    overwrite=True))
        self.assertEqual(max(Image.open(framed).size), 1200)


class SteadyPreviewTest(unittest.TestCase):
    """The quick draft and the sharp render frame the picture alike: switching from one to the other
    must not nudge, squeeze or zoom it (the straighten trim was measured in pixels of each size)."""

    @staticmethod
    def shift(a, b):
        fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
        r = np.abs(np.fft.ifft2(fa * np.conj(fb)))
        y, x = np.unravel_index(np.argmax(r), r.shape)
        h, w = a.shape
        return (y if y < h // 2 else y - h, x if x < w // 2 else x - w)

    def test_draft_and_sharp_render_agree(self):
        f = test_file()
        for name, st in (("plain", engine.Settings()), ("straightened", engine.Settings(fine_rotation=2.3)),
                         ("straightened the other way", engine.Settings(fine_rotation=-4.1)),
                         ("turned and flipped", engine.Settings(rotation=1, mirror=True, fine_rotation=1.2))):
            draft, full = engine.preview(f, st, 800), engine.preview(f, st, 1600)
            with self.subTest(name):
                self.assertLessEqual(abs(full.shape[0] / 2 - draft.shape[0]), 1.0, "the draft is taller or shorter")
                self.assertLessEqual(abs(full.shape[1] / 2 - draft.shape[1]), 1.0, "the draft is wider or narrower")
                small = np.asarray(Image.fromarray(full).resize(draft.shape[1::-1], Image.LANCZOS)).astype(float)
                self.assertEqual(self.shift(small.mean(-1), draft.astype(float).mean(-1)), (0, 0))


class OriginalScanTest(unittest.TestCase):
    def test_the_original_is_the_negative_turned_like_the_frame(self):
        f = test_file()
        neg = engine.original_preview(f, engine.Settings(), 600)
        conv = engine.preview(f, engine.Settings(), 600)
        self.assertEqual(neg.dtype, np.uint8)
        h, w = neg.shape[:2]
        mid = neg[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, 3).mean(0)
        self.assertGreater(mid[0], mid[2] + 10, "the negative's orange mask should show: red above blue")
        # a negative: where the picture is light, the scan is dark (inside the picture, away from the border)
        a = neg[h // 5:4 * h // 5, w // 5:4 * w // 5].mean(-1).ravel().astype(float)
        b = conv[h // 5:4 * h // 5, w // 5:4 * w // 5].mean(-1).ravel().astype(float)
        self.assertLess(np.corrcoef(a, b)[0, 1], -0.5, "the original should be the inverted negative")
        turned = engine.original_preview(f, engine.Settings(rotation=1), 600)
        self.assertEqual(turned.shape[:2], (w, h))


class GpuTest(unittest.TestCase):
    """The viewer renders on the graphics card: the same picture as the CPU (the reference), framing
    must still not change the colours, and a failing card falls back to the CPU."""

    def setUp(self):
        if not engine.gpu_active() or engine._gpu_processor() is None:
            self.skipTest("no usable graphics card here")

    def test_the_card_draws_what_the_cpu_draws(self):
        f = test_file()
        for size in (800, 1600, 3000):
            for name, st in (("plain", engine.Settings()), ("turned", engine.Settings(rotation=1, mirror=True)),
                             ("edited", engine.Settings(contrast=1.2, cyan=1.04, saturation=1.2))):
                c = engine.preview(f, st, size).astype(float)
                g = engine.preview(f, st, size, gpu=True).astype(float)
                with self.subTest(size=size, case=name):
                    self.assertEqual(c.shape, g.shape)
                    d = np.abs(c - g).max(-1)
                    self.assertLess(d.mean(), 0.5)
                    self.assertLess((d > 10).mean(), 0.001, "the card and the CPU disagree on some of the picture")

    def test_framing_does_not_change_colours_on_the_card(self):
        f = test_file()

        def centre(st):
            a = engine.preview(f, st, 1600, gpu=True)
            h, w = a.shape[:2]
            return a[h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40].reshape(-1, 3).astype(float).mean(0)
        base = centre(engine.Settings())
        for kw in ({"rotation": 1}, {"fine_rotation": 2.2}, {"crop": (0.3, 0.3, 0.7, 0.7)}):
            with self.subTest(kw):
                self.assertLess(np.abs(centre(engine.Settings(**kw)) - base).max(), 1.5)

    def test_a_failing_card_falls_back_to_the_cpu(self):
        class Broken:
            def render_display_array(self, *a, **kw):
                raise RuntimeError("device lost")
        real = engine._gpu_processor
        engine._gpu_processor = lambda: Broken()
        try:
            out = engine.preview(test_file(), engine.Settings(), 800, gpu=True)
            self.assertTrue(engine._gpu_failed)
        finally:
            engine._gpu_processor = real
            engine._gpu_failed = False
        np.testing.assert_array_equal(out, engine.preview(test_file(), engine.Settings(), 800))


class SelfTestTest(unittest.TestCase):
    """The packaged app's self-test (every export format read back, contact sheet, heal, clean area,
    border, TIFF codecs, roll analysis, the graphics card) passes from source too, so it stays true."""

    def test_the_self_test_passes(self):
        import contextlib
        import io

        from simpleneg import app
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = app.self_test(str(test_file("selftest", dust=True)))
        failed = [line for line in out.getvalue().splitlines() if "SELFTEST FAIL" in line]
        self.assertEqual(code, 0, "\n".join(failed))
        self.assertGreaterEqual(out.getvalue().count("SELFTEST PASS"), 30)


class MixerTest(unittest.TestCase):
    """The colour mixer runs on the CPU engine: a viewer render with it must not lose it on the card."""

    def test_the_mixer_shows_on_the_viewer_render(self):
        f = test_file()
        st = engine.Settings(hsl_saturation=(0.0,) * 5 + (-1.0,) + (0.0,) * 2)
        plain = engine.preview(f, engine.Settings(), 800, gpu=True).astype(float)
        mixed = engine.preview(f, st, 800, gpu=True).astype(float)
        cpu = engine.preview(f, st, 800).astype(float)
        self.assertLess(np.abs(mixed - cpu).mean(), 0.1, "the card and the CPU disagree")
        self.assertGreater(np.abs(mixed - plain).max(), 50, "the mixer did not reach the viewer")

    def test_exposure_on_card_and_cpu_alike_and_not_density(self):
        f = test_file()
        for exp in (0.5, -0.5):
            st = engine.Settings(exposure=exp)
            g = engine.preview(f, st, 800, gpu=True).astype(float)
            c = engine.preview(f, st, 800).astype(float)
            with self.subTest(exposure=exp):
                self.assertLess(np.abs(g - c).mean(), 0.5, "the card and the CPU disagree")
        plain = engine.preview(f, engine.Settings(), 800).astype(float)
        up = engine.preview(f, engine.Settings(exposure=0.5), 800).astype(float)
        dens = engine.preview(f, engine.Settings(density=0.8), 800).astype(float)
        self.assertGreater(np.median(up), np.median(plain) + 10)
        self.assertGreater(np.abs(up - dens).mean(), 3, "Exposure is just Density again")
        self.assertLessEqual(up.max(), 255)

    def test_the_light_and_colour_sliders_reach_negpy(self):
        st = engine.Settings(highlight_tone=0.1, shadow_tone=0.2, white_point=0.3, black_point=-0.4, temp=0.5, tint=-0.6)
        lab = engine.to_config(st).lab
        self.assertEqual((lab.tone_highlights, lab.tone_shadows, lab.tone_whites, lab.tone_blacks), (0.1, 0.2, 0.3, -0.4))
        self.assertEqual((lab.wb_temp, lab.wb_tint), (0.5, -0.6))

    def test_card_and_cpu_agree_on_every_light_and_colour_slider(self):
        # the graphics card renders these while you drag; the CPU exports: they must look the same
        f = test_file(dust=False)
        z = (0.0,) * 8
        cases = [dict(highlight_tone=v) for v in (1.0, -1.0)] + [dict(shadow_tone=v) for v in (1.0, -1.0)] \
            + [dict(white_point=1.0), dict(black_point=-1.0), dict(temp=1.0, tint=-1.0), dict(temp=-1.0, tint=1.0),
               dict(hsl_hue=(1.0,) + z[1:], hsl_saturation=z[:5] + (-1.0,) + z[6:], hsl_luminance=z[:3] + (-1.0,) + z[4:])]
        for kw in cases:
            st = engine.Settings(**kw)
            g = engine.preview(f, st, 600, gpu=True).astype(float)
            c = engine.preview(f, st, 600).astype(float)
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                self.assertLess(np.abs(g - c).mean(), 0.1)

    def test_tone_sliders_work_like_lightrooms(self):
        f = test_file(dust=False)

        def tones(st):
            a = engine.preview(f, st, 600).astype(float).mean(-1)
            h, w = a.shape
            a = a[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
            return np.percentile(a, 1), np.percentile(a, 50), np.percentile(a, 97)
        lo, mid, hi = tones(engine.Settings())
        for kw, end, sign in ((dict(highlight_tone=1.0), 2, 1), (dict(highlight_tone=-1.0), 2, -1),
                              (dict(shadow_tone=1.0), 0, 1), (dict(shadow_tone=-1.0), 0, -1),
                              (dict(white_point=1.0), 2, 1), (dict(black_point=1.0), 0, 1)):
            t = tones(engine.Settings(**kw))
            with self.subTest(**kw):
                self.assertGreater(sign * (t[end] - (lo, mid, hi)[end]), 2.0, "its end of the tones did not move")

    def test_temp_and_tint_turn_the_picture_and_keep_its_brightness(self):
        f = test_file(dust=False)
        base = engine.preview(f, engine.Settings(), 600).astype(float)
        for kw, more, less in ((dict(temp=1.0), 0, 2), (dict(temp=-1.0), 2, 0), (dict(tint=1.0), 0, 1), (dict(tint=-1.0), 1, 0)):
            d = engine.preview(f, engine.Settings(**kw), 600).astype(float) - base
            with self.subTest(**kw):
                self.assertGreater(d[..., more].mean() - d[..., less].mean(), 8, "it did not turn the colour")
                self.assertLess(abs((d @ np.array([0.2126, 0.7152, 0.0722])).mean()), 3, "it changed the brightness")

    def test_a_picked_colour_finds_its_mixer_band(self):
        for rgb, band in (((220, 40, 40), 0), ((230, 130, 40), 1), ((230, 210, 50), 2), ((60, 180, 70), 3),
                          ((40, 180, 190), 4), ((40, 90, 230), 5), ((140, 70, 210), 6), ((210, 50, 160), 7)):
            with self.subTest(rgb=rgb):
                self.assertEqual(engine.mixer_band(rgb), band)
        self.assertIsNone(engine.mixer_band((128, 128, 128)))

    def test_the_white_balance_dropper_makes_the_spot_grey(self):
        f = test_file(dust=False)
        st = engine.Settings(temp=0.6, tint=-0.4)               # a strong cast to take out
        for _ in range(2):                                       # a second click refines the first
            a = engine.preview(f, st, 600)
            h, w = a.shape[:2]
            spot = a[int(h * 0.45):int(h * 0.55), int(w * 0.45):int(w * 0.55)].reshape(-1, 3).mean(0)
            st.temp, st.tint = engine.neutral_temp_tint(spot, st.temp, st.tint)
        a = engine.preview(f, st, 600).astype(float)
        spot = a[int(h * 0.45):int(h * 0.55), int(w * 0.45):int(w * 0.55)].reshape(-1, 3).mean(0)
        self.assertLess(spot.max() - spot.min(), 6, f"the clicked spot is not grey: {spot}")


class LightTest(unittest.TestCase):
    """Settings › Scanning light: both lights use the film's own dye matrix (no fitted rig
    correction); RGB light adds no extra separation and decodes the raw linear."""

    def test_each_light_gets_its_own_separation(self):
        stock_film = engine.crosstalk_for_stock({"displayName": "Kodak Gold 200", "colorType": "Color Negative"})
        self.assertEqual(stock_film, "Kodak Gold 200 (approx)")
        dyes = engine.CrosstalkProfiles.get_matrix(stock_film)
        white_cfg = engine.to_config(engine.Settings(film=stock_film))
        rgb_cfg = engine.to_config(engine.Settings(film=stock_film, light="rgb"))
        white, rgb = white_cfg.process, rgb_cfg.process
        np.testing.assert_allclose(white.crosstalk_matrix, engine.film_matrix(tuple(dyes)))
        np.testing.assert_allclose(rgb.crosstalk_matrix, engine.film_matrix(tuple(dyes)))
        self.assertAlmostEqual(white_cfg.exposure.dye_separation, 1.1)
        self.assertAlmostEqual(rgb_cfg.exposure.dye_separation, 1.0)
        self.assertFalse(white.linear_raw)
        self.assertTrue(rgb.linear_raw)
        for m in (white.crosstalk_matrix, rgb.crosstalk_matrix):    # rows keep grey grey
            np.testing.assert_allclose(np.array(m).reshape(3, 3).sum(1), 1.0, atol=1e-9)

    def test_the_light_changes_the_conversion(self):
        f = test_file(dust=False)
        film = "Kodak Gold 200 (approx)"
        white = engine.preview(f, engine.Settings(film=film), 500).astype(float)
        rgb = engine.preview(f, engine.Settings(film=film, light="rgb"), 500).astype(float)
        self.assertGreater(np.abs(white - rgb).mean(), 1.0, "RGB light converts like white light")
        g = engine.preview(f, engine.Settings(film=film, light="rgb"), 500, gpu=True).astype(float)
        self.assertLess(np.abs(g - rgb).mean(), 0.5, "the graphics card and the CPU disagree for RGB light")


class PrintLookTest(unittest.TestCase):
    def test_every_print_look_renders_and_changes_the_picture(self):
        f = test_file(dust=False)
        for grayscale in (False, True):
            looks = engine.print_looks(grayscale)
            self.assertEqual(looks[0], ("neutral", "None"))
            base = engine.preview(f, engine.Settings(grayscale=grayscale, paper="neutral"), 300).astype(float)
            for key, name in looks[1:]:
                st = engine.Settings(grayscale=grayscale, paper=key)
                cpu = engine.preview(f, st, 300).astype(float)
                gpu = engine.preview(f, st, 300, gpu=True).astype(float)
                with self.subTest(look=name, bw=grayscale):
                    self.assertGreater(np.abs(cpu - base).mean(), 0.5, "the look did nothing")
                    self.assertLess(np.abs(gpu - cpu).mean(), 1.0, "the card and the CPU disagree")

    def test_a_look_of_the_other_kind_renders_as_none(self):
        f = test_file(dust=False)
        base = engine.preview(f, engine.Settings(grayscale=True), 300)
        wrong = engine.preview(f, engine.Settings(grayscale=True, paper="kodak_endura"), 300)
        np.testing.assert_array_equal(base, wrong)


class StockMatchTest(unittest.TestCase):
    """A catalogue stock finds its own film profile: a speed alone ("400") is not a match."""

    def match(self, name):
        return engine.crosstalk_for_stock({"displayName": name, "stockName": name.split(" ", 1)[1]})

    def test_stocks_find_their_own_profile(self):
        for stock, profile in {
            "Kodak UltraMax 400": "Kodak Ultra Max 400 (approx)",          # not Fujicolor 400 by its speed
            "Kodak Gold 200": "Kodak Gold 200 (approx)",
            "Fujifilm 400": "Fujicolor 400 (approx)",
            "Kodak Ektacolor Pro 400 (formerly Portra 400)": "Kodak Ektacolor Pro 400 / Portra 400 (approx)",
            "Kodak Vision3 250D AHU": "Kodak VISION3 250D 5207/7207 (approx)",
            "CineStill 800T": "Kodak VISION3 500T 5219/7219 (approx)",
            "Silbersalz35 50D": "Kodak VISION3 50D 5203/7203 (approx)",
            "Flic Film Cine Colour 200T": "Kodak VISION3 200T 5213/7213 (approx)",
        }.items():
            with self.subTest(stock=stock):
                self.assertEqual(self.match(stock), profile)

    def test_a_speed_alone_is_not_a_match(self):
        for stock in ("Kodak ColorPlus 200", "Lomography CN 400", "Kodak Kodacolor 100", "CineStill 400D"):
            with self.subTest(stock=stock):
                self.assertEqual(self.match(stock), "Generic C41")


class LabPrintTest(unittest.TestCase):
    """Lab print, the default look: the colours a negative takes on photo paper (engine.PRINT_CHARACTER),
    as colour-mixer values under the photo's own mixer."""

    def test_it_is_the_default_and_listed_right_after_none(self):
        self.assertEqual(engine.Settings().paper, engine.LAB_PRINT)
        self.assertEqual(engine.print_looks(False)[1], (engine.LAB_PRINT, "Lab print"))
        self.assertNotIn(engine.LAB_PRINT, dict(engine.print_looks(True)), "a colour look offered for B&W film")

    def test_it_turns_the_mixer_and_none_takes_it_off(self):
        lab = engine.to_config(engine.Settings())
        none = engine.to_config(engine.Settings(paper="neutral"))
        self.assertEqual(lab.exposure.paper_profile, "neutral", "Lab print is not a NegPy paper")
        self.assertEqual(none.lab.hsl_hue, (0.0,) * 8)
        hue, sat = engine.print_character("Generic C41")
        np.testing.assert_allclose(lab.lab.hsl_hue, hue)
        np.testing.assert_allclose(lab.lab.hsl_saturation, sat)
        self.assertLess(hue[2], -0.2, "yellows do not turn toward amber")
        self.assertLess(sat[3], -0.15, "greens are not quieter")

    def test_the_photos_own_mixer_goes_on_top(self):
        own = engine.Settings(hsl_hue=(0.0, 0.0, 0.5, 0, 0, 0, 0, 0), hsl_saturation=(0.0, 0.0, 0.0, -1.0, 0, 0, 0, 0))
        cfg = engine.to_config(own).lab
        hue, sat = engine.print_character(own.film)
        self.assertAlmostEqual(cfg.hsl_hue[2], 0.5 + hue[2])
        self.assertAlmostEqual(cfg.hsl_saturation[3], -1.0, msg="taking a colour out must still take it out")
        self.assertEqual(own.hsl_hue[2], 0.5, "the look was written into the photo's own sliders")

    def test_each_stock_has_its_own(self):
        gold = engine.print_character("Kodak Gold 200 (approx)")
        fuji = engine.print_character("Fujicolor Superia X-TRA 400 (approx)")
        self.assertLess(gold[0][2], 0, "Kodak yellows turn toward amber")
        self.assertGreater(fuji[0][2], 0, "Fujifilm yellows stay lemon")
        for name in engine.PRINT_CHARACTER:
            self.assertIn(name, engine.films(), f"{name} is not a film DumbConvert offers")

    def test_it_changes_colour_film_only(self):
        f = test_file(dust=False)
        lab = engine.preview(f, engine.Settings(), 300).astype(float)
        none = engine.preview(f, engine.Settings(paper="neutral"), 300).astype(float)
        self.assertGreater(np.abs(lab - none).mean(), 0.5, "Lab print did nothing")
        gpu = engine.preview(f, engine.Settings(), 300, gpu=True).astype(float)
        self.assertLess(np.abs(gpu - lab).mean(), 1.0, "the card and the CPU disagree")
        bw = engine.preview(f, engine.Settings(grayscale=True), 300)
        np.testing.assert_array_equal(bw, engine.preview(f, engine.Settings(grayscale=True, paper="neutral"), 300))


class RollColourTest(unittest.TestCase):
    """Roll mode: one baseline for the roll must not make a frame's colour depend on its exposure,
    nor on the white balance a camera picked for each shot."""

    def test_a_roll_of_brackets_stays_as_grey_as_single_frames(self):
        # Thin frames must not go pink nor dense ones blue: the dye-slope correction holds the grey
        # at the ends of the roll's exposure range as well as at its middle.
        from PIL import Image, ImageCms
        from support import PATCHES, TMP, negative
        T = ImageCms.buildTransform(ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB")
        stops = (-2.0, -1.0, 0.0, 1.0, 2.0)
        files = [negative(TMP / "brackets" / f"b{i}.tif", exposure=2.0 ** s, dust=False) for i, s in enumerate(stops)]

        def grey_error(f, st):
            a = engine.preview(f, st, 700)
            h, w = a.shape[:2]
            m = (int(h * 0.08), int(w * 0.07))
            out = []
            for key in ("grey_dark", "grey_mid", "grey_light"):
                cx, cy = PATCHES[key][1]
                y, x = int(m[0] + cy * (h - 2 * m[0])), int(m[1] + cx * (w - 2 * m[1]))
                p = np.ascontiguousarray(a[y - 5:y + 5, x - 5:x + 5])
                ab = np.asarray(ImageCms.applyTransform(Image.fromarray(p), T))[..., 1:].view(np.int8).astype(float)
                out.append(np.hypot(*ab.reshape(-1, 2).mean(0)))
            return float(np.mean(out))

        floors, ceils, slope, outliers, axis = engine.analyze_roll([(f, engine.Settings()) for f in files])
        self.assertFalse(any(outliers), "brackets of one film, scanned alike, flagged as outliers")
        for s, f in zip(stops, files):
            own = grey_error(f, engine.Settings())
            roll = grey_error(f, engine.Settings(roll_floors=floors, roll_ceils=ceils, roll_slope=slope, roll_axis=axis))
            with self.subTest(stops=s):
                self.assertLess(roll, own + 2.0, f"roll mode drifts at {s:+} stops: {roll:.1f} against {own:.1f}")

    def test_a_roll_pins_its_raws_to_one_white_balance(self):
        cfg = engine.to_config(engine.Settings(raw_wb=(1.78, 1.0, 2.11)))
        self.assertEqual(cfg.process.decode_wb, (1.78, 1.0, 2.11))
        self.assertIsNone(engine.to_config(engine.Settings()).process.decode_wb)
        self.assertIsNone(engine.roll_white_balance([test_file(dust=False)]), "a TIFF scan has no camera white balance")


class RefineTest(unittest.TestCase):
    """Every Refine slider must change the picture."""

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
