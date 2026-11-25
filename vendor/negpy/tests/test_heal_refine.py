"""A heal lands fully: what still stands out under a stroke after one fill is filled again."""

import numpy as np

from negpy.features.retouch.logic import strokes_to_score
from negpy.services.rendering.image_processor import _refine_heal


def _speck():
    rng = np.random.default_rng(3)
    img = np.clip(0.4 + 0.01 * rng.standard_normal((200, 300, 3)), 0, 1).astype(np.float32)
    yy, xx = np.mgrid[0:200, 0:300]
    img[(yy - 100) ** 2 + (xx - 150) ** 2 < 25] = 0.05
    return img


def test_clean_film_is_left_as_it_is():
    img = np.full((120, 160, 3), 0.4, np.float32)
    stroke = ([[0.5, 0.5]], 40.0, 0.0, 0.0)
    np.testing.assert_array_equal(_refine_heal(img, (stroke,), ()), img)


def test_a_partly_filled_speck_is_filled_again():
    img = _speck()
    half = img.copy()
    half[95:106, 145:156] = (half[95:106, 145:156] + 0.4) / 2      # one fill that landed half way
    stroke = ([[0.5, 0.5]], 160.0, 0.0, 0.0)                        # a brush well wider than the speck
    assert strokes_to_score(half, (stroke,), ()) is not None
    out = _refine_heal(half, (stroke,), ())
    assert abs(float(out[100, 150].mean()) - 0.4) < abs(float(half[100, 150].mean()) - 0.4) / 2
