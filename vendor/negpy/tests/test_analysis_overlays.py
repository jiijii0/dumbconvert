"""Clipping Overlay: the masks it washes over the frame, the zone strip's hover that names a
zone, and the wash itself."""

import numpy as np

from negpy.features.exposure.analysis import clip_masks


def test_clip_masks_use_the_histograms_end_bins() -> None:
    rgb = np.full((2, 3, 3), 0.5, dtype=np.float32)
    rgb[0, 0] = (0.001, 0.5, 0.5)  # one channel in the black bin
    rgb[0, 1] = (0.5, 0.999, 0.5)  # one channel in the white bin
    rgb[1, 2] = (1.0 / 256, 0.5, 0.5)  # first value past the black bin

    low, high = clip_masks(rgb)

    assert low.tolist() == [[True, False, False], [False, False, False]]
    assert high.tolist() == [[False, True, False], [False, False, False]]


def test_masks_are_strided_to_the_edge_cap() -> None:
    rgb = np.zeros((300, 1000, 3), dtype=np.float32)

    low, _high = clip_masks(rgb, max_edge=256)

    assert max(low.shape) <= 256
    assert low.all()
