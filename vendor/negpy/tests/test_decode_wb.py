"""A roll pins its camera RAWs to one white balance (ProcessConfig.decode_wb)."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.logic import decode_wb, linear_raw_token
from negpy.services.rendering.image_processor import ImageProcessor


def _config(**process) -> WorkspaceConfig:
    c = WorkspaceConfig()
    return replace(c, process=replace(c.process, **process))


def test_the_pin_reaches_the_raw_decode():
    seen = {}

    def fake_decode(self, path, linear_raw, **kw):
        seen["wb"] = kw.get("wb_override")
        return np.zeros((4, 4, 3), np.uint16), {}

    with patch.object(ImageProcessor, "_decode_sensor_rgb", fake_decode):
        ip = ImageProcessor()
        try:
            ip._decode_oriented_f32("x.RW2", _config(decode_wb=(1.8, 1.0, 2.1)))
        except Exception:
            pass                                          # only the decode call matters here
    assert seen["wb"] == (1.8, 1.0, 2.1)


def test_the_pin_keys_the_decode_and_a_linear_raw_ignores_it():
    plain, pinned = _config(), _config(decode_wb=(1.8, 1.0, 2.1))
    assert ImageProcessor._source_key("x.RW2", plain, False, True) != ImageProcessor._source_key("x.RW2", pinned, False, True)
    assert linear_raw_token(plain.process) != linear_raw_token(pinned.process)
    assert decode_wb(_config(decode_wb=(1.8, 1.0, 2.1), linear_raw=True).process) is None


def test_a_saved_list_comes_back_as_a_tuple():
    assert _config(decode_wb=[1.8, 1, 2.1]).process.decode_wb == (1.8, 1.0, 2.1)
