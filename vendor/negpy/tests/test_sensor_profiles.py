import os

import numpy as np
import pytest

from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.sensor import SensorProfiles


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "sensor_dir", str(tmp_path))


def test_list_get_and_none(tmp_path):
    _write(
        os.path.join(tmp_path, "my_sensor.toml"),
        'name = "My Sensor"\nmatrix = [[1.0, -0.1, 0.0], [0.0, 1.1, -0.3], [0.0, -0.3, 1.1]]\n',
    )
    assert SensorProfiles.list_profiles() == ["None", "My Sensor"]
    assert SensorProfiles.get_matrix("My Sensor") == [1.0, -0.1, 0.0, 0.0, 1.1, -0.3, 0.0, -0.3, 1.1]
    assert SensorProfiles.get_matrix("None") is None
    assert SensorProfiles.get_matrix("missing") is None
    assert SensorProfiles.is_bundled("None")
    assert not SensorProfiles.is_bundled("My Sensor")


def test_name_falls_back_to_stem(tmp_path):
    _write(os.path.join(tmp_path, "rig_a.toml"), "matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]\n")
    assert "rig_a" in SensorProfiles.list_profiles()


def test_save_round_trips_and_delete(tmp_path):
    matrix = [1.004, -0.02, 0.001, -0.118, 1.01, -0.04, -0.042, -0.149, 1.006]
    path = SensorProfiles.save("Setup A", matrix)
    assert os.path.isfile(path)
    assert path.startswith(str(tmp_path))
    assert SensorProfiles.get_matrix("Setup A") == matrix
    SensorProfiles.delete("Setup A")
    assert SensorProfiles.get_matrix("Setup A") is None
    SensorProfiles.delete("Nonexistent")  # no-op, no raise


def test_malformed_skipped(tmp_path):
    _write(os.path.join(tmp_path, "bad_shape.toml"), "matrix = [[1.0, 0.0], [0.0, 1.0]]\n")
    _write(os.path.join(tmp_path, "bad_toml.toml"), "matrix = [[[not valid\n")
    _write(os.path.join(tmp_path, "no_matrix.toml"), 'name = "x"\n')
    assert SensorProfiles.list_profiles() == ["None"]


def _fake_decode(monkeypatch, captures):
    def fake(self, path):
        return np.tile(np.array(captures[path], dtype=np.float32), (16, 16, 1))

    monkeypatch.setattr("negpy.desktop.view.widgets.sensor_calibration_dialog.SensorCalibrationDialog._decode", fake)


def _sensor_sidebar():
    from unittest.mock import MagicMock

    from negpy.desktop.session import AppState
    from negpy.desktop.view.sidebar.sensor import SensorSidebar

    controller = MagicMock()
    controller.state = AppState()
    return controller, SensorSidebar(controller)


def test_two_names_sharing_a_slug_stay_two_profiles(tmp_path):
    """Names that slugify to the same stem stay separate profiles."""
    SensorProfiles.save("Rig A", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    SensorProfiles.save("Rig-A", [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0])

    assert SensorProfiles.get_matrix("Rig A") == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert SensorProfiles.get_matrix("Rig-A") == [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
