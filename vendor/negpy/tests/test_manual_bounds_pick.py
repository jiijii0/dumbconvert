import pytest

from negpy.features.exposure.normalization import LogNegativeBounds, resolve_bounds_detailed
from negpy.features.process.models import ProcessConfig

_AUTO = LogNegativeBounds((-2.0, -2.1, -2.2), (-0.3, -0.4, -0.5))
_DMIN = (-0.1, -0.2, -0.6)
_DMAX = (-1.8, -1.9, -2.4)


def _t(b: LogNegativeBounds) -> tuple:
    return tuple(b.floors), tuple(b.ceils)


def _resolve(**kw) -> tuple:
    final, base = resolve_bounds_detailed(ProcessConfig(**kw), lambda: _AUTO)
    return _t(final), _t(base)


@pytest.mark.parametrize(
    ("picks", "expected"),
    [
        ({}, (_AUTO.floors, _AUTO.ceils)),
        ({"manual_dmin": _DMIN}, (_AUTO.floors, _DMIN)),
        ({"manual_dmax": list(_DMAX)}, (_DMAX, _AUTO.ceils)),
        ({"manual_dmin": _DMIN, "manual_dmax": _DMAX}, (_DMAX, _DMIN)),
    ],
)
def test_a_picked_end_replaces_that_end_of_the_analysis(picks, expected):
    final, base = _resolve(**picks)
    assert base == expected
    assert final == expected


def test_roll_average_on_both_axes_wins_over_picks():
    locked = LogNegativeBounds((-1.5, -1.5, -1.5), (-0.2, -0.2, -0.2))
    final, _ = _resolve(
        manual_dmin=_DMIN,
        manual_dmax=_DMAX,
        use_luma_average=True,
        use_color_average=True,
        locked_floors=locked.floors,
        locked_ceils=locked.ceils,
    )
    assert final == _t(locked)


def _controller():
    from dataclasses import replace
    from unittest.mock import MagicMock

    from negpy.desktop.controller import AppController
    from negpy.desktop.session import AppState

    ctrl = AppController.__new__(AppController)
    ctrl.state = AppState()
    ctrl.state.config = replace(ctrl.state.config, process=replace(ctrl.state.config.process, local_floors=(-1.0,) * 3, lock_bounds=True))
    ctrl.state.last_metrics = {"final_bounds": LogNegativeBounds((-1.0, -1.0, -1.0), (0.0, 0.0, 0.0))}
    ctrl.set_status = MagicMock()
    ctrl.set_active_tool = MagicMock()
    ctrl.tool_sync_requested = MagicMock()
    ctrl.apply_config = lambda config, **_k: setattr(ctrl.state, "config", config)
    return ctrl


def _click(ctrl, field, value):
    from unittest.mock import MagicMock

    ctrl._sample_normalized_patch = MagicMock(return_value=[value] * 8 + [(1.0, 1.0, 1.0)])
    ctrl._handle_bounds_pick(0.5, 0.5, field)


def test_a_pick_location_never_reaches_the_thumbnail():
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.assets.thumbnail_fingerprint import thumbnail_fingerprint

    cfg = WorkspaceConfig()
    moved = replace(cfg, process=replace(cfg.process, manual_dmin_at=(0.1, 0.2)))

    def fp(c):
        return thumbnail_fingerprint(c, workspace_color_space="Adobe RGB", input_icc_path=None)

    assert fp(cfg) == fp(moved)
