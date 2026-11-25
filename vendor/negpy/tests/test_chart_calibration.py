import numpy as np
import pytest

from negpy.features.process import chart as ch
from negpy.kernel.image.logic import lab_to_rgb_working, working_oetf_encode

_CELL = 20
_GAP = 6


def _corners_px() -> np.ndarray:
    pitch = _CELL + _GAP
    c0 = _GAP + _CELL / 2
    return np.array([[c0, c0], [c0 + 5 * pitch, c0], [c0 + 5 * pitch, c0 + 3 * pitch], [c0, c0 + 3 * pitch]])


def _chart_image(rgb: np.ndarray, order: np.ndarray = ch.ORIENTATIONS[0]) -> np.ndarray:
    """A 6x4 card of linear working-RGB patches on black, read back in ``order``."""
    pitch = _CELL + _GAP
    img = np.zeros((_GAP + 4 * pitch, _GAP + 6 * pitch, 3), dtype=np.float32)
    for cell, patch in enumerate(order):
        r, c = divmod(cell, ch.COLS)
        img[_GAP + r * pitch : _GAP + r * pitch + _CELL, _GAP + c * pitch : _GAP + c * pitch + _CELL] = rgb[patch]
    return img


def _reference_rgb(chart: ch.ChartSpec) -> np.ndarray:
    return np.clip(lab_to_rgb_working(ch.reference_lab(chart).astype(np.float32)), 1e-4, 1.0)


@pytest.mark.parametrize(
    "lab1, lab2, expected",
    [
        # Sharma, Wu & Dalal (2005) test pairs 1, 7, 17 and 25.
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ],
)
def test_delta_e2000_matches_the_published_pairs(lab1, lab2, expected) -> None:
    assert ch.delta_e2000(np.array([lab1]), np.array([lab2]))[0] == pytest.approx(expected, abs=1e-4)


def test_color_only_delta_e_leaves_lightness_out() -> None:
    ref = np.array([[50.0, 20.0, -10.0], [50.0, 20.0, -10.0]])
    lab = np.array([[70.0, 20.0, -10.0], [80.0, 26.0, -4.0]])
    assert ch.delta_e_color(ref, lab)[0] == 0.0
    assert ch.delta_e_color(ref, lab)[1] == pytest.approx(ch.delta_e2000(ref[1:], np.array([[50.0, 26.0, -4.0]]))[0])


def test_reference_lab_keeps_a_neutral_neutral() -> None:
    spec = ch.ChartSpec("gray", ("g",) * 24, ((50.0, 0.0, 0.0),) * 24, (0,))
    assert np.abs(ch.reference_lab(spec)[0] - (50.0, 0.0, 0.0)).max() < 0.05


def test_charts_are_complete() -> None:
    for chart in ch.CHARTS:
        assert len(chart.patch_names) == len(chart.lab_d50) == ch.ROWS * ch.COLS
        assert all(abs(chart.lab_d50[i][1]) < 2.5 and abs(chart.lab_d50[i][2]) < 3.0 for i in chart.exposure_patches)


def test_patch_centers_span_the_corners_row_major() -> None:
    corners = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 3.0], [0.0, 3.0]])
    centers = ch.patch_centers(corners)
    assert centers.shape == (24, 2)
    assert np.allclose(centers[0], (0, 0)) and np.allclose(centers[5], (5, 0)) and np.allclose(centers[23], (5, 3))
    assert np.allclose(centers[7], (1, 1))


@pytest.mark.parametrize("perm", [(0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0)])
def test_patch_centers_take_the_corners_in_any_order_and_any_way_up(perm) -> None:
    sideways = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 5.0], [0.0, 5.0]])[list(perm)]
    centers = {tuple(np.round(c, 6)) for c in ch.patch_centers(sideways)}
    assert centers == {(float(x), float(y)) for x in range(4) for y in range(6)}


def test_a_sideways_chart_with_crossed_dots_reads_its_colors() -> None:
    chart = ch.COLORCHECKER_24
    img = _chart_image(_reference_rgb(chart))
    width = img.shape[1]
    corners = np.array([(y, width - 1 - x) for x, y in _corners_px()])[[0, 2, 1, 3]]
    enc = working_oetf_encode(np.ascontiguousarray(np.rot90(img)))
    lab = ch.measure_lab(enc, corners, chart, ch.detect_orientation(enc, corners, chart))
    assert ch.delta_e2000(ch.reference_lab(chart), lab).mean() < 1.0


@pytest.mark.parametrize("which", range(4))
def test_orientation_is_found_from_the_colors(which: int) -> None:
    chart = ch.COLORCHECKER_24
    img = working_oetf_encode(_chart_image(_reference_rgb(chart), ch.ORIENTATIONS[which]))
    assert np.array_equal(ch.detect_orientation(img, _corners_px(), chart), ch.ORIENTATIONS[which])


def test_fit_undoes_a_known_density_mix() -> None:
    chart = ch.SPYDERCHECKR_24
    density = -np.log10(_reference_rgb(chart))
    mix = np.array([[0.9, 0.15, -0.05], [-0.1, 0.85, 0.25], [0.05, 0.2, 0.75]])

    def render(matrix: np.ndarray) -> np.ndarray:
        m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        m = m / m.sum(axis=1, keepdims=True)
        out = 10.0 ** -(density @ mix.T @ m.T)
        return working_oetf_encode(_chart_image(np.clip(out, 0.0, 1.0).astype(np.float32), ch.ORIENTATIONS[1]))

    fit = ch.fit_chart_matrix(render, _corners_px(), chart)
    assert np.array_equal(fit.order, ch.ORIENTATIONS[1])
    assert fit.before.mean() > 4.0
    assert fit.after.mean() < 0.4 * fit.before.mean()
    assert fit.after_color.mean() < 0.4 * fit.before_color.mean()
    assert np.all(fit.after_color <= fit.after + 1e-9)
    assert np.allclose(np.reshape(fit.matrix, (3, 3)).diagonal(), 1.0)


def test_fit_stops_when_cancelled() -> None:
    chart = ch.SPYDERCHECKR_24
    img = working_oetf_encode(_chart_image(_reference_rgb(chart)))
    with pytest.raises(ch.FitCancelled):
        ch.fit_chart_matrix(lambda _m: img, _corners_px(), chart, cancelled=lambda: True)
