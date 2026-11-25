"""The H&D chart must draw the same knees the engine renders with.

Regression guard for the grade-coupled toe/shoulder: the render path bases its
knees on grade_coupled_shape, and the chart used to pass the raw slider values,
diverging at hard grades.
"""

import numpy as np

from negpy.features.exposure.logic import (
    CharacteristicCurve,
    compute_pivot,
    grade_to_slope,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.process.models import ProcessMode


def _reference_points(config, toe, shoulder):
    slope = grade_to_slope(config.grade, None)
    d_min = EXPOSURE_CONSTANTS["d_min"]
    pivot = compute_pivot(slope, config.density, d_min=d_min)
    curve = CharacteristicCurve(
        contrast=slope,
        pivot=pivot,
        d_min=d_min,
        toe=toe,
        toe_width=config.toe_width,
        shoulder=shoulder,
        shoulder_width=config.shoulder_width,
    )
    return slope, pivot, curve


# (centre, density range, offsets) of a frame's mask plane.
_WIDE_MASK = (0.5, 1.4, (0.0, 0.0, 0.0))


def test_preflash_curve_matches_the_kernel():
    from negpy.features.exposure.logic import apply_characteristic_curve

    slope, pivot, grade = 2.9, 0.21, 90.0
    x = np.linspace(-0.4, 1.0, 64, dtype=np.float32)
    img = np.ascontiguousarray(np.repeat(x[None, :, None], 3, axis=2))
    for preflash in (0.4, 0.9):
        out = apply_characteristic_curve(img, (pivot, slope), (pivot, slope), (pivot, slope), frame_grade=grade, preflash=preflash)
        kernel = -np.log10(np.maximum(np.asarray(out)[0, :, 1], 1e-9))
        curve = CharacteristicCurve(slope, pivot, preflash=preflash, grade=grade)
        np.testing.assert_allclose(kernel, np.asarray(curve(x)).ravel(), atol=1e-4)


def test_print_curve_takes_the_papers_midtone_gamma():
    """Zone tools and tone-limited masks read print_curve; the render uses the paper's own gamma."""
    from negpy.features.exposure.logic import effective_midtone_gamma, print_curve
    from negpy.features.exposure.papers import effective_paper_profile

    config = ExposureConfig(paper_profile="kodak_endura", midtone_gamma=0.1)
    curve = print_curve(config, 2.0, 0.5, ProcessMode.C41)
    paper = effective_paper_profile("kodak_endura", ProcessMode.C41)
    assert curve.midtone_gamma == effective_midtone_gamma(paper, 0.1)
    assert curve.midtone_gamma != effective_midtone_gamma(None, 0.1)
