"""The lumen-likeness evidence field (moved from src.candidates; still used by
scripts/label_case.py to shade its 3D view)."""

import numpy as np
import pytest

from src.lumen_evidence import LUMEN_BAND, MAX_BAND_TIGHTENING, adaptive_lumen_band, lumen_likeness


def test_lumen_likeness_rejects_calcium_and_accepts_lumen():
    # the whole point of the band: bone/calcium far above lumen HU must not
    # score higher than lumen itself
    values = np.array([0.0, 0.5, 1.0, 1.3, 2.5, 4.0])
    likeness = lumen_likeness(values)
    assert likeness[0] == 0.0            # background tissue
    assert likeness[2] == pytest.approx(1.0)  # lumen
    assert likeness[3] == pytest.approx(1.0)  # slightly brighter lumen
    assert likeness[4] == 0.0            # calcium
    assert likeness[5] == 0.0            # bone


def test_adaptive_band_leaves_clean_cases_alone():
    band, tightening = adaptive_lumen_band(
        background_noise_hu=40.0, contrast_span_hu=400.0, reference_hu=450.0,
        contrast_threshold_hu=80.0,
    )
    assert tightening == 0.0
    assert band == LUMEN_BAND


def test_adaptive_band_tightens_when_noise_rivals_the_contrast_span():
    band, tightening = adaptive_lumen_band(
        background_noise_hu=163.0, contrast_span_hu=146.0, reference_hu=84.0,
        contrast_threshold_hu=80.0,
    )
    assert tightening == pytest.approx(1.0)
    assert band[0] == pytest.approx(LUMEN_BAND[0] + MAX_BAND_TIGHTENING)
    assert band[1] == pytest.approx(LUMEN_BAND[1] + MAX_BAND_TIGHTENING)
    assert band[2] == LUMEN_BAND[2]
    assert band[3] == LUMEN_BAND[3]


def test_adaptive_band_tightens_for_a_case_that_barely_clears_the_contrast_gate():
    _band, tightening = adaptive_lumen_band(
        background_noise_hu=20.0, contrast_span_hu=200.0, reference_hu=90.0,
        contrast_threshold_hu=80.0,
    )
    assert tightening > 0.5


def test_adaptive_band_never_gates_a_case_into_silence():
    band, tightening = adaptive_lumen_band(
        background_noise_hu=5000.0, contrast_span_hu=10.0, reference_hu=81.0,
        contrast_threshold_hu=80.0,
    )
    assert tightening <= 1.0
    assert band[0] < band[1] < band[2] < band[3]
    assert band[1] < 1.0
